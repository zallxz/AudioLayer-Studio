"""FFmpeg / ffprobe wrappers. All calls use argument lists (never a shell)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..core.errors import InvalidMedia, ProcessingFailed
from ..core.schemas import Loudness, MediaInfo

WORKING_SR = 48000


def ffmpeg_bin() -> str:
    b = shutil.which("ffmpeg")
    if not b:
        raise ProcessingFailed("FFmpeg not found on PATH", hint="Install FFmpeg (see README).")
    return b


def ffprobe_bin() -> str:
    b = shutil.which("ffprobe")
    if not b:
        raise ProcessingFailed("ffprobe not found on PATH", hint="Install FFmpeg (see README).")
    return b


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _f(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def probe(path: Path) -> MediaInfo:
    res = run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    if res.returncode != 0:
        raise InvalidMedia(
            f"Could not read media file: {res.stderr.strip()[:300] or 'unknown error'}",
            hint="The file may be corrupted or not a supported audio/video format.",
        )
    data = json.loads(res.stdout or "{}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None and a is None:
        raise InvalidMedia("File contains no audio or video streams")
    fps = None
    if v is not None and v.get("avg_frame_rate") and v["avg_frame_rate"] != "0/0":
        n, _, d = v["avg_frame_rate"].partition("/")
        fps = (_f(n) or 0) / (_f(d) or 1)
    container_duration = _f(fmt.get("duration"))
    v_start = _f(v.get("start_time")) if v else None
    a_start = _f(a.get("start_time")) if a else None
    offset = 0.0
    if v_start is not None and a_start is not None:
        offset = round(a_start - v_start, 6)
    return MediaInfo(
        filename=path.name,
        container=str(fmt.get("format_name", "")),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        has_video=v is not None,
        has_audio=a is not None,
        video_duration=(_f(v.get("duration")) or container_duration) if v else None,
        audio_duration=(_f(a.get("duration")) or container_duration) if a else None,
        sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        channels=int(a["channels"]) if a and a.get("channels") else None,
        audio_codec=a.get("codec_name") if a else None,
        video_codec=v.get("codec_name") if v else None,
        fps=round(fps, 3) if fps else None,
        width=v.get("width") if v else None,
        height=v.get("height") if v else None,
        audio_start_offset=offset,
    )


def extract_audio(src: Path, dst: Path, *, channels: int, sample_rate: int = WORKING_SR) -> None:
    """Decode the first audio stream to a float32 WAV working master.

    ``aresample=async=1`` pads/stretches tiny timestamp gaps so the decoded audio stays
    aligned with the container timeline (prevents A/V drift with generated media).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(src),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", "aresample=async=1:first_pts=0",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "pcm_f32le", "-rf64", "auto", str(dst),
    ]
    res = run(cmd)
    if res.returncode != 0 or not dst.exists():
        raise InvalidMedia(f"Audio extraction failed: {res.stderr.strip()[:400]}")


def make_silence(dst: Path, duration: float, *, channels: int = 2, sample_rate: int = WORKING_SR) -> None:
    layout = "stereo" if channels == 2 else "mono"
    res = run([
        ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl={layout}", "-t", f"{duration:.6f}",
        "-c:a", "pcm_f32le", str(dst),
    ])
    if res.returncode != 0:
        raise ProcessingFailed(f"Could not create silent track: {res.stderr[:300]}")


_EBUR = re.compile(r"I:\s+(-?[\d.]+|-inf) LUFS.*?LRA:\s+(-?[\d.]+) LU.*?Peak:\s+(-?[\d.]+|-inf) dBFS", re.S)


def measure_loudness(path: Path) -> Loudness:
    res = run([
        ffmpeg_bin(), "-nostdin", "-v", "info", "-i", str(path),
        "-af", "ebur128=peak=true", "-f", "null", "-",
    ])
    text = res.stderr
    idx = text.rfind("Summary:")
    m = _EBUR.search(text[idx:] if idx >= 0 else text)
    if not m:
        return Loudness()

    def num(s: str) -> float | None:
        return None if "inf" in s else float(s)

    return Loudness(integrated_lufs=num(m.group(1)), lra=num(m.group(2)), true_peak_dbtp=num(m.group(3)))


LOUDNESS_PRESETS: dict[str, tuple[float, float]] = {
    # name: (integrated LUFS, true-peak dBTP)
    "youtube": (-14.0, -1.0),
    "tiktok": (-14.0, -1.0),
    "podcast": (-16.0, -1.0),
    "broadcast": (-23.0, -1.0),
}


def loudnorm(src: Path, dst: Path, preset: str) -> dict[str, Any]:
    """Two-pass EBU R128 loudness normalisation via ffmpeg's loudnorm filter."""
    target_i, target_tp = LOUDNESS_PRESETS[preset]
    base = f"loudnorm=I={target_i}:TP={target_tp}:LRA=11"
    first = run([ffmpeg_bin(), "-nostdin", "-v", "info", "-i", str(src), "-af", base + ":print_format=json", "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", first.stderr, re.S)
    if not m:
        raise ProcessingFailed("Loudness measurement failed")
    meas = json.loads(m.group(0))
    second_filter = (
        f"{base}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:"
        f"measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:"
        f"offset={meas['target_offset']}:linear=true,aresample={WORKING_SR}"
    )
    res = run([ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(src), "-af", second_filter,
               "-c:a", "pcm_f32le", str(dst)])
    if res.returncode != 0:
        raise ProcessingFailed(f"Loudness normalisation failed: {res.stderr[:300]}")
    return {"preset": preset, "target_lufs": target_i, "target_tp": target_tp, "measured": meas}


def encode(src_wav: Path, dst: Path, fmt: str) -> None:
    """Encode a float WAV to the delivery format (sample-accurate, no trimming)."""
    if fmt == "wav":
        codec = ["-c:a", "pcm_s24le", "-ar", str(WORKING_SR)]
    elif fmt == "flac":
        codec = ["-c:a", "flac", "-sample_fmt", "s32", "-ar", str(WORKING_SR)]
    elif fmt == "mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "320k", "-ar", str(WORKING_SR)]
    else:
        raise ValueError(fmt)
    res = run([ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(src_wav), *codec, str(dst)])
    if res.returncode != 0:
        raise ProcessingFailed(f"Encoding {fmt} failed: {res.stderr[:300]}")


def mux_video(video_src: Path, audio_wav: Path, dst: Path, *, audio_offset: float = 0.0) -> None:
    """Copy the original video stream untouched and attach new audio."""
    ext = dst.suffix.lower()
    acodec = ["-c:a", "flac"] if ext == ".mkv" else ["-c:a", "aac", "-b:a", "320k"]
    cmd = [ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-i", str(video_src)]
    if abs(audio_offset) > 1e-4:
        cmd += ["-itsoffset", f"{audio_offset:.6f}"]
    cmd += ["-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *acodec]
    if ext in {".mp4", ".mov"}:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(dst)]
    res = run(cmd)
    if res.returncode != 0:
        raise ProcessingFailed(f"Video remux failed: {res.stderr[:400]}")
