#!/usr/bin/env python3
"""Build a synthetic "AI-generated video" with KNOWN ground-truth stems.

Mixes real speech (LibriSpeech sample), background music (Kevin MacLeod, CC BY) and birdsong from
librosa's example-data server with synthesized SFX (impacts, whoosh) and technical noise
(hiss + 60 Hz hum), then muxes it with a test-pattern video. Ground-truth stems are written
next to the video so separation quality can be *measured* (see scripts/evaluate.py).

Usage: python scripts/make_demo_media.py OUT_DIR [--seconds 20] [--no-noise]
Source audio is downloaded on demand and is not committed to the repository.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


def load_example(key: str, seconds: float, offset: float = 0.0) -> np.ndarray:
    import librosa

    y, _ = librosa.load(librosa.example(key), sr=SR, mono=True, offset=offset, duration=seconds)
    return y.astype(np.float32)


def place(dst: np.ndarray, src: np.ndarray, at: float, rms_db: float) -> None:
    """Add ``src`` at time ``at`` normalised to an RMS level of ``rms_db`` dBFS (active part)."""
    i = int(at * SR)
    n = min(len(src), len(dst) - i)
    if n <= 0:
        return
    s = src[:n]
    active = s[np.abs(s) > 0.02 * np.max(np.abs(s))] if np.any(s) else s
    rms = float(np.sqrt(np.mean(active**2))) if len(active) else 1.0
    dst[i : i + n] += s * (10 ** (rms_db / 20) / max(rms, 1e-9))


def impact(rng: np.random.Generator) -> np.ndarray:
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 9)
    crack = rng.standard_normal(n) * np.exp(-t * 35)
    return (0.8 * body + 0.5 * crack).astype(np.float32)


def whoosh(rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    n = int(1.2 * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    for k in range(0, n, 2400):
        fc = 300 + 5000 * (k / n)
        sos = butter(2, [fc * 0.7, min(fc * 1.3, SR / 2 - 100)], btype="band", fs=SR, output="sos")
        out[k : k + 2400] = sosfilt(sos, noise[k : k + 2400])
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    return (out * env * 3).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    n = int(a.seconds * SR)
    rng = np.random.default_rng(7)

    dialogue = np.zeros(n, np.float32)
    place(dialogue, load_example("libri1", 8.0), 3.0, -20)
    bgm = np.zeros(n, np.float32)
    place(bgm, load_example("nutcracker", a.seconds, 5.0), 0.0, -26)
    sfx = np.zeros(n, np.float32)
    place(sfx, impact(rng), 1.0, -16)
    place(sfx, load_example("robin", 3.0), 13.0, -24)  # bird song = environmental SFX/ambience
    place(sfx, whoosh(rng), 17.5, -22)
    noise = np.zeros(n, np.float32)
    if not a.no_noise:
        tt = np.arange(n) / SR
        noise += (rng.standard_normal(n) * 10 ** (-56 / 20)).astype(np.float32)
        noise += (0.004 * (np.sin(2 * np.pi * 60 * tt) + 0.5 * np.sin(2 * np.pi * 120 * tt))).astype(np.float32)
    stems = {"dialogue": dialogue, "bgm": bgm, "sfx": sfx, "noise": noise}
    mix = sum(stems.values())
    peak = float(np.max(np.abs(mix)))
    scale = 0.9 / peak if peak > 0.9 else 1.0
    for k, v in stems.items():
        sf.write(out / f"truth_{k}.wav", np.repeat((v * scale)[:, None], 2, 1), SR, subtype="FLOAT")
    wav = out / "demo_mix.wav"
    sf.write(wav, np.repeat((mix * scale)[:, None], 2, 1), SR, subtype="FLOAT")
    video = out / "demo_ai_video.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc2=size=640x360:rate=24:duration={a.seconds}", "-i", str(wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(video),
    ], check=True)
    print(f"wrote {video} and ground-truth stems in {out}")
    print("timeline: dialogue 3–11 s, BGM 0–end, impact at 1.0 s, birdsong 13–16 s, whoosh 17.5 s, hiss+60 Hz hum throughout")


if __name__ == "__main__":
    main()
