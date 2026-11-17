"""Shared fixtures. Synthetic audio only — tests are deterministic and need no model weights.

``ScheduleClassifier`` and ``OracleSeparator`` are *test doubles*: they let us test the
pipeline mechanics (timestamps, routing, reconstruction, export, API, MCP) exactly, while
real-model behaviour is covered by the opt-in ``-m models`` tests.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiolayer.audio import io as aio
from audiolayer.classification.base import TRACKS, FrameProbs
from audiolayer.config import Settings
from audiolayer.core.components import Components
from audiolayer.core.service import AudioLayerService

SR = 48000


def tone(freq: float, seconds: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_stereo(path: Path, mono: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), np.repeat(mono[:, None], 2, axis=1), sr, subtype="FLOAT")
    return path


class ScheduleClassifier:
    """Returns probability 0.9 inside scheduled [start, end) windows, 0.0 elsewhere."""

    name = "schedule-test-double"
    license = "test"
    calibrated = True

    def __init__(self, schedule: dict[str, dict[str, list[tuple[float, float]]]], hop: float = 0.05):
        # schedule: {"vocals"|"accompaniment"|"mix": {track: [(start, end), ...]}}
        self.schedule = schedule
        self.hop = hop

    def frame_probs(self, path: Path, on_progress=None, check_cancel=None) -> FrameProbs:
        sr, _, n = aio.info(path)
        frames = int(np.ceil(n / sr / self.hop))
        key = "vocals" if path.stem == "vocals" else "accompaniment" if path.stem == "accompaniment" else "mix"
        probs = {t: np.zeros(frames) for t in TRACKS}
        for track, wins in self.schedule.get(key, {}).items():
            for a, b in wins:
                probs[track][int(round(a / self.hop)) : int(round(b / self.hop))] = 0.9
        if on_progress:
            on_progress(1.0)
        return FrameProbs(hop=self.hop, probs=probs, top_labels=[])


class OracleSeparator:
    """Writes the known voice signal as 'vocals' and the exact complement as accompaniment."""

    name = "oracle-test-double"
    license = "test"

    def __init__(self, voice_path: Path):
        self.voice_path = voice_path

    def separate(self, master: Path, out_dir: Path, on_progress=None, check_cancel=None) -> dict[str, Path]:
        sr, ch, n = aio.info(master)
        x = aio.read_range(master, 0, n)
        v = aio.fit_length(aio.read_range(self.voice_path, 0, n), n)
        out = {"vocals": out_dir / "vocals.wav", "accompaniment": out_dir / "accompaniment.wav"}
        sf.write(str(out["vocals"]), v, sr, subtype="FLOAT")
        sf.write(str(out["accompaniment"]), x - v, sr, subtype="FLOAT")
        if on_progress:
            on_progress(1.0)
        return out


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(workspace=tmp_path / "ws", model_dir=tmp_path / "models", input_dirs=[tmp_path / "inputs"],
                 allow_model_download=False, chunk_seconds=4.0, max_duration_s=600)
    s.ensure_dirs()
    (tmp_path / "inputs").mkdir(exist_ok=True)
    return s


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Path]:
    """10 s scene: speech-like voice 2–5 s, music bed 0–10 s, impact at 7.0–7.4 s."""
    d = tmp_path / "inputs"
    d.mkdir(exist_ok=True)
    n = 10 * SR
    voice = np.zeros(n, np.float32)
    seg = tone(220, 3.0, 0.25) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(3 * SR) / SR)).astype(np.float32)
    voice[2 * SR : 5 * SR] = seg
    music = tone(440, 10.0, 0.08) + tone(660, 10.0, 0.05)
    fx = np.zeros(n, np.float32)
    burst = (np.random.default_rng(0).standard_normal(int(0.4 * SR)) * 0.1 * np.exp(-np.arange(int(0.4 * SR)) / SR * 12)).astype(np.float32)
    fx[int(7.0 * SR) : int(7.0 * SR) + len(burst)] = burst
    mix = voice + music + fx
    return {
        "mix": write_stereo(d / "scene.wav", mix),
        "voice": write_stereo(tmp_path / "voice_truth.wav", voice),
    }


SCENE_SCHEDULE = {
    "vocals": {"dialogue": [(2.0, 5.0)]},
    "accompaniment": {"music": [(0.0, 10.0)], "sfx": [(7.0, 7.4)]},
    "mix": {"dialogue": [(2.0, 5.0)], "music": [(0.0, 10.0)], "sfx": [(7.0, 7.4)]},
}


@pytest.fixture
def service(settings: Settings, scene: dict[str, Path]) -> AudioLayerService:
    comps = Components(classifier=ScheduleClassifier(SCENE_SCHEDULE), separator=OracleSeparator(scene["voice"]), asr=None)
    svc = AudioLayerService(settings, comps)
    yield svc
    svc.jobs.shutdown()


@pytest.fixture
def detect_only_service(settings: Settings) -> AudioLayerService:
    comps = Components(classifier=ScheduleClassifier(SCENE_SCHEDULE), separator=None, asr=None)
    svc = AudioLayerService(replace(settings), comps)
    yield svc
    svc.jobs.shutdown()


def wait_job(svc: AudioLayerService, job_id: str, owner: str = "local", timeout: float = 60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = svc.job(job_id, owner)
        if j.status not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


def make_video(path: Path, audio: Path | None, seconds: float = 3.0) -> Path:
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
           f"testsrc2=size=160x120:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path
