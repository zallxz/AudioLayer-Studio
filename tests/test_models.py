"""Real-model integration test (opt-in): ``pytest -m models``.

Needs the ``ml`` extra, model weights (downloaded on first run) and network access for the
demo audio. It runs PANNs + Demucs + faster-whisper on a synthetic scene with known ground
truth and checks *measured* behaviour — it does not assume quality.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from audiolayer.core.components import build_components
from audiolayer.core.schemas import AnalyzeOptions
from audiolayer.core.service import AudioLayerService

from conftest import wait_job

pytestmark = pytest.mark.models
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("demucs") is None:
        pytest.skip("ml extra not installed")
    out = tmp_path_factory.mktemp("demo")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_media.py"), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"demo media unavailable: {r.stderr[-300:]}")
    return out


def test_real_models_end_to_end(demo: Path, settings, tmp_path):
    from dataclasses import replace

    s = replace(settings, allow_model_download=True, model_dir=Path.home() / ".cache" / "audiolayer", chunk_seconds=30)
    svc = AudioLayerService(s, build_components(s))
    try:
        pr = svc.create_from_file(demo / "demo_ai_video.mp4", "demo.mp4", "local", move=False)
        j = wait_job(svc, svc.analyze(pr.id, AnalyzeOptions(), "local").id, timeout=1800)
        assert j.status == "succeeded", j.error
        pr = svc.get(pr.id, "local")
        assert pr.analysis.separated
        dlg = [x for x in pr.segments if x.category == "dialogue"]
        assert dlg and any(x.start < 4.0 and x.end > 9.0 for x in dlg), dlg  # speech is at 3–11 s
        assert any(x.category in ("bgm", "music") for x in pr.segments)
        assert any(i.kind == "hiss" for i in pr.analysis.issues)
        assert any(x.transcript for x in dlg)
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate import mono, si_sdr

        est = mono(svc.store.path(pr.id, "stems", "dialogue.flac"))
        ref = mono(demo / "truth_dialogue.wav")
        mix = mono(svc.store.path(pr.id, "audio", "master.wav"))
        n = min(len(est), len(ref))
        gain = si_sdr(est[:n], ref[:n]) - si_sdr(mix[:n], ref[:n])
        print(f"dialogue SI-SDR improvement: {gain:.1f} dB")
        assert gain > 3.0
        assert np.isfinite(gain)
    finally:
        svc.jobs.shutdown()
