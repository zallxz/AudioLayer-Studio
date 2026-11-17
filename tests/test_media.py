"""File validation, audio extraction, quality analyzers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from audiolayer.audio import io as aio
from audiolayer.audio import quality
from audiolayer.audio.ffmpeg import probe
from audiolayer.core import pipeline
from audiolayer.core.errors import Forbidden, InvalidMedia, LimitExceeded, NotFound
from audiolayer.core.paths import safe_filename, validate_input_path
from audiolayer.core.store import ProjectStore

from conftest import SR, make_video, tone, write_stereo


def test_safe_filename_strips_paths_and_specials():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my clip (1).mp4") == "my clip _1_.mp4"
    assert safe_filename("") == "upload"


def test_validate_input_path(tmp_path: Path):
    root = tmp_path / "in"
    root.mkdir()
    ok = write_stereo(root / "a.wav", tone(440, 0.5))
    assert validate_input_path(str(ok), [root]) == ok.resolve()
    assert validate_input_path("a.wav", [root]) == ok.resolve()  # relative to first root
    outside = write_stereo(tmp_path / "b.wav", tone(440, 0.5))
    with pytest.raises(Forbidden):
        validate_input_path(str(outside), [root])
    link = root / "link.wav"
    os.symlink(outside, link)
    with pytest.raises(Forbidden):  # symlink escaping the allowlist
        validate_input_path(str(link), [root])
    (root / "x.txt").write_text("hi")
    with pytest.raises(InvalidMedia):
        validate_input_path(str(root / "x.txt"), [root])
    with pytest.raises(NotFound):
        validate_input_path(str(root / "missing.wav"), [root])
    with pytest.raises(Forbidden):
        validate_input_path(str(ok), [])  # local access disabled


def test_corrupted_media_rejected(tmp_path: Path, settings):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(2000))
    with pytest.raises(InvalidMedia):
        probe(bad)
    with pytest.raises(InvalidMedia):
        pipeline.ingest(ProjectStore(settings), settings, bad)


def test_duration_limit(tmp_path: Path, settings):
    from dataclasses import replace

    f = write_stereo(tmp_path / "long.wav", tone(220, 3.0))
    with pytest.raises(LimitExceeded):
        pipeline.ingest(ProjectStore(settings), replace(settings, max_duration_s=1.0), f)


def test_extract_audio_from_video_keeps_original(tmp_path: Path, settings):
    wav = write_stereo(tmp_path / "a.wav", tone(440, 3.0))
    vid = make_video(tmp_path / "clip.mp4", wav, 3.0)
    digest = hashlib.sha256(vid.read_bytes()).hexdigest()
    store = ProjectStore(settings)
    pr = pipeline.ingest(store, settings, vid)
    assert pr.media.has_video and pr.media.has_audio
    assert pr.media.width == 160 and pr.media.fps == 25
    master = store.path(pr.id, "audio", "master.wav")
    sr, ch, n = aio.info(master)
    assert sr == 48000 and ch == 2
    assert abs(n / sr - 3.0) < 0.05
    orig = store.path(pr.id, pr.original_file)
    assert hashlib.sha256(orig.read_bytes()).hexdigest() == digest
    assert hashlib.sha256(vid.read_bytes()).hexdigest() == digest  # source untouched
    assert not os.access(orig, os.W_OK) or os.geteuid() == 0  # read-only (root ignores modes)


def test_video_without_audio_gets_silent_master(tmp_path: Path, settings):
    vid = make_video(tmp_path / "silent.mp4", None, 2.0)
    store = ProjectStore(settings)
    pr = pipeline.ingest(store, settings, vid)
    assert pr.media.has_audio is False
    issues, _, _ = quality.analyze(store.path(pr.id, "audio", "master.wav"), pr.media)
    assert issues[0].kind == "no_audio_track"


def _media(tmp_path: Path, x: np.ndarray):
    f = write_stereo(tmp_path / "q.wav", x)
    return f, probe(f)


def test_quality_detects_hum(tmp_path: Path):
    t = np.arange(4 * SR) / SR
    x = (0.2 * np.sin(2 * np.pi * 1000 * t) * (t % 1 < 0.5) + 0.02 * np.sin(2 * np.pi * 60 * t)
         + 0.01 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    f, m = _media(tmp_path, x)
    issues, segs, prof = quality.analyze(f, m)
    assert prof.hum_hz == 60.0
    assert any(i.kind == "hum" for i in issues)
    assert any(s.category == "noise" for s in segs)


def test_quality_detects_hiss_and_clipping(tmp_path: Path):
    rng = np.random.default_rng(1)
    t = np.arange(4 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 300 * t) * (t % 2 < 1) + rng.standard_normal(len(t)) * 0.01).astype(np.float32)
    x[SR : SR + 2000] = 1.0
    f, m = _media(tmp_path, x)
    issues, segs, prof = quality.analyze(f, m)
    kinds = {i.kind for i in issues}
    assert "hiss" in kinds and prof.hiss
    assert "clipping" in kinds
    clip = [s for s in segs if "clipping" in s.flags]
    assert clip and clip[0].start <= 1.0 + 0.1 and clip[0].end >= 1.0


def test_quality_silent_and_clean(tmp_path: Path):
    f, m = _media(tmp_path, np.zeros(SR, np.float32))
    issues, _, _ = quality.analyze(f, m)
    assert [i.kind for i in issues] == ["silent_audio"]
    f, m = _media(tmp_path, tone(440, 2.0, 0.3))
    issues, segs, prof = quality.analyze(f, m)
    assert not prof.has_removable_noise
    assert not any(s.category == "noise" for s in segs)
