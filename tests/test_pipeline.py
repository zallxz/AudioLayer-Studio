"""Separation pipeline, reconstruction, segment editing and export (using test doubles)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from audiolayer.audio import io as aio
from audiolayer.core.errors import Conflict
from audiolayer.core.schemas import AnalyzeOptions, ExportOptions
from audiolayer.separation.mixer import render_mix

from conftest import SR, wait_job


@pytest.fixture
def analyzed(service, scene):
    pr = service.create_from_file(scene["mix"], "scene.wav", "local", move=False)
    job = service.analyze(pr.id, AnalyzeOptions(transcribe=False), "local")
    j = wait_job(service, job.id)
    assert j.status == "succeeded", j.error
    return service.get(pr.id, "local")


def _read(p):
    x, _ = sf.read(str(p), dtype="float32", always_2d=True)
    return x


def test_segments_and_timestamps(analyzed):
    segs = analyzed.segments
    dlg = [s for s in segs if s.category == "dialogue"]
    fx = [s for s in segs if s.category == "sfx"]
    mus = [s for s in segs if s.category in ("bgm", "music")]
    assert len(dlg) == 1 and abs(dlg[0].start - 2.0) < 0.1 and abs(dlg[0].end - 5.0) < 0.1
    assert len(fx) == 1 and abs(fx[0].start - 7.0) < 0.15
    assert mus and mus[0].category == "bgm"  # music under dialogue → background role
    assert analyzed.analysis.separated and analyzed.status == "analyzed"


def test_streams_reconstruct_master_exactly(service, analyzed):
    st = service.store
    master = _read(st.path(analyzed.id, "audio", "master.wav"))
    total = sum(_read(st.path(analyzed.id, "streams", f"{k}.wav")) for k in ("voice", "music", "fx", "noise"))
    assert total.shape == master.shape
    assert np.max(np.abs(total - master)) < 1e-5
    # Category stems (16-bit preview FLAC) also sum back to the master within quantisation error
    stems = sum(_read(st.path(analyzed.id, "stems", f"{c}.flac")) for c in ("dialogue", "bgm", "sfx", "music", "noise"))
    assert np.max(np.abs(stems - master)) < 5e-4


def test_removing_dialogue_removes_the_voice(service, analyzed, scene):
    st = service.store
    streams = {k: st.path(analyzed.id, "streams", f"{k}.wav") for k in ("voice", "music", "fx", "noise")}
    out = st.path(analyzed.id, "renders", "t.wav")
    render_mix(streams, analyzed.segments, ["dialogue"], out)
    y = _read(out)[:, 0]
    master = _read(st.path(analyzed.id, "audio", "master.wav"))[:, 0]
    voice = _read(scene["voice"])[:, 0]
    np.testing.assert_allclose(y, master - voice, atol=1e-4)
    # keep-everything render is identical to the master (non-destructive)
    render_mix(streams, analyzed.segments, [], out)
    np.testing.assert_allclose(_read(out)[:, 0], master, atol=1e-5)


def test_segment_mute_solo_and_undo(service, analyzed):
    st = service.store
    fx = next(s for s in analyzed.segments if s.category == "sfx")
    service.update_segment(analyzed.id, fx.id, "local", muted=True)
    pr = service.get(analyzed.id, "local")
    streams = {k: st.path(pr.id, "streams", f"{k}.wav") for k in ("voice", "music", "fx", "noise")}
    out = st.path(pr.id, "renders", "m.wav")
    render_mix(streams, pr.segments, [], out)
    y = _read(out)[:, 0]
    fx_stream = _read(streams["fx"])[:, 0]
    master = _read(st.path(pr.id, "audio", "master.wav"))[:, 0]
    a, b = int((fx.start + 0.05) * SR), int((fx.end - 0.05) * SR)
    np.testing.assert_allclose(y[a:b], (master - fx_stream)[a:b], atol=1e-4)
    np.testing.assert_allclose(y[: SR], master[: SR], atol=1e-5)  # untouched elsewhere
    # solo: only that segment is audible
    service.update_segment(pr.id, fx.id, "local", muted=False, solo=True)
    pr = service.get(pr.id, "local")
    render_mix(streams, pr.segments, [], out)
    y = _read(out)[:, 0]
    assert np.max(np.abs(y[: SR * 6])) < 1e-6
    # undo twice restores original flags
    service.undo(pr.id, "local")
    service.undo(pr.id, "local")
    seg = next(s for s in service.get(pr.id, "local").segments if s.id == fx.id)
    assert not seg.muted and not seg.solo
    with pytest.raises(Conflict):
        service.undo(pr.id, "local")


def test_relabel_reroutes_stems(service, analyzed):
    mus = next(s for s in analyzed.segments if s.category == "bgm")
    res = service.update_segment(analyzed.id, mus.id, "local", category="music")
    assert res["segment"]["category"] == "music" and res["segment"]["edited"]
    j = wait_job(service, res["render_job"]["id"])
    assert j.status == "succeeded"
    st = service.store
    music = _read(st.path(analyzed.id, "stems", "music.flac"))
    bgm = _read(st.path(analyzed.id, "stems", "bgm.flac"))
    assert np.sqrt(np.mean(music**2)) > 0.01 and np.max(np.abs(bgm)) < 1e-3


def test_presets(service, analyzed):
    pr = service.apply_preset(analyzed.id, "music_and_effects", "local")
    assert pr.mix.removed == ["dialogue"]
    pr = service.apply_preset(analyzed.id, "remove_music", "local")
    assert pr.mix.removed == ["bgm", "music"]
    assert service.limitations_for(pr, ["dialogue"])


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3"])
def test_export_formats_and_sync(service, analyzed, fmt):
    opts = ExportOptions(items=["mix", "dialogue", "bgm", "sfx", "music", "noise", "original", "metadata"],
                         format=fmt, remove=["bgm", "music"], edge_fade_ms=0)
    j = wait_job(service, service.export(analyzed.id, opts, "local").id, timeout=120)
    assert j.status == "succeeded", j.error
    d = service.store.path(analyzed.id, j.result["dir"])
    _, _, n_master = aio.info(service.store.path(analyzed.id, "audio", "master.wav"))
    audio = [f for f in j.result["files"] if f.endswith("." + fmt)]
    assert len(audio) == 7
    for f in audio:
        info = sf.info(str(d / f))
        assert info.samplerate == 48000
        if fmt in ("wav", "flac"):
            assert info.frames == n_master  # sample-accurate, silence preserved
        else:
            assert abs(info.frames - n_master) < 48000 * 0.06
        if fmt == "wav":
            assert info.subtype == "PCM_24"
    meta = json.loads((d / "scene_metadata.json").read_text())
    assert meta["mix"]["removed_categories"] == [] or "segments" in meta
    assert (d / "scene_segments.csv").read_text().startswith("id,category,start,end")
    with zipfile.ZipFile(service.store.path(analyzed.id, j.result["zip"])) as z:
        assert len(z.namelist()) == len(j.result["files"])


def test_export_cleaned_mix_matches_reconstruction(service, analyzed, scene):
    j = wait_job(service, service.export(analyzed.id, ExportOptions(items=["mix"], format="wav", remove=["dialogue"],
                                                                    edge_fade_ms=0, zip=False), "local").id)
    d = service.store.path(analyzed.id, j.result["dir"])
    y = _read(d / "scene_cleaned.wav")[:, 0]
    master = _read(service.store.path(analyzed.id, "audio", "master.wav"))[:, 0]
    voice = _read(scene["voice"])[:, 0]
    np.testing.assert_allclose(y, master - voice, atol=2e-4)  # 24-bit quantisation


def test_loudness_preset(service, analyzed):
    j = wait_job(service, service.export(analyzed.id, ExportOptions(items=["mix"], loudness_preset="youtube", zip=False),
                                         "local").id, timeout=120)
    assert j.status == "succeeded", j.error
    lufs = j.result["info"]["mix_loudness_after"]["integrated_lufs"]
    assert abs(lufs - (-14)) < 1.5


def test_detection_only_mode(detect_only_service, scene):
    svc = detect_only_service
    pr = svc.create_from_file(scene["mix"], "scene.wav", "local", move=False)
    j = wait_job(svc, svc.analyze(pr.id, AnalyzeOptions(), "local").id)
    assert j.status == "succeeded"
    pr = svc.get(pr.id, "local")
    assert not pr.analysis.separated
    assert all("not_removable" in s.flags for s in pr.segments)
    assert any("Detection only" in x for x in pr.analysis.limitations)
    with pytest.raises(Conflict):
        svc.render_preview(pr.id, "local")
    j = wait_job(svc, svc.export(pr.id, ExportOptions(items=["mix", "dialogue"], remove=["bgm"]), "local").id)
    assert j.status == "succeeded"
    assert {s["item"] for s in j.result["skipped"]} == {"mix", "dialogue"}


def test_cancel_analysis(service, scene):
    pr = service.create_from_file(scene["mix"], "scene.wav", "local", move=False)
    job = service.analyze(pr.id, AnalyzeOptions(), "local")
    service.cancel(job.id, "local")
    j = wait_job(service, job.id)
    assert j.status in ("cancelled", "succeeded")  # may finish before the cancel lands
    if j.status == "cancelled":
        assert service.get(pr.id, "local").status in ("uploaded", "analyzed")
