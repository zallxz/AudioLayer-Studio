"""Torch-free engine pieces: numpy STFT/iSTFT (Demucs-compatible), REPET-SIM, engine selection, mirror."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from audiolayer.core import components
from audiolayer.models import registry
from audiolayer.separation.demucs_onnx import HOP, NFFT, ispec_cac, istft, spec_cac, stft
from audiolayer.separation.routing import nn_median_background, tf_foreground


def test_stft_istft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 30000))
    z = stft(x, 2048, 512)
    assert z.shape == (2, 1025, 1 + 30000 // 512)
    np.testing.assert_allclose(istft(z, 512, 30000), x, atol=1e-9)


def test_stft_matches_torch_when_available():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(1).standard_normal((1, 20000))
    ref = torch.stft(torch.from_numpy(x), NFFT, HOP, window=torch.hann_window(NFFT, dtype=torch.float64),
                     win_length=NFFT, normalized=True, center=True, return_complex=True, pad_mode="reflect").numpy()
    np.testing.assert_allclose(stft(x, NFFT, HOP), ref, atol=1e-9)


def test_demucs_spec_roundtrip():
    # band-limited signal: HTDemucs drops the Nyquist bin, so white noise would not round-trip exactly
    L = 50000
    t = np.arange(L) / 44100
    mix = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 1000 * t)])[None].astype(np.float32)
    mag = spec_cac(mix)
    assert mag.shape == (1, 4, NFFT // 2, int(np.ceil(L / HOP)))
    y = ispec_cac(mag[:, None], L)  # one "source" equal to the input spectrogram
    np.testing.assert_allclose(y[0, 0][:, 3000:-3000], mix[0][:, 3000:-3000], atol=1e-5)


def test_demucs_spec_matches_torch_when_available():
    pytest.importorskip("demucs")
    torch = pytest.importorskip("torch")
    from demucs.pretrained import get_model

    m = get_model("htdemucs").models[0]
    x = torch.randn(1, 2, 30000)
    ref = m._magnitude(m._spec(x)).numpy()
    np.testing.assert_allclose(spec_cac(x.numpy()), ref, atol=1e-4)
    np.testing.assert_allclose(ispec_cac(ref[:, None], 30000)[:, 0], m._ispec(m._spec(x), 30000).numpy(), atol=1e-4)


def test_repet_separates_one_off_event():
    sr = 16000
    t = np.arange(sr * 8) / sr
    music = 0.2 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0)  # repeating pattern
    hit = np.zeros_like(t)
    hit[sr * 4: sr * 4 + 1600] = np.random.default_rng(3).standard_normal(1600) * 0.5
    x = np.stack([music + hit, music + hit], axis=1).astype(np.float32)
    fg = tf_foreground(x, sr)
    assert fg is not None and fg.shape == x.shape
    e_hit = np.sum(fg[sr * 4: sr * 4 + 1600, 0] ** 2) / np.sum(hit[sr * 4: sr * 4 + 1600] ** 2)
    e_music = np.sum(fg[: sr * 3, 0] ** 2) / np.sum(music[: sr * 3] ** 2)
    assert e_hit > 0.3 and e_music < 0.1


def test_nn_median_background_shapes():
    S = np.abs(np.random.default_rng(4).standard_normal((64, 40)))
    bg = nn_median_background(S, width=3)
    assert bg.shape == S.shape and np.all(bg >= 0)


def test_engine_selection(settings, monkeypatch):
    monkeypatch.setattr(components, "deps_available", lambda spec: False)
    monkeypatch.setattr(components, "_has", lambda mod: mod == "onnxruntime")
    assert components.pick_engine(settings) == "onnx"
    assert components.pick_engine(replace(settings, engine="torch")) == "none"
    monkeypatch.setattr(components, "_has", lambda mod: False)
    assert components.pick_engine(settings) == "none"
    c = components.build_components(settings)
    assert c.classifier.name == "heuristic-dsp" and c.separator is None


def test_mirror_download_joins_parts_and_verifies(tmp_path: Path, monkeypatch):
    payload = b"abcdefghij" * 1000
    parts = [payload[:4000], payload[4000:]]
    sha = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []

    def fake_fetch(url: str, f) -> None:  # type: ignore[no-untyped-def]
        calls.append(url)
        f.write(parts[int(url[-2:])])

    monkeypatch.setattr(registry, "_fetch", fake_fetch)
    spec = registry.ModelSpec(key="t", role="r", name="n", license="MIT", source="s", size_mb=0, python_deps=(),
                              filename="m.onnx", mirror=(registry.MirrorFile("m.onnx", sha, 2),))
    p = registry.ensure_weights(spec, tmp_path, allow_download=True)
    assert p.read_bytes() == payload and len(calls) == 2 and calls[0].endswith("m.onnx.part00")
    bad = replace(spec, key="t2", mirror=(registry.MirrorFile("m.onnx", "0" * 64, 2),))
    with pytest.raises(registry.ModelUnavailable):
        registry.ensure_weights(bad, tmp_path, allow_download=True)
    assert not (tmp_path / "t2" / "m.onnx").exists()
