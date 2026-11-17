"""Torch-free Demucs (htdemucs) separation with ONNX Runtime.

The network itself runs from ``htdemucs_core.onnx`` (exported by scripts/export_onnx.py); the
spectrogram transforms that ONNX cannot express (complex STFT/iSTFT) are re-implemented here in
numpy, matching ``demucs.spec`` (``torch.stft``/``istft`` with ``normalized=True``, ``center=True``,
reflect padding, periodic Hann window) and ``HTDemucs._spec`` / ``_ispec``.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import io as aio
from ..config import Settings
from ..models.registry import MODELS, ensure_weights
from .base import CancelFn, ProgressFn

SAMPLE_RATE = 44100
NFFT = 4096
HOP = 1024
SEGMENT = 343980  # int(7.8 s * 44100), htdemucs training length
SOURCES = ("drums", "bass", "other", "vocals")
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float64)  # periodic, like torch


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(..., L) -> (..., F, T) complex, = torch.stft(center=True, reflect, normalized=True, onesided)."""
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    n_frames = 1 + (xp.shape[-1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[..., idx] * _hann(n_fft)
    z = np.fft.rfft(frames, axis=-1) / math.sqrt(n_fft)
    return np.swapaxes(z, -1, -2)


def istft(z: np.ndarray, hop: int, length: int) -> np.ndarray:
    """(..., F, T) complex -> (..., length), = torch.istft(center=True, normalized=True)."""
    n_fft = 2 * (z.shape[-2] - 1)
    win = _hann(n_fft)
    frames = np.fft.irfft(np.swapaxes(z, -1, -2), n=n_fft, axis=-1) * math.sqrt(n_fft) * win
    T = frames.shape[-2]
    total = n_fft + hop * (T - 1)
    y = np.zeros(z.shape[:-2] + (total,))
    wsum = np.zeros(total)
    for t in range(T):
        y[..., t * hop: t * hop + n_fft] += frames[..., t, :]
        wsum[t * hop: t * hop + n_fft] += win**2
    y = y / np.where(wsum > 1e-11, wsum, 1.0)
    start = n_fft // 2
    out = y[..., start: start + length]
    if out.shape[-1] < length:
        out = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(0, length - out.shape[-1])])
    return out


def spec_cac(mix: np.ndarray, nfft: int = NFFT, hop: int = HOP) -> np.ndarray:
    """HTDemucs._spec + _magnitude(cac=True): (B, C, L) -> (B, 2C, F, T) float32."""
    L = mix.shape[-1]
    le = int(math.ceil(L / hop))
    pad = hop // 2 * 3
    x = np.pad(mix.astype(np.float64), [(0, 0), (0, 0), (pad, pad + le * hop - L)], mode="reflect")
    z = stft(x, nfft, hop)[..., :-1, :]
    z = z[..., 2: 2 + le]
    B, C, Fq, T = z.shape
    m = np.stack([z.real, z.imag], axis=2)  # (B, C, 2, F, T)
    return m.reshape(B, C * 2, Fq, T).astype(np.float32)


def ispec_cac(x: np.ndarray, length: int, hop: int = HOP) -> np.ndarray:
    """HTDemucs._mask(cac) + _ispec: (B, S, 2C, F, T) -> (B, S, C, length)."""
    B, S, C2, Fq, T = x.shape
    m = x.reshape(B, S, C2 // 2, 2, Fq, T).astype(np.float64)
    z = m[:, :, :, 0] + 1j * m[:, :, :, 1]
    z = np.pad(z, [(0, 0)] * 3 + [(0, 1), (2, 2)])
    pad = hop // 2 * 3
    le = hop * int(math.ceil(length / hop)) + 2 * pad
    y = istft(z, hop, le)
    return y[..., pad: pad + length]


def _segment_weight(n: int) -> np.ndarray:
    w = np.concatenate([np.arange(1, n // 2 + 1), np.arange(n - n // 2, 0, -1)]).astype(np.float64)
    return w / w.max()


class DemucsOnnxSeparator:
    name = "demucs-htdemucs-onnx"
    license = MODELS["demucs-onnx"].license

    def __init__(self, settings: Settings):
        self.settings = settings

    def _session(self) -> Any:
        import onnxruntime as ort

        with _LOCK:
            if "s" not in _CACHE:
                path = ensure_weights(MODELS["demucs-onnx"], self.settings.model_dir,
                                      allow_download=self.settings.allow_model_download)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _CACHE["s"] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            return _CACHE["s"]

    def separate_array(self, mix: np.ndarray, check_cancel: CancelFn | None = None) -> np.ndarray:
        """mix (2, L) at 44.1 kHz, already normalised -> (4 sources, 2, L)."""
        sess = self._session()
        L = mix.shape[-1]
        stride = int(0.75 * SEGMENT)
        weight = _segment_weight(SEGMENT)
        out = np.zeros((len(SOURCES), 2, L))
        wsum = np.zeros(L)
        for off in range(0, max(L, 1), stride):
            if check_cancel:
                check_cancel()
            chunk = mix[:, off: off + SEGMENT]
            n = chunk.shape[-1]
            seg = np.zeros((1, 2, SEGMENT), dtype=np.float32)
            seg[0, :, :n] = chunk
            mag = spec_cac(seg)
            xs, xt = sess.run(None, {"mix": seg, "mag": mag})
            y = ispec_cac(xs, SEGMENT) + xt.astype(np.float64)
            w = weight[:n]
            out[:, :, off: off + n] += y[0, :, :, :n] * w
            wsum[off: off + n] += w
            if off + SEGMENT >= L:
                break
        return (out / np.maximum(wsum, 1e-8)).astype(np.float32)

    def separate(self, master: Path, out_dir: Path, on_progress: ProgressFn | None = None,
                 check_cancel: CancelFn | None = None) -> dict[str, Path]:
        sr, ch, _ = aio.info(master)
        vocal_idx = SOURCES.index("vocals")

        def fn(x: np.ndarray) -> dict[str, np.ndarray]:
            stereo = x if x.shape[1] == 2 else np.repeat(x[:, :1], 2, axis=1)
            y = aio.resample(stereo, sr, SAMPLE_RATE).T  # (2, L)
            ref = y.mean(0)
            mean, std = float(ref.mean()), float(ref.std()) + 1e-8
            est = self.separate_array(((y - mean) / std).astype(np.float32), check_cancel) * std + mean
            voc = aio.fit_length(aio.resample(est[vocal_idx].T, SAMPLE_RATE, sr), len(x))
            if ch == 1:
                voc = voc.mean(axis=1, keepdims=True)
            return {"vocals": voc, "accompaniment": x - voc}

        out = {k: out_dir / f"{k}.wav" for k in ("vocals", "accompaniment")}
        aio.process_chunked(master, ["vocals", "accompaniment"], out, fn, chunk_s=self.settings.chunk_seconds,
                            overlap_s=2.0, on_progress=on_progress, check_cancel=check_cancel)
        return out
