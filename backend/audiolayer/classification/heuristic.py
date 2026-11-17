"""DSP-only fallback classifier (no PyTorch required).

This is NOT a trained model. It uses hand-written acoustic features (harmonicity,
syllabic 2–8 Hz modulation, spectral flux, flatness) so the app still runs on machines
without the ML extras. Accuracy is limited; every segment it produces is flagged as
uncertain and its confidences are capped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..audio import io as aio
from .base import CancelFn, FrameProbs, ProgressFn

SR = 16000
HOP_S = 0.05
CAP = 0.6


def _sig(x: np.ndarray, center: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) * slope))


class HeuristicClassifier:
    name = "heuristic-dsp"
    license = "Apache-2.0 (this project)"
    calibrated = False

    def frame_probs(self, path: Path, on_progress: ProgressFn | None = None,
                    check_cancel: CancelFn | None = None) -> FrameProbs:
        sr, _, n = aio.info(path)
        hop = int(HOP_S * SR)
        win = 1024
        window = np.hanning(win)
        feats: dict[str, list[float]] = {k: [] for k in ("rms", "harm", "flux", "flat", "low", "pitch")}
        prev_mag: np.ndarray | None = None
        freqs = np.fft.rfftfreq(win, 1 / SR)
        speech_band = (freqs > 250) & (freqs < 4000)
        block_in = int(sr * 20)
        done = 0
        for _, block in aio.iter_blocks(path, block_in):
            if check_cancel:
                check_cancel()
            y = aio.resample(aio.to_mono(block), sr, SR)
            for f0 in range(0, len(y), hop):
                fr = y[f0 : f0 + win]
                if len(fr) < win:
                    fr = np.pad(fr, (0, win - len(fr)))
                rms = float(np.sqrt(np.mean(fr**2) + 1e-12))
                mag = np.abs(np.fft.rfft(fr * window)) + 1e-9
                ac = np.fft.irfft(mag**2)
                lo, hi = SR // 400, SR // 70
                harm = float(np.max(ac[lo:hi]) / (ac[0] + 1e-9))
                lag = int(np.argmax(ac[lo:hi]) + lo)
                flux = 0.0 if prev_mag is None else float(np.sum(np.maximum(mag - prev_mag, 0)) / (np.sum(mag) + 1e-9))
                prev_mag = mag
                feats["rms"].append(rms)
                feats["harm"].append(harm)
                feats["flux"].append(flux)
                feats["flat"].append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))
                feats["low"].append(float(np.sum(mag[speech_band] ** 2) / np.sum(mag**2)))
                feats["pitch"].append(SR / lag)
            done += len(block)
            if on_progress:
                on_progress(min(1.0, done / max(n, 1)))
        f = {k: np.asarray(v) for k, v in feats.items()}
        db = 20 * np.log10(f["rms"] + 1e-12)
        active = _sig(db, -45, 0.5)
        # syllabic modulation: 2-8 Hz energy fluctuation over ~1 s
        env = db - uniform_filter1d(db, size=int(1.0 / HOP_S))
        mod = uniform_filter1d(np.abs(env), size=int(0.6 / HOP_S))
        pitch_var = np.abs(np.diff(np.log(f["pitch"]), prepend=np.log(f["pitch"][:1] + 1e-9)))
        stable = 1 - _sig(uniform_filter1d(pitch_var, size=int(0.5 / HOP_S)), 0.08, 40)
        harm = uniform_filter1d(f["harm"], size=5)
        speech = active * _sig(harm, 0.35, 10) * _sig(mod, 4.0, 1.0) * _sig(f["low"], 0.5, 8)
        music = active * _sig(harm, 0.45, 10) * stable * (1 - _sig(mod, 6.0, 1.0))
        sfx = active * _sig(uniform_filter1d(f["flux"], size=3), 0.25, 15) * (1 - _sig(harm, 0.5, 10))
        flat = uniform_filter1d(f["flat"], size=int(1.0 / HOP_S))
        noise = active * _sig(flat, 0.45, 20) * (1 - _sig(f["flux"], 0.15, 20))
        tracks = {
            "dialogue": np.clip(speech, 0, CAP),
            "singing": np.zeros_like(speech),
            "music": np.clip(music, 0, CAP),
            "sfx": np.clip(sfx, 0, CAP),
            "noise": np.clip(noise, 0, CAP),
            "bgm_hint": np.zeros_like(speech),
            "fg_hint": np.zeros_like(speech),
        }
        return FrameProbs(hop=HOP_S, probs=tracks, top_labels=[])
