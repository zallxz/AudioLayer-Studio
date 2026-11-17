"""Noise separation chain: high-pass (rumble/DC) → hum notches → spectral gating (hiss).

The chain only runs for problems the quality analysis actually measured (unless forced),
so clean audio is never degraded. What it removes becomes the Noise stream, which keeps
the invariant ``clean + noise == input``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from ..audio.quality import NoiseProfile

try:
    import noisereduce as _nr
except ImportError:  # pragma: no cover - optional dependency
    _nr = None


def build_chain(profile: NoiseProfile, sr: int, *, force: bool = False):  # type: ignore[no-untyped-def]
    sos_list: list[np.ndarray] = []
    if profile.rumble or abs(profile.dc_offset) > 0.005 or force:
        sos_list.append(butter(4, 30, btype="highpass", fs=sr, output="sos"))
    if profile.hum_hz:
        for k in range(1, 9):
            f = profile.hum_hz * k
            if f >= sr / 2 - 100:
                break
            b, a = iirnotch(f, Q=35 if k == 1 else 25, fs=sr)
            sos_list.append(tf2sos(b, a))
    use_gate = profile.hiss or force

    def apply(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float64)
        for sos in sos_list:
            y = sosfiltfilt(sos, y, axis=0)
        if use_gate and _nr is not None:
            y = _nr.reduce_noise(y=y.T, y_noise=_noise_clip(y, sr).T, sr=sr, stationary=True,
                                 prop_decrease=0.85, n_std_thresh_stationary=1.5).T
        return y.astype(np.float32)

    return apply


def describe(profile: NoiseProfile, force: bool = False) -> list[str]:
    steps = []
    if profile.rumble or abs(profile.dc_offset) > 0.005 or force:
        steps.append("30 Hz high-pass (rumble/DC)")
    if profile.hum_hz:
        steps.append(f"notch filters at {profile.hum_hz:.0f} Hz × 1–8")
    if profile.hiss or force:
        steps.append("stationary spectral gating (hiss)")
    return steps


def _noise_clip(y: np.ndarray, sr: int, frame_s: float = 0.05, total_s: float = 2.0) -> np.ndarray:
    """Concatenate the quietest frames of ``y`` as the noise profile.

    Estimating the profile from the whole signal (noisereduce's default) treats sustained
    speech/music as "noise" and gates it away; the quietest ~2 s are a far better proxy for
    the stationary floor (hiss) in AI-generated audio.
    """
    h = max(1, int(sr * frame_s))
    k = len(y) // h
    if k < 4:
        return y
    frames = y[: k * h].reshape(k, h, -1)
    rms = np.sqrt(np.mean(frames**2, axis=(1, 2)))
    take = max(2, min(k, int(total_s / frame_s)))
    idx = np.sort(np.argsort(rms)[:take])
    return frames[idx].reshape(-1, y.shape[1])
