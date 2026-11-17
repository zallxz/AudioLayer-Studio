"""Waveform peak summaries for the timeline (computed in streaming blocks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io as aio


def compute_peaks(path: Path, points_per_second: int = 50) -> dict[str, object]:
    sr, _, n = aio.info(path)
    bucket = max(1, sr // points_per_second)
    mins: list[float] = []
    maxs: list[float] = []
    carry = np.zeros(0, dtype=np.float32)
    for _, block in aio.iter_blocks(path, bucket * 2000):
        mono = np.concatenate([carry, block.mean(axis=1)])
        k = len(mono) // bucket
        if k:
            m = mono[: k * bucket].reshape(k, bucket)
            mins.extend(np.round(m.min(axis=1), 4).tolist())
            maxs.extend(np.round(m.max(axis=1), 4).tolist())
        carry = mono[k * bucket :]
    if len(carry):
        mins.append(round(float(carry.min()), 4))
        maxs.append(round(float(carry.max()), 4))
    return {
        "sample_rate": sr,
        "frames": n,
        "duration": n / sr if sr else 0.0,
        "points_per_second": sr / bucket,
        "min": mins,
        "max": maxs,
    }
