"""Streaming audio I/O helpers built on soundfile; nothing here loads a whole file."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def info(path: Path) -> tuple[int, int, int]:
    i = sf.info(str(path))
    return i.samplerate, i.channels, i.frames


def iter_blocks(path: Path, block_frames: int) -> Iterator[tuple[int, np.ndarray]]:
    with sf.SoundFile(str(path)) as f:
        pos = 0
        while True:
            data = f.read(block_frames, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            yield pos, data
            pos += len(data)


def read_range(path: Path, start: int, stop: int) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        start = max(0, start)
        stop = min(stop, f.frames)
        f.seek(start)
        return f.read(max(0, stop - start), dtype="float32", always_2d=True)


def open_writer(path: Path, sr: int, channels: int) -> sf.SoundFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sf.SoundFile(str(path), "w", samplerate=sr, channels=channels, subtype="FLOAT", format="WAV")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Zero-phase polyphase resampling along axis 0 (keeps timing aligned)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    y = resample_poly(x, sr_out // g, sr_in // g, axis=0)
    return np.asarray(y, dtype=np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def process_chunked(
    src: Path,
    outputs: list[str],
    out_paths: dict[str, Path],
    fn: Callable[[np.ndarray], dict[str, np.ndarray]],
    *,
    chunk_s: float,
    overlap_s: float = 2.0,
    on_progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    """Run ``fn`` over overlapping chunks of ``src`` and stream results to ``out_paths``.

    Each chunk is processed with ``overlap_s`` of context on both sides; neighbouring
    chunks are linearly cross-faded over ``overlap_s`` around the boundary so there are
    no clicks, and every output has exactly the same number of frames as the input.
    """
    sr, ch, n = info(src)
    hop = max(int(chunk_s * sr), sr)
    ov = min(int(overlap_s * sr), hop // 2)
    half = ov // 2
    writers = {k: open_writer(out_paths[k], sr, ch) for k in outputs}
    pending: dict[str, np.ndarray] | None = None  # previous chunk's [b - half, b + half)
    try:
        starts = list(range(0, max(n, 1), hop))
        for idx, s0 in enumerate(starts):
            if check_cancel:
                check_cancel()
            s1 = min(n, s0 + hop)
            a, b = max(0, s0 - ov), min(n, s1 + ov)
            x = read_range(src, a, b)
            res = fn(x)
            for k in outputs:
                res[k] = fit_length(res[k].astype(np.float32, copy=False), len(x))
            # Region of this chunk to emit: [s0 - half, s1 - half) except at edges.
            emit_a = s0 - half if idx > 0 else 0
            emit_b = s1 - half if idx < len(starts) - 1 else n
            for k in outputs:
                seg = res[k][emit_a - a : emit_b - a].copy()
                if pending is not None and half > 0:
                    w = len(pending[k])
                    fade = np.linspace(0.0, 1.0, w, dtype=np.float32)[:, None]
                    seg[:w] = pending[k] * (1 - fade) + seg[:w] * fade
                if idx < len(starts) - 1 and half > 0:
                    # Keep the tail [s1 - half, s1 + half) to blend with the next chunk.
                    tail = res[k][s1 - half - a : min(s1 + half, b) - a]
                    writers[k].write(seg)
                    if pending is None:
                        pending = {}
                    pending[k] = tail
                else:
                    writers[k].write(seg)
            if on_progress:
                on_progress((idx + 1) / len(starts))
    finally:
        for w in writers.values():
            w.close()
