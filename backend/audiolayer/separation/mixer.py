"""Render category stems and cleaned mixes from streams + (editable) segments.

Rendering never re-runs a model: it is a streaming multiply-add over the four separated
streams with gain envelopes derived from the segment list, so label edits, mutes, solos
and category removal are cheap, non-destructive and fully reversible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import io as aio
from ..core.schemas import CATEGORIES, DEFAULT_STREAM_CATEGORY, STREAMS, Category, Segment

RAMP_S = 0.02


def inside(t: np.ndarray, start: float, end: float, ramp: float = RAMP_S, duration: float | None = None) -> np.ndarray:
    """1 inside [start, end], 0 outside, with linear ramps of ``ramp`` centred on the edges.

    Edges that coincide with the start/end of the file get no ramp (nothing to fade against).
    """
    r = max(ramp, 1e-4)
    if start <= 1e-3:
        start = -1.0
    if duration is not None and end >= duration - 1e-2:
        end = duration + 1.0
    up = np.clip((t - (start - r / 2)) / r, 0, 1)
    down = np.clip(((end + r / 2) - t) / r, 0, 1)
    return np.minimum(up, down).astype(np.float32)


def category_weights(t: np.ndarray, stream: str, segments: Iterable[Segment],
                     duration: float | None = None) -> dict[str, np.ndarray]:
    """Fraction of ``stream`` routed to each category at times ``t`` (sums to 1)."""
    acc: dict[str, np.ndarray] = {c: np.zeros(len(t), dtype=np.float32) for c in CATEGORIES}
    covered = np.zeros(len(t), dtype=np.float32)
    for s in segments:
        if s.stream != stream or s.end < t[0] - 1 or s.start > t[-1] + 1:
            continue
        g = inside(t, s.start, s.end, duration=duration)
        acc[s.category] += g
        covered += g
    over = np.maximum(covered, 1.0)
    for c in CATEGORIES:
        acc[c] /= over
    default = DEFAULT_STREAM_CATEGORY[stream]
    acc[default] += np.clip(1.0 - covered, 0, 1)
    return acc


def stream_gain(t: np.ndarray, stream: str, segments: list[Segment], removed: set[str],
                duration: float | None = None) -> np.ndarray:
    cw = category_weights(t, stream, segments, duration)
    solos = [s for s in segments if s.solo]
    if solos:
        g = np.zeros(len(t), dtype=np.float32)
        for s in solos:
            if s.stream == stream:
                g = np.maximum(g, inside(t, s.start, s.end, duration=duration))
        return g
    g = sum((cw[c] for c in CATEGORIES if c not in removed), np.zeros(len(t), dtype=np.float32))
    for s in segments:
        if s.muted and s.stream == stream and not (s.end < t[0] - 1 or s.start > t[-1] + 1):
            g = g * (1 - inside(t, s.start, s.end, duration=duration))
    return np.asarray(g, dtype=np.float32)


def _iter(streams: dict[str, Path], block_s: float = 10.0):  # type: ignore[no-untyped-def]
    first = next(iter(streams.values()))
    sr, ch, n = aio.info(first)
    block = int(block_s * sr)
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        t = (s0 + np.arange(s1 - s0)) / sr
        data = {k: aio.read_range(p, s0, s1) for k, p in streams.items()}
        yield sr, ch, n, s0, s1, t, data


def render_mix(streams: dict[str, Path], segments: list[Segment], removed: Iterable[str], dst: Path,
               *, edge_fade_ms: float = 0.0, check_cancel: Callable[[], None] | None = None) -> Path:
    rem = set(removed)
    first = next(iter(streams.values()))
    sr, ch, n = aio.info(first)
    fade = int(sr * edge_fade_ms / 1000)
    with aio.open_writer(dst, sr, ch) as w:
        for sr, ch, n, s0, s1, t, data in _iter(streams):
            if check_cancel:
                check_cancel()
            out = np.zeros((s1 - s0, ch), dtype=np.float32)
            for k in STREAMS:
                if k in data:
                    out += data[k] * stream_gain(t, k, segments, rem, n / sr)[:, None]
            out = _apply_edge_fade(out, s0, n, fade)
            w.write(out)
    return dst


def render_category(streams: dict[str, Path], segments: list[Segment], category: Category, dst: Path,
                    *, subtype: str = "FLOAT", fmt: str = "WAV") -> Path:
    first = next(iter(streams.values()))
    sr, ch, _ = aio.info(first)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(dst), "w", samplerate=sr, channels=ch, subtype=subtype, format=fmt) as w:
        for _sr, _ch, n, s0, s1, t, data in _iter(streams):
            out = np.zeros((s1 - s0, ch), dtype=np.float32)
            for k in STREAMS:
                if k in data:
                    out += data[k] * category_weights(t, k, segments, n / _sr)[category][:, None]
            if subtype.startswith("PCM"):
                out = np.clip(out, -1, 1)
            w.write(out)
    return dst


def _apply_edge_fade(x: np.ndarray, s0: int, n: int, fade: int) -> np.ndarray:
    if fade <= 0:
        return x
    idx = s0 + np.arange(len(x))
    g = np.ones(len(x), dtype=np.float32)
    head = idx < fade
    g[head] = idx[head] / fade
    tail = idx >= n - fade
    g[tail] = np.minimum(g[tail], (n - 1 - idx[tail]) / fade)
    return x * g[:, None]


def edge_fade_file(src: Path, dst: Path, edge_fade_ms: float) -> Path:
    sr, ch, n = aio.info(src)
    fade = int(sr * edge_fade_ms / 1000)
    with aio.open_writer(dst, sr, ch) as w:
        for s0, block in aio.iter_blocks(src, sr * 10):
            w.write(_apply_edge_fade(block, s0, n, fade))
    return dst
