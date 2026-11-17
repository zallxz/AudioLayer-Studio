"""Turn framewise probability tracks into time segments per category.

Design choices:
* hysteresis thresholds + gap merging → stable segments instead of flicker;
* every category is segmented independently → overlapping sources are preserved;
* confidence = mean probability inside the segment; segments below ``UNCERTAIN`` are kept
  but marked ``uncertain`` instead of being forced into a label;
* BGM vs foreground Music is decided by a transparent rule (dialogue overlap, singing,
  relative level, AudioSet "background/soundtrack" hints) and flagged ``role_uncertain``
  when the evidence is weak, so the user can correct it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from ..core.schemas import Category, LabelScore, Segment, Stream
from .base import LABEL_CATEGORY, FrameProbs, audioset_labels

UNCERTAIN = 0.35


@dataclass(frozen=True)
class SegParams:
    hi: float
    lo: float
    min_len: float
    max_gap: float


PARAMS: dict[str, SegParams] = {
    "dialogue": SegParams(0.30, 0.15, 0.30, 0.45),
    "music": SegParams(0.30, 0.15, 1.00, 2.00),
    "sfx": SegParams(0.18, 0.08, 0.15, 0.25),
    "noise": SegParams(0.35, 0.20, 0.50, 0.50),
}


def hysteresis(p: np.ndarray, hi: float, lo: float) -> np.ndarray:
    on = np.zeros(len(p), dtype=bool)
    state = False
    for i, v in enumerate(p):
        if not state and v >= hi:
            state = True
        elif state and v < lo:
            state = False
        on[i] = state
    return on


def regions(mask: np.ndarray, hop: float, max_gap: float, min_len: float) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out: list[list[int]] = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        if (i - out[-1][1]) * hop <= max_gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return [(a, b) for a, b in out if (b - a) * hop >= min_len]


def _labels_for(fp: FrameProbs, a: int, b: int, cats: set[str]) -> list[LabelScore]:
    if not fp.top_labels:
        return []
    names = audioset_labels()
    idx_of = {n: i for i, n in enumerate(names)}
    totals: dict[str, float] = {}
    for row in fp.top_labels[a:b]:
        for name, score in row:
            if LABEL_CATEGORY.get(idx_of.get(name, -1)) in cats:
                totals[name] = totals.get(name, 0.0) + score
    n = max(1, b - a)
    top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [LabelScore(label=k, score=round(v / n, 3)) for k, v in top]


_DESC = {
    "dialogue": "Speech",
    "music": "Music",
    "bgm": "Background music",
    "sfx": "Sound effect",
    "noise": "Noise",
}


def segment_track(fp: FrameProbs, track: str, category: Category, stream: Stream,
                  label_cats: set[str], calibrated: bool = True) -> list[Segment]:
    prm = PARAMS["music" if category in ("bgm", "music") else category]
    p = fp.probs[track]
    if len(p) == 0:
        return []
    p = median_filter(p, size=3, mode="nearest")
    mask = hysteresis(p, prm.hi if calibrated else prm.hi * 0.8, prm.lo if calibrated else prm.lo * 0.8)
    out: list[Segment] = []
    for a, b in regions(mask, fp.hop, prm.max_gap, prm.min_len):
        conf = float(np.clip(np.mean(p[a:b]), 0, 1))
        labels = _labels_for(fp, a, b, label_cats)
        desc = labels[0].label if labels else _DESC[category]
        flags = [] if calibrated else ["heuristic_classifier"]
        out.append(Segment(
            category=category, start=round(a * fp.hop, 3), end=round(b * fp.hop, 3),
            confidence=round(conf, 3), uncertain=(conf < UNCERTAIN) or not calibrated,
            stream=stream, description=desc, labels=labels, flags=flags,
        ))
    return out


def assign_music_role(seg: Segment, fp_music: FrameProbs, dialogue_track: np.ndarray | None,
                      rel_level_db: np.ndarray | None) -> None:
    """Decide BGM vs foreground Music for one music segment (mutates ``seg``)."""
    a = int(seg.start / fp_music.hop)
    b = max(a + 1, int(seg.end / fp_music.hop))
    sing = float(np.mean(fp_music.probs["singing"][a:b])) if b > a else 0.0
    bgm_hint = float(np.mean(fp_music.probs["bgm_hint"][a:b]))
    fg_hint = float(np.mean(fp_music.probs["fg_hint"][a:b]))
    dlg = 0.0
    if dialogue_track is not None and len(dialogue_track):
        dlg = float(np.mean(dialogue_track[a:b] > PARAMS["dialogue"].hi)) if b <= len(dialogue_track) else 0.0
    level = float(np.median(rel_level_db[a:b])) if rel_level_db is not None and b <= len(rel_level_db) and b > a else None

    score = 0.0  # >0 → foreground music, <0 → BGM
    reasons: list[str] = []
    if dlg > 0.15:
        # music playing under speech is, by definition, a background bed
        score -= 1.5 + dlg
        reasons.append(f"dialogue over {dlg:.0%} of it")
    if sing > 0.3:
        score += 1.5
        reasons.append("contains singing")
    if level is not None:
        if level > -4:
            score += 1.0
            reasons.append("dominates the mix")
        elif level < -12:
            score -= 1.0
            reasons.append("sits well below the mix")
    score += 1.5 * (fg_hint - bgm_hint)
    seg.category = "music" if score > 0 else "bgm"
    seg.description = ("Foreground music" if seg.category == "music" else "Background music") + (
        f" — {seg.labels[0].label}" if seg.labels and seg.labels[0].label not in ("Music",) else "")
    if abs(score) < 0.75:
        seg.flags.append("role_uncertain")
        seg.uncertain = True
    if reasons:
        seg.description += " (" + ", ".join(reasons) + ")"


def refine_boundaries(seg: Segment, env: np.ndarray, env_hop: float, search: float = 0.4) -> None:
    """Snap segment edges to the energy envelope of its stream (±``search`` s)."""
    if len(env) == 0:
        return
    a = int(seg.start / env_hop)
    b = int(seg.end / env_hop)
    lo, hi = max(0, a - int(search / env_hop)), min(len(env), b + int(search / env_hop))
    if hi - lo < 2:
        return
    region = env[lo:hi]
    peak = float(np.max(region))
    if peak <= 1e-6:
        return
    thr = peak * 0.08
    above = np.flatnonzero(region > thr)
    if len(above) == 0:
        return
    new_a = lo + int(above[0])
    new_b = lo + int(above[-1]) + 1
    if new_a <= a + int(search / env_hop):
        seg.start = round(new_a * env_hop, 3)
    if new_b >= b - int(search / env_hop):
        seg.end = round(new_b * env_hop, 3)


def envelope(x: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    h = max(1, int(sr * hop_s))
    m = np.abs(x if x.ndim == 1 else x.mean(axis=1))
    k = len(m) // h
    if k == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(m[: k * h].reshape(k, h) ** 2, axis=1))
