"""Classifier adapter interface and AudioSet → category mapping."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

# Probability tracks every classifier must return (each in [0, 1], one value per frame).
TRACKS = ("dialogue", "singing", "music", "sfx", "noise", "bgm_hint", "fg_hint")

ProgressFn = Callable[[float], None]
CancelFn = Callable[[], None]


@dataclass
class FrameProbs:
    hop: float
    probs: dict[str, np.ndarray]
    # Optional per-frame top labels with scores (for human-readable descriptions)
    top_labels: list[list[tuple[str, float]]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(next(iter(self.probs.values()))) if self.probs else 0

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop


class Classifier(Protocol):
    name: str
    license: str
    calibrated: bool  # False for heuristic fallbacks: confidences are only indicative

    def frame_probs(self, path: Path, on_progress: ProgressFn | None = None,
                    check_cancel: CancelFn | None = None) -> FrameProbs: ...


# ---- AudioSet ontology mapping (indices into the 527-class label list) ----------------
_DATA = Path(__file__).resolve().parent.parent / "models" / "data" / "audioset_labels.csv"


@lru_cache(maxsize=1)
def audioset_labels() -> list[str]:
    with _DATA.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return [r[2] for r in rows]


def _r(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


GROUPS: dict[str, list[int]] = {
    # speech and other vocal, non-musical human sounds
    "dialogue": _r(0, 26) + [38, 39] + _r(41, 50),
    "singing": _r(27, 37) + [254, 255],
    # instruments + genres (bells 200-207 count as SFX)
    "music": [137] + _r(138, 199) + _r(208, 282),
    "noise": [285, 496, 513, 515, 516, 517, 518, 520, 521],
    "bgm_hint": [246, 253, 267, 268, 270, 272],
    "fg_hint": [216, 219, 238, 266] + _r(27, 36),
}
IGNORED = set([500, 501, 502, 503, 505] + _r(506, 512) + [524, 525, 526])
_ASSIGNED = set(GROUPS["dialogue"]) | set(GROUPS["singing"]) | set(GROUPS["music"]) | set(GROUPS["noise"])
GROUPS["sfx"] = [i for i in range(527) if i not in _ASSIGNED and i not in IGNORED]

LABEL_CATEGORY: dict[int, str] = {}
for _cat in ("dialogue", "singing", "music", "noise", "sfx"):
    for _i in GROUPS[_cat]:
        LABEL_CATEGORY.setdefault(_i, _cat)


def audioset_to_tracks(p: np.ndarray) -> dict[str, np.ndarray]:
    """(frames, 527) probabilities -> category tracks (max over member labels)."""
    return {k: p[:, idx].max(axis=1) for k, idx in GROUPS.items()}
