"""Separator adapter interface.

A separator splits the working master into two complementary signals that sum back to
the input: ``vocals`` (all voice: speech + singing) and ``accompaniment`` (everything
else: music + effects + ambience). Finer routing into categories happens afterwards using
the classifier (see ``audiolayer.separation.routing``), so any vocal/accompaniment model
(Demucs, MDX/Roformer, a cinematic DnR model…) can be dropped in.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ProgressFn = Callable[[float], None]
CancelFn = Callable[[], None]


class Separator(Protocol):
    name: str
    license: str

    def separate(self, master: Path, out_dir: Path, on_progress: ProgressFn | None = None,
                 check_cancel: CancelFn | None = None) -> dict[str, Path]:
        """Write ``vocals.wav`` and ``accompaniment.wav`` (same length/rate as master)."""
        ...
