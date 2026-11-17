"""Metadata exports: JSON, CSV segment report, SRT transcript, Audacity / DAW marker labels."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..core.schemas import CATEGORY_DEFINITIONS, Project


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def srt_ts(t: float) -> str:
    return fmt_ts(t).replace(".", ",")


def metadata_json(project: Project, export_info: dict[str, Any]) -> str:
    data = {
        "schema": "audiolayer.metadata/v1",
        "project": {"id": project.id, "name": project.name, "created_at": project.created_at},
        "media": project.media.model_dump(),
        "category_definitions": CATEGORY_DEFINITIONS,
        "analysis": project.analysis.model_dump() if project.analysis else None,
        "mix": {"removed_categories": project.mix.removed},
        "segments": [s.model_dump() for s in project.segments],
        "export": export_info,
        "timebase": "seconds from the start of the extracted audio (aligned to video start; "
        f"audio_start_offset={project.media.audio_start_offset}s)",
    }
    return json.dumps(data, indent=2)


def segments_csv(project: Project) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "category", "start", "end", "duration", "start_tc", "end_tc", "confidence", "uncertain",
                "stream", "description", "flags", "muted", "edited", "original_category", "intelligibility",
                "transcript"])
    for s in project.segments:
        w.writerow([s.id, s.category, f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}", fmt_ts(s.start),
                    fmt_ts(s.end), f"{s.confidence:.3f}", s.uncertain, s.stream, s.description,
                    ";".join(s.flags), s.muted, s.edited, s.original_category or "",
                    "" if s.intelligibility is None else f"{s.intelligibility:.3f}", s.transcript or ""])
    return buf.getvalue()


def dialogue_srt(project: Project) -> str:
    lines: list[str] = []
    i = 0
    for s in sorted(project.segments, key=lambda x: x.start):
        if s.category != "dialogue" or not s.transcript:
            continue
        i += 1
        text = s.transcript
        if "unintelligible" in s.flags:
            text = f"[unintelligible] {text}"
        lines += [str(i), f"{srt_ts(s.start)} --> {srt_ts(s.end)}", text, ""]
    return "\n".join(lines)


def audacity_labels(project: Project) -> str:
    """Tab-separated label track (File → Import → Labels in Audacity; also readable by many DAWs)."""
    rows = []
    for s in sorted(project.segments, key=lambda x: x.start):
        label = f"{s.category.upper()} {s.confidence:.2f} {s.description}".strip()
        rows.append(f"{s.start:.6f}\t{s.end:.6f}\t{label}")
    return "\n".join(rows) + "\n"


def readme(project: Project, files: list[str], info: dict[str, Any]) -> str:
    removed = ", ".join(info.get("removed") or []) or "nothing"
    lims = project.analysis.limitations if project.analysis else []
    lines = [
        f"AudioLayer Studio export — {project.name}",
        "",
        "All audio files start at t=0 of the source video and have identical length, so they line up",
        "sample-accurately when dropped at the start of a timeline (silence is preserved).",
        f"Format: {info.get('format')} 48 kHz. Cleaned mix removes: {removed}.",
        f"Loudness preset: {info.get('loudness_preset')}.",
        "",
        "Stems sum back to the original (before any loudness normalisation). Noise stem = what the",
        "denoiser removed. Segment-level mutes apply to the cleaned mix only.",
        "",
        "Files:",
        *[f"  - {f}" for f in files],
        "",
        "Limitations reported by the analysis:",
        *([f"  - {x}" for x in lims] or ["  - none"]),
        "",
    ]
    return "\n".join(lines)
