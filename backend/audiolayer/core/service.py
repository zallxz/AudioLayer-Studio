"""Application service: the single façade used by both the REST API and the MCP server.

Every public method takes an ``owner`` (authenticated principal). In local single-user
mode the owner is ``"local"``; with ``AUDIOLAYER_AUTH_TOKENS`` set, each token maps to a
principal and projects/jobs are only visible to their owner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Settings
from ..export.exporter import export_project
from ..separation.mixer import render_mix
from ..workers.jobs import JobContext, JobManager
from . import pipeline
from .components import Components, build_components, system_info
from .errors import Conflict, Forbidden, NotFound
from .paths import validate_input_path
from .schemas import (
    CATEGORIES,
    CATEGORY_DEFINITIONS,
    AnalyzeOptions,
    Category,
    ExportOptions,
    Job,
    Project,
    Segment,
)
from .store import ProjectStore

PRESETS: dict[str, dict[str, Any]] = {
    "remove_music": {"remove": ["bgm", "music"], "label": "Remove all music (BGM + foreground)",
                     "why": "Generator added music you did not ask for."},
    "dialogue_only": {"remove": ["bgm", "music", "sfx", "noise"], "label": "Dialogue only",
                      "why": "Clean voice track for re-scoring or captions."},
    "music_and_effects": {"remove": ["dialogue"], "label": "Music & Effects (M&E)",
                          "why": "Replace the AI voice / dub into another language while keeping the soundscape."},
    "remove_noise": {"remove": ["noise"], "label": "Remove noise", "why": "Hiss, hum, rumble, DC."},
    "remove_sfx": {"remove": ["sfx"], "label": "Remove sound effects", "why": "Random or wrong SFX."},
    "remove_gibberish": {"remove": [], "mute_flag": "unintelligible", "label": "Mute gibberish dialogue",
                         "why": "Mute only dialogue segments the recogniser flags as unintelligible."},
    "clear": {"remove": [], "label": "Original (nothing removed)", "why": "Reset the selection."},
}
HISTORY_LIMIT = 50


def _retitle(seg: Segment, category: str) -> None:
    """Keep a music segment's description in step with its BGM/Music role."""
    if category not in ("bgm", "music"):
        return
    old, new = ("Background music", "Foreground music") if category == "music" else ("Foreground music", "Background music")
    if seg.description.startswith(old):
        seg.description = new + seg.description[len(old):]


class AudioLayerService:
    def __init__(self, settings: Settings, components: Components | None = None):
        self.settings = settings
        settings.ensure_dirs()
        self.store = ProjectStore(settings)
        self.components = components or build_components(settings)
        self.jobs = JobManager(self.store, settings.workers)

    # ---- system ---------------------------------------------------------------
    def system(self) -> dict[str, Any]:
        info = system_info(self.settings, self.components)
        info["categories"] = CATEGORY_DEFINITIONS
        info["presets"] = {k: {"label": v["label"], "why": v["why"], "remove": v["remove"]} for k, v in PRESETS.items()}
        return info

    # ---- projects -------------------------------------------------------------
    def create_from_file(self, path: Path, filename: str, owner: str, *, move: bool = True) -> Project:
        return pipeline.ingest(self.store, self.settings, path, name=filename, owner=owner, move=move)

    def create_from_local_path(self, raw: str, owner: str) -> Project:
        roots = list(self.settings.input_dirs)
        p = validate_input_path(raw, roots)
        return pipeline.ingest(self.store, self.settings, p, owner=owner, move=False)

    def list_projects(self, owner: str) -> list[Project]:
        return self.store.list(owner=owner)

    def get(self, project_id: str, owner: str) -> Project:
        return self.store.load(project_id, owner=owner)

    def delete(self, project_id: str, owner: str) -> None:
        if self.jobs.active_for(project_id):
            raise Conflict("Project has running jobs; cancel them first")
        self.store.delete(project_id, owner=owner)

    # ---- jobs -----------------------------------------------------------------
    def _busy(self, project_id: str, kinds: tuple[str, ...] = ("analyze",)) -> None:
        for j in self.jobs.active_for(project_id):
            if j.kind in kinds:
                raise Conflict(f"A {j.kind} job is already running for this project", hint=f"job_id={j.id}")

    def analyze(self, project_id: str, opts: AnalyzeOptions, owner: str) -> Job:
        self.get(project_id, owner)
        self._busy(project_id, ("analyze", "render_stems", "export"))
        with self.store.edit(project_id) as pr:
            pr.status = "analyzing"
            pr.error = None

        def run(ctx: JobContext) -> dict[str, Any]:
            return pipeline.analyze(self.store, self.settings, self.components, project_id, opts, ctx)

        def done(job: Job) -> None:
            if job.status != "succeeded":
                with self.store.edit(project_id) as pr:
                    pr.status = "error" if job.status == "failed" else ("analyzed" if pr.analysis else "uploaded")
                    pr.error = job.error

        return self.jobs.submit(project_id, "analyze", run, owner=owner, on_done=done)

    def job(self, job_id: str, owner: str) -> Job:
        job = self.jobs.get(job_id)
        if job.owner != owner:
            raise NotFound(f"Job {job_id} not found")
        return job

    def cancel(self, job_id: str, owner: str) -> Job:
        self.job(job_id, owner)
        return self.jobs.cancel(job_id)

    # ---- segments & mix state ------------------------------------------------------
    def _snapshot(self, pr: Project) -> None:
        pr.history.append({"segments": [s.model_dump() for s in pr.segments], "removed": list(pr.mix.removed)})
        del pr.history[:-HISTORY_LIMIT]
        pr.redo.clear()

    def segments(self, project_id: str, owner: str, category: str | None = None) -> list[Segment]:
        pr = self.get(project_id, owner)
        segs = pr.segments
        if category:
            if category not in CATEGORIES:
                raise NotFound(f"Unknown category {category!r}")
            segs = [s for s in segs if s.category == category]
        return segs

    def update_segment(self, project_id: str, segment_id: str, owner: str, *, category: Category | None = None,
                       muted: bool | None = None, solo: bool | None = None, start: float | None = None,
                       end: float | None = None) -> dict[str, Any]:
        self.get(project_id, owner)
        rerender = False
        with self.store.edit(project_id) as pr:
            seg = next((s for s in pr.segments if s.id == segment_id), None)
            if seg is None:
                raise NotFound(f"Segment {segment_id} not found")
            self._snapshot(pr)
            dur = pr.media.audio_duration or 1e9
            if category is not None and category != seg.category:
                if category not in CATEGORIES:
                    raise NotFound(f"Unknown category {category!r}")
                _retitle(seg, category)
                seg.category = category
                seg.edited = True
                seg.uncertain = False
                if "role_uncertain" in seg.flags:
                    seg.flags.remove("role_uncertain")
                rerender = True
            if start is not None or end is not None:
                ns = seg.start if start is None else max(0.0, float(start))
                ne = seg.end if end is None else min(dur, float(end))
                if ne <= ns:
                    raise Conflict("Segment end must be after start")
                seg.start, seg.end, seg.edited = round(ns, 3), round(ne, 3), True
                rerender = True
            if muted is not None:
                seg.muted = bool(muted)
            if solo is not None:
                seg.solo = bool(solo)
            separated = bool(pr.analysis and pr.analysis.separated)
            result = seg.model_dump()
        job = self._rerender(project_id, owner) if (rerender and separated) else None
        return {"segment": result, "render_job": job.model_dump() if job else None,
                "note": None if separated or not rerender else
                "Label saved. This analysis has no separation, so the change affects reports only."}

    def _rerender(self, project_id: str, owner: str) -> Job:
        def run(ctx: JobContext) -> dict[str, Any]:
            ctx.stage("render", 0, 1, "Re-rendering category stems")
            return {"stems_version": pipeline.render_preview_stems(self.store, project_id)}

        return self.jobs.submit(project_id, "render_stems", run, owner=owner)

    def set_removed(self, project_id: str, categories: Sequence[str], owner: str) -> Project:
        bad = [c for c in categories if c not in CATEGORIES]
        if bad:
            raise NotFound(f"Unknown categories: {bad}")
        self.get(project_id, owner)
        with self.store.edit(project_id) as pr:
            self._snapshot(pr)
            pr.mix.removed = [c for c in CATEGORIES if c in categories]  # type: ignore[misc]
            return pr

    def apply_preset(self, project_id: str, preset: str, owner: str) -> Project:
        if preset not in PRESETS:
            raise NotFound(f"Unknown preset {preset!r}", hint="Presets: " + ", ".join(PRESETS))
        self.get(project_id, owner)
        p = PRESETS[preset]
        with self.store.edit(project_id) as pr:
            self._snapshot(pr)
            pr.mix.removed = list(p["remove"])
            flag = p.get("mute_flag")
            for s in pr.segments:
                s.solo = False
                if flag:
                    s.muted = flag in s.flags
                elif preset == "clear":
                    s.muted = False
            return pr

    def undo(self, project_id: str, owner: str, *, redo: bool = False) -> Project:
        self.get(project_id, owner)
        with self.store.edit(project_id) as pr:
            src, dst = (pr.redo, pr.history) if redo else (pr.history, pr.redo)
            if not src:
                raise Conflict("Nothing to " + ("redo" if redo else "undo"))
            state = src.pop()
            dst.append({"segments": [s.model_dump() for s in pr.segments], "removed": list(pr.mix.removed)})
            labels_before = [(s.id, s.category, s.start, s.end) for s in pr.segments]
            pr.segments = [Segment.model_validate(s) for s in state["segments"]]
            pr.mix.removed = state["removed"]
            changed = labels_before != [(s.id, s.category, s.start, s.end) for s in pr.segments]
            separated = bool(pr.analysis and pr.analysis.separated)
        if changed and separated:
            self._rerender(project_id, owner)
        return self.get(project_id, owner)

    def reset_edits(self, project_id: str, owner: str) -> Project:
        self.get(project_id, owner)
        with self.store.edit(project_id) as pr:
            self._snapshot(pr)
            for s in pr.segments:
                if s.original_category:
                    _retitle(s, s.original_category)
                    s.category = s.original_category
                s.muted = s.solo = s.edited = False
            pr.mix.removed = []
            separated = bool(pr.analysis and pr.analysis.separated)
        if separated:
            self._rerender(project_id, owner)
        return self.get(project_id, owner)

    # ---- rendering & export ----------------------------------------------------------
    def limitations_for(self, pr: Project, removed: Sequence[str]) -> list[str]:
        out: list[str] = []
        if not pr.analysis:
            return ["Project has not been analyzed yet."]
        if not pr.analysis.separated:
            out.append("No separation for this analysis: removal cannot change the audio. Re-run analysis with "
                       "separation enabled.")
            return out
        if "sfx" in removed or "bgm" in removed or "music" in removed:
            n = sum("overlaps_music_split_estimated" in s.flags for s in pr.segments)
            if n:
                out.append(f"{n} SFX segment(s) overlap music; the music/SFX split there is estimated.")
        if "bgm" in removed and "music" not in removed or "music" in removed and "bgm" not in removed:
            n = sum("role_uncertain" in s.flags for s in pr.segments if s.category in ("bgm", "music"))
            if n:
                out.append(f"{n} music segment(s) have an uncertain BGM/Music role — check them in the timeline.")
        if "noise" in removed:
            if not any(s.category == "noise" and s.stream == "noise" for s in pr.segments):
                out.append("No removable noise was detected; the noise stem is (near) silent.")
            if any("clipping" in s.flags for s in pr.segments):
                out.append("Clipping cannot be removed by separation.")
        if "dialogue" in removed:
            out.append("Removing dialogue can leave faint vocal artifacts in music/effects; check sung parts.")
        return out

    def render_preview(self, project_id: str, owner: str, remove: Sequence[str] | None = None) -> Job:
        pr = self.get(project_id, owner)
        streams = pipeline.stream_paths(self.store, project_id)
        if not streams:
            raise Conflict("No separated streams for this project", hint="Run analyze_media with separate=true")
        removed = list(pr.mix.removed if remove is None else remove)
        key = hashlib.sha256(json.dumps({
            "r": sorted(removed), "s": [(s.id, s.category, s.start, s.end, s.muted, s.solo) for s in pr.segments],
        }).encode()).hexdigest()[:12]

        def run(ctx: JobContext) -> dict[str, Any]:
            ctx.stage("render", 0, 1, "Rendering preview mix")
            dst = self.store.path(project_id, "renders", f"preview_{key}.wav")
            if not dst.exists():
                tmp = dst.with_suffix(".tmp.wav")
                render_mix(streams, pr.segments, removed, tmp, check_cancel=ctx.check_cancel)
                tmp.replace(dst)
            return {"file": f"renders/{dst.name}", "removed": removed,
                    "limitations": self.limitations_for(pr, removed)}

        return self.jobs.submit(project_id, "render_preview", run, owner=owner)

    def export(self, project_id: str, opts: ExportOptions, owner: str) -> Job:
        pr = self.get(project_id, owner)
        self._busy(project_id, ("analyze",))
        removed = list(opts.remove if opts.remove is not None else pr.mix.removed)

        def run(ctx: JobContext) -> dict[str, Any]:
            res = export_project(self.store, project_id, opts, ctx)
            res["limitations"] = self.limitations_for(self.store.load(project_id), removed)
            return res

        return self.jobs.submit(project_id, "export", run, owner=owner)

    def resolve_file(self, project_id: str, rel: str, owner: str) -> Path:
        self.get(project_id, owner)
        allowed = ("exports/", "renders/", "stems/", "audio/master.wav", "original/")
        if not any(rel == a or rel.startswith(a) for a in allowed):
            raise Forbidden("File is not downloadable")
        p = self.store.path(project_id, rel)
        if not p.is_file():
            raise NotFound(f"File not found: {rel}")
        return p

    def quality_report(self, project_id: str, owner: str) -> dict[str, Any]:
        pr = self.get(project_id, owner)
        if not pr.analysis:
            raise Conflict("Project has not been analyzed yet")
        a = pr.analysis
        return {"loudness": a.loudness.model_dump(), "issues": [i.model_dump() for i in a.issues],
                "limitations": a.limitations, "category_presence": a.category_presence,
                "presets": {k: v["label"] for k, v in PRESETS.items()}}
