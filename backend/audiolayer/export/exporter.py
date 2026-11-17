"""Export cleaned mix, category stems, original audio, remuxed video and metadata."""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from ..audio import ffmpeg as ff
from ..core.errors import Conflict
from ..core.pipeline import stream_paths
from ..core.schemas import CATEGORIES, ExportOptions, Project
from ..core.store import ProjectStore
from ..separation.mixer import edge_fade_file, render_category, render_mix
from ..workers.jobs import JobContext
from . import reports


def export_project(store: ProjectStore, project_id: str, opts: ExportOptions, ctx: JobContext) -> dict[str, Any]:
    project: Project = store.load(project_id)
    if project.status != "analyzed" and any(i in opts.items for i in ("dialogue", "bgm", "sfx", "music", "noise")):
        raise Conflict("Analyze the project before exporting stems")
    streams = stream_paths(store, project_id)
    removed = list(opts.remove if opts.remove is not None else project.mix.removed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = store.path(project_id, "exports", stamp)
    work = store.path(project_id, "exports", f".work-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    master = store.path(project_id, "audio", "master.wav")
    base = project.name.replace(" ", "_")[:60] or "audio"
    files: list[str] = []
    skipped: list[dict[str, str]] = []
    info: dict[str, Any] = {"format": opts.format, "removed": removed, "loudness_preset": opts.loudness_preset,
                            "edge_fade_ms": opts.edge_fade_ms, "created_at": time.time()}
    items = list(dict.fromkeys(opts.items))
    steps = max(1, len(items))
    try:
        mix_wav: Path | None = None
        for i, item in enumerate(items):
            ctx.check_cancel()
            ctx.stage(f"export:{item}", i / steps, (i + 1) / steps, f"Exporting {item}")
            if item in ("mix", "video") and mix_wav is None:
                mix_wav = work / "mix.wav"
                if streams:
                    render_mix(streams, project.segments, removed, mix_wav, edge_fade_ms=opts.edge_fade_ms,
                               check_cancel=ctx.check_cancel)
                else:
                    if removed or any(s.muted or s.solo for s in project.segments):
                        skipped.append({"item": "mix", "reason": "No separation available: the cleaned mix "
                                        "equals the original audio (removal needs separation)."})
                    edge_fade_file(master, mix_wav, opts.edge_fade_ms)
                if opts.loudness_preset != "none":
                    normed = work / "mix_norm.wav"
                    info["loudness"] = ff.loudnorm(mix_wav, normed, opts.loudness_preset)
                    mix_wav = normed
                info["mix_loudness_after"] = ff.measure_loudness(mix_wav).model_dump()
            if item == "mix":
                assert mix_wav is not None
                suffix = "cleaned" if removed else "mix"
                name = f"{base}_{suffix}.{opts.format}"
                ff.encode(mix_wav, out_dir / name, opts.format)
                files.append(name)
            elif item in CATEGORIES:
                if not streams:
                    skipped.append({"item": item, "reason": "Stems need separation (not available for this analysis)."})
                    continue
                tmp = work / f"{item}.wav"
                render_category(streams, project.segments, item, tmp)  # type: ignore[arg-type]
                name = f"{base}_stem_{item}.{opts.format}"
                ff.encode(tmp, out_dir / name, opts.format)
                files.append(name)
            elif item == "original":
                name = f"{base}_original_audio.{opts.format}"
                ff.encode(master, out_dir / name, opts.format)
                files.append(name)
            elif item == "video":
                if not project.media.has_video:
                    skipped.append({"item": "video", "reason": "Source has no video stream."})
                    continue
                assert mix_wav is not None
                src = store.path(project_id, project.original_file)
                ext = src.suffix.lower() if src.suffix.lower() in (".mp4", ".mov", ".mkv") else ".mp4"
                name = f"{base}_cleaned_video{ext}"
                # master was extracted with first_pts=0, i.e. already aligned to the container start
                ff.mux_video(src, mix_wav, out_dir / name)
                files.append(name)
            elif item == "metadata":
                (out_dir / f"{base}_segments.csv").write_text(reports.segments_csv(project), encoding="utf-8")
                (out_dir / f"{base}_audacity_labels.txt").write_text(reports.audacity_labels(project), encoding="utf-8")
                files += [f"{base}_segments.csv", f"{base}_audacity_labels.txt"]
                srt = reports.dialogue_srt(project)
                if srt.strip():
                    (out_dir / f"{base}_dialogue.srt").write_text(srt, encoding="utf-8")
                    files.append(f"{base}_dialogue.srt")
                info["skipped"] = skipped
                (out_dir / f"{base}_metadata.json").write_text(reports.metadata_json(project, info), encoding="utf-8")
                files.append(f"{base}_metadata.json")
        info["skipped"] = skipped
        (out_dir / "README.txt").write_text(reports.readme(project, files, info), encoding="utf-8")
        files.append("README.txt")
        zip_name = None
        if opts.zip:
            ctx.stage("zip", 0.98, 1.0, "Packaging ZIP")
            zip_name = f"{base}_audiolayer_{stamp}.zip"
            with zipfile.ZipFile(store.path(project_id, "exports", zip_name), "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(out_dir / f, arcname=f"{base}_audiolayer/{f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    entry = {
        "id": stamp, "dir": f"exports/{stamp}", "files": files, "zip": f"exports/{zip_name}" if zip_name else None,
        "skipped": skipped, "info": info,
    }
    with store.edit(project_id) as pr:
        pr.exports.append(entry)
    return entry
