"""The processing pipeline: ingest → analyze (detect/separate/classify) → render → export.

Detect → Classify → Preview → Isolate/Remove → Export.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import ffmpeg as ff
from ..audio import io as aio
from ..audio import quality
from ..audio.peaks import compute_peaks
from ..classification.base import FrameProbs
from ..classification.segmenter import assign_music_role, envelope, refine_boundaries, segment_track
from ..config import Settings
from ..models.registry import MODELS
from ..separation import denoise, routing
from ..separation.mixer import render_category
from ..workers.jobs import JobContext
from .components import Components
from .errors import InvalidMedia, LimitExceeded
from .paths import safe_filename
from .schemas import CATEGORIES, Analysis, AnalyzeOptions, MediaInfo, ModelUse, Project, Segment
from .store import ProjectStore

STREAM_FILES = ("voice", "music", "fx", "noise")


# ---------------------------------------------------------------------------- ingest
def ingest(store: ProjectStore, settings: Settings, src: Path, *, name: str | None = None,
           owner: str = "local", move: bool = False) -> Project:
    """Copy (or move) a media file into a new project, probe it and extract the master."""
    size = src.stat().st_size
    if size > settings.max_upload_mb * 1024 * 1024:
        raise LimitExceeded(f"File is {size / 1e6:.0f} MB; limit is {settings.max_upload_mb} MB")
    media = ff.probe(src)
    media.filename = safe_filename(name or src.name)
    dur = media.video_duration or media.audio_duration or 0.0
    if dur <= 0:
        raise InvalidMedia("Could not determine media duration (file may be truncated)")
    if dur > settings.max_duration_s:
        raise LimitExceeded(f"Media is {dur:.0f}s long; limit is {settings.max_duration_s:.0f}s",
                            hint="Raise AUDIOLAYER_MAX_DURATION_S or trim the file.")
    fname = safe_filename(name or src.name)
    project = Project(name=Path(fname).stem, owner=owner, media=media, original_file=f"original/{fname}")
    store.create_dirs(project.id)
    dst = store.path(project.id, project.original_file)
    try:
        if move:
            shutil.move(str(src), dst)
        else:
            shutil.copy2(src, dst)
        dst.chmod(0o444)  # the original is never modified
        master = store.path(project.id, "audio", "master.wav")
        channels = 1 if (media.channels or 2) == 1 else 2
        media.working_channels = channels
        if media.has_audio:
            ff.extract_audio(dst, master, channels=channels)
        else:
            ff.make_silence(master, dur, channels=channels)
        sr, ch, n = aio.info(master)
        media.working_sample_rate = sr
        if media.audio_duration is None:
            media.audio_duration = n / sr
        project.media = media
        store.save(project)
    except Exception:
        shutil.rmtree(store.project_dir(project.id), ignore_errors=True)
        raise
    return project


# ---------------------------------------------------------------------------- analyze
def _fp_to_json(fp: FrameProbs) -> dict[str, Any]:
    return {"hop": fp.hop, "probs": {k: np.round(v, 4).tolist() for k, v in fp.probs.items()}}


def _stream_env(path: Path, hop_s: float = 0.01) -> np.ndarray:
    sr, _, _ = aio.info(path)
    parts = [envelope(b, sr, hop_s) for _, b in aio.iter_blocks(path, int(sr * hop_s) * 3000)]
    return np.concatenate(parts) if parts else np.zeros(0)


def analyze(store: ProjectStore, settings: Settings, comps: Components, project_id: str,
            opts: AnalyzeOptions, ctx: JobContext) -> dict[str, Any]:
    t0 = time.time()
    timings: dict[str, float] = {}
    project = store.load(project_id)
    master = store.path(project_id, "audio", "master.wav")
    sr, ch, n = aio.info(master)
    duration = n / sr
    limitations: list[str] = list(comps.notes)
    models: list[ModelUse] = []

    # 1) quality / creator-issue analysis -----------------------------------------
    ctx.stage("quality", 0.0, 0.08, "Measuring loudness and audio problems")
    loud = ff.measure_loudness(master)
    issues, noise_segs, profile = quality.analyze(master, project.media, loud)
    timings["quality"] = time.time() - t0
    ctx.check_cancel()

    silent = any(i.kind in ("silent_audio", "no_audio_track") for i in issues)
    separated = bool(opts.separate and comps.separator is not None and not silent)
    if opts.separate and comps.separator is None:
        limitations.append("Separation was requested but no separator is installed; results are detection-only.")

    clf = comps.classifier
    models.append(ModelUse(role="classification", name=clf.name, license=clf.license,
                           note="" if clf.calibrated else "heuristic, low accuracy"))
    segments: list[Segment] = []
    fp_voice: FrameProbs | None = None
    fp_acc: FrameProbs | None = None
    sep_dir = store.path(project_id, "analysis")
    streams_dir = store.path(project_id, "streams")
    for old in streams_dir.glob("*.wav"):
        old.unlink()

    if silent:
        pass
    elif separated:
        assert comps.separator is not None
        # 2) neural separation: vocals / accompaniment ---------------------------
        t = time.time()
        ctx.stage("separation", 0.08, 0.55, f"Separating sources with {comps.separator.name}")
        sep = comps.separator.separate(master, sep_dir, on_progress=ctx.progress, check_cancel=ctx.check_cancel)
        models.append(ModelUse(role="separation", name=comps.separator.name, license=comps.separator.license))
        timings["separation"] = time.time() - t

        # 3) classify each separated signal (overlap-aware) ------------------------
        t = time.time()
        ctx.stage("classification", 0.55, 0.75, f"Detecting sound events with {clf.name}")
        fp_voice = clf.frame_probs(sep["vocals"], ctx.sub(0, 0.5), ctx.check_cancel)
        fp_acc = clf.frame_probs(sep["accompaniment"], ctx.sub(0.5, 1.0), ctx.check_cancel)
        timings["classification"] = time.time() - t

        # 4) noise chain + stream routing --------------------------------------
        t = time.time()
        force = opts.denoise == "on"
        use_noise = opts.denoise != "off" and (profile.has_removable_noise or force)
        chain = denoise.build_chain(profile, sr, force=force) if use_noise else None
        steps = denoise.describe(profile, force) if use_noise else []
        if steps:
            models.append(ModelUse(role="noise reduction", name="; ".join(steps),
                                   license=MODELS["noisereduce"].license + " / BSD (scipy)"))
        ctx.stage("routing", 0.75, 0.85, "Building category streams")
        w = routing.weights(fp_voice, fp_acc)
        streams = routing.build_streams(sep["vocals"], sep["accompaniment"], streams_dir, w, fp_acc.hop, chain,
                                        on_progress=ctx.progress, check_cancel=ctx.check_cancel)
        timings["routing"] = time.time() - t

        # 5) segments ---------------------------------------------------------------
        music_fp = FrameProbs(hop=fp_acc.hop, probs=dict(fp_acc.probs), top_labels=fp_acc.top_labels)
        m = min(len(fp_acc.probs["music"]), len(fp_voice.probs["singing"]))
        music_track = fp_acc.probs["music"].copy()
        music_track[:m] = np.maximum(music_track[:m], fp_voice.probs["singing"][:m] * fp_acc.probs["music"][:m] * 2)
        music_fp.probs["music"] = np.clip(music_track, 0, 1)
        music_fp.probs["singing"] = np.zeros_like(music_track)
        music_fp.probs["singing"][:m] = fp_voice.probs["singing"][:m]
        cal = bool(clf.calibrated)
        dlg = segment_track(fp_voice, "dialogue", "dialogue", "voice", {"dialogue"}, cal)
        mus = segment_track(music_fp, "music", "bgm", "music", {"music", "singing"}, cal)
        fx = segment_track(fp_acc, "sfx", "sfx", "fx", {"sfx"}, cal)
        # relative level of music stream vs the mix, per classifier frame
        env_mix = _stream_env(master, fp_acc.hop)
        env_music = _stream_env(streams["music"], fp_acc.hop)
        k = min(len(env_mix), len(env_music))
        rel = 20 * np.log10((env_music[:k] + 1e-9) / (env_mix[:k] + 1e-9))
        for s in mus:
            assign_music_role(s, music_fp, fp_voice.probs["dialogue"], rel)
        overlap = w["overlap"]
        for s in fx:
            a, b = int(s.start / fp_acc.hop), int(s.end / fp_acc.hop) + 1
            if np.any(overlap[a:b]):
                s.flags.append("overlaps_music_split_estimated")
        envs = {"voice": _stream_env(streams["voice"]), "fx": _stream_env(streams["fx"])}
        for s in dlg:
            refine_boundaries(s, envs["voice"], 0.01)
        for s in fx:
            refine_boundaries(s, envs["fx"], 0.01, search=0.3)
        clf_noise = segment_track(fp_acc, "noise", "noise", "noise" if use_noise else "mix", {"noise"}, cal)
        for s in clf_noise:
            if not use_noise:
                s.flags.append("not_removable")
            s.flags.append("detected_by_classifier")
        segments = dlg + mus + fx + clf_noise
        for s in noise_segs:
            if s.stream == "noise" and not use_noise:
                s.stream = "mix"
                s.flags.append("not_removable")
        segments += noise_segs
        n_overlap = sum("overlaps_music_split_estimated" in s.flags for s in fx)
        if n_overlap:
            limitations.append(
                f"{n_overlap} sound-effect segment(s) overlap music. The separator splits voice from "
                "everything else; music vs effects inside the accompaniment is a soft time-based estimate "
                "there, so some music may remain in SFX (and vice versa)."
            )
        if profile.clipping_regions:
            limitations.append("Clipping is baked into the waveform; it is marked but cannot be removed.")
    else:
        # detection-only path ---------------------------------------------------------
        t = time.time()
        ctx.stage("classification", 0.08, 0.8, f"Detecting sound events with {clf.name}")
        fp_mix = clf.frame_probs(master, ctx.progress, ctx.check_cancel)
        timings["classification"] = time.time() - t
        cal = bool(clf.calibrated)
        dlg = segment_track(fp_mix, "dialogue", "dialogue", "mix", {"dialogue"}, cal)
        mus = segment_track(fp_mix, "music", "bgm", "mix", {"music", "singing"}, cal)
        for s in mus:
            assign_music_role(s, fp_mix, fp_mix.probs["dialogue"], None)
        fx = segment_track(fp_mix, "sfx", "sfx", "mix", {"sfx"}, cal)
        clf_noise = segment_track(fp_mix, "noise", "noise", "mix", {"noise"}, cal)
        segments = dlg + mus + fx + clf_noise + [s for s in noise_segs]
        for s in segments:
            s.stream = "mix"
            if "not_removable" not in s.flags:
                s.flags.append("not_removable")
        fp_acc = fp_mix
        if not silent:
            limitations.append("Detection only: run analysis with separation to isolate or remove categories.")

    # 6) speech recognition --------------------------------------------------------
    asr_name = "none"
    dlg_segs = [s for s in segments if s.category == "dialogue"]
    if comps.asr is not None and opts.transcribe and not silent and (dlg_segs or not separated):
        t = time.time()
        ctx.stage("transcription", 0.85, 0.95, "Transcribing dialogue")
        src = streams_dir / "voice.wav" if separated else master
        try:
            from ..classification.speech import attach_transcripts

            asr = comps.asr.transcribe(src, opts.language, ctx.progress, ctx.check_cancel)
            asr_name = asr.model
            others = [s for s in segments if s.category != "dialogue"]
            dlg_new = attach_transcripts(dlg_segs, asr, duration)
            for s in dlg_new:
                if separated and s.stream == "mix":
                    s.stream = "voice"
                if not separated and "not_removable" not in s.flags:
                    s.flags.append("not_removable")
                    s.stream = "mix"
            segments = others + dlg_new
            models.append(ModelUse(role="speech recognition", name=asr.model, license=comps.asr.license))
            store.write_json(project_id, "analysis/transcript.json", {
                "language": asr.language, "language_probability": asr.language_probability,
                "segments": [s.__dict__ for s in asr.segments],
            })
            n_bad = sum("unintelligible" in s.flags for s in dlg_new)
            if n_bad:
                from .schemas import QualityIssue

                issues.append(QualityIssue(
                    kind="unintelligible_dialogue", severity="warning",
                    title=f"{n_bad} unintelligible dialogue segment(s)",
                    detail="Speech-like audio that the recogniser could not turn into confident words — typical of "
                    "AI 'gibberish' voices. Mute those segments or remove Dialogue (keeps music & effects).",
                    fix="remove_gibberish",
                ))
        except Exception as e:  # noqa: BLE001 - ASR is optional; never fail the analysis
            limitations.append(f"Transcription failed: {e}")
        timings["transcription"] = time.time() - t

    segments.sort(key=lambda s: (s.start, s.category))
    for s in segments:
        s.original_category = s.category
        s.start = max(0.0, min(s.start, duration))
        s.end = max(s.start, min(s.end, duration))

    presence: dict[str, float] = {c: round(sum(s.duration for s in segments if s.category == c) / max(duration, 1e-6), 3)
                for c in CATEGORIES}
    analysis = Analysis(
        classifier=clf.name, separator=comps.separator.name if (separated and comps.separator) else "none", asr=asr_name,
        frame_hop=fp_acc.hop if fp_acc else 0.0, separated=separated, loudness=loud, issues=issues,
        limitations=limitations, models=models, category_presence=presence, timings=timings,
    )
    if fp_acc is not None:
        store.write_json(project_id, "analysis/frames.json", {
            "accompaniment_or_mix": _fp_to_json(fp_acc),
            "voice": _fp_to_json(fp_voice) if fp_voice else None,
        })

    # 7) preview stems + peaks ----------------------------------------------------------
    ctx.stage("render", 0.95, 1.0, "Rendering preview stems")
    with store.edit(project_id) as pr:
        pr.analysis = analysis
        pr.segments = segments
        pr.history.clear()
        pr.redo.clear()
        pr.status = "analyzed"
        pr.error = None
        pr.mix.removed = []
    if separated:
        render_preview_stems(store, project_id)
    peaks = compute_peaks(master)
    store.path(project_id, "analysis", "peaks.json").write_text(json.dumps(peaks))
    timings["total"] = time.time() - t0
    with store.edit(project_id) as pr:
        if pr.analysis:
            pr.analysis.timings = timings
    return {"project_id": project_id, "segments": len(segments), "separated": separated,
            "issues": len(issues), "limitations": limitations}


def stream_paths(store: ProjectStore, project_id: str) -> dict[str, Path] | None:
    d = store.path(project_id, "streams")
    paths = {k: d / f"{k}.wav" for k in STREAM_FILES}
    return paths if all(p.exists() for p in paths.values()) else None


def render_preview_stems(store: ProjectStore, project_id: str) -> int:
    streams = stream_paths(store, project_id)
    if not streams:
        return 0
    project = store.load(project_id)
    for c in CATEGORIES:
        tmp = store.path(project_id, "stems", f".{c}.tmp.flac")
        render_category(streams, project.segments, c, tmp, subtype="PCM_16", fmt="FLAC")
        tmp.replace(store.path(project_id, "stems", f"{c}.flac"))
    with store.edit(project_id) as pr:
        pr.stems_version += 1
        return pr.stems_version


def load_peaks(store: ProjectStore, project_id: str) -> dict[str, Any]:
    p = store.path(project_id, "analysis", "peaks.json")
    if p.exists():
        return dict(json.loads(p.read_text()))
    peaks = compute_peaks(store.path(project_id, "audio", "master.wav"))
    p.write_text(json.dumps(peaks))
    return peaks


def media_summary(media: MediaInfo) -> dict[str, Any]:
    return media.model_dump()
