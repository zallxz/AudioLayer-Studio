"""AudioLayer Studio MCP server (official MCP Python SDK v2, ``MCPServer``).

Local (Claude Desktop / Claude Code):   audiolayer-mcp                 (stdio)
Remote (custom connector, behind TLS):  audiolayer-api  → mounted at /mcp (Streamable HTTP)

The package directory is called ``mcp_server`` (not ``mcp``) so it never shadows the SDK.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from ..api.auth import current_principal
from ..config import get_settings
from ..core.errors import AudioLayerError
from ..core.schemas import CATEGORIES, AnalyzeOptions, Category, ExportOptions, Job
from ..core.service import PRESETS, AudioLayerService

log = logging.getLogger("audiolayer.mcp")

INSTRUCTIONS = """AudioLayer Studio cleans up audio from AI-generated videos.
Workflow: analyze_media → poll get_processing_status until succeeded → get_analysis /
list_audio_segments → optionally update_segment_category → remove_audio_category or a
cleanup preset → export_cleaned_audio / export_audio_stems → poll get_processing_status for
output files. Categories: dialogue, bgm (background music), sfx, music (foreground music /
songs), noise (hiss, hum, rumble — never intentional ambience). Long operations return a
job_id immediately; never assume they finished. Results include confidence scores and an
explicit list of limitations — relay them to the user instead of overstating quality."""

T = TypeVar("T")
CategoryList = Annotated[list[Category], Field(description="Categories: dialogue, bgm, sfx, music, noise")]


def _wrap(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def inner(*a: Any, **kw: Any) -> T:
        try:
            return fn(*a, **kw)
        except AudioLayerError as e:
            msg = f"[{e.code}] {e.message}" + (f" Hint: {e.hint}" if e.hint else "")
            raise ToolError(msg) from e

    return inner


def build_server(service: AudioLayerService, *, remote: bool = False) -> MCPServer:
    settings = service.settings
    mcp = MCPServer(
        name="audiolayer-studio", title="AudioLayer Studio", version="0.1.0", instructions=INSTRUCTIONS,
        website_url="https://github.com/zallxz/AudioLayer-Studio",
    )

    def owner() -> str:
        return current_principal.get() if remote else "local"

    def file_ref(project_id: str, rel: str) -> dict[str, str]:
        ref = {"relative_path": rel}
        if remote:
            base = settings.public_base_url
            ref["download_url"] = f"{base}/api/projects/{project_id}/files/{rel}"
        else:
            ref["path"] = str(service.store.path(project_id, rel))
        return ref

    def job_view(job: Job) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": job.id, "project_id": job.project_id, "kind": job.kind, "status": job.status,
            "progress": job.progress, "stage": job.stage, "message": job.message, "error": job.error,
        }
        if job.status == "succeeded":
            res = dict(job.result)
            if job.kind == "export":
                out["files"] = [file_ref(job.project_id, f"{res['dir']}/{f}") for f in res.get("files", [])]
                if res.get("zip"):
                    out["zip"] = file_ref(job.project_id, res["zip"])
                out["skipped"] = res.get("skipped", [])
                out["limitations"] = res.get("limitations", [])
            elif job.kind == "render_preview":
                out["file"] = file_ref(job.project_id, res["file"])
                out["limitations"] = res.get("limitations", [])
            else:
                out["result"] = res
        if job.status in ("queued", "running"):
            out["next"] = "Call get_processing_status again in a few seconds."
        return out

    ro = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    rw = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    @mcp.tool(annotations=rw)
    @_wrap
    def analyze_media(
        file_path: Annotated[str | None, Field(description="Local media path inside an allowed input folder "
                                               "(local server only). Supported: mp4, mov, mkv, webm, wav, mp3, aac, m4a, flac.")] = None,
        project_id: Annotated[str | None, Field(description="Re-analyze an existing/uploaded project instead.")] = None,
        separate: Annotated[bool, Field(description="Run source separation (required to isolate/remove).")] = True,
        transcribe: Annotated[bool, Field(description="Transcribe dialogue + flag gibberish speech.")] = True,
        denoise: Annotated[Literal["auto", "on", "off"], Field(description="Noise stem: auto = only if hiss/hum detected")] = "auto",
        language: Annotated[str | None, Field(description="ISO language hint for transcription, e.g. 'en'.")] = None,
    ) -> dict[str, Any]:
        """Start analysis of a media file: extract audio, detect problems, separate sources and classify
        dialogue / BGM / SFX / music / noise over time. Returns immediately with a job_id; poll
        get_processing_status. The original file is never modified."""
        if bool(file_path) == bool(project_id):
            raise AudioLayerError("Provide exactly one of file_path or project_id")
        if file_path:
            if remote:
                raise AudioLayerError("file_path is not available on the remote server",
                                      hint="Use create_upload_url, upload the file, then pass project_id.")
            pr = service.create_from_local_path(file_path, owner())
        else:
            assert project_id is not None
            pr = service.get(project_id, owner())
        job = service.analyze(pr.id, AnalyzeOptions(separate=separate, transcribe=transcribe, denoise=denoise,
                                                    language=language), owner())
        return {"project_id": pr.id, "job_id": job.id, "status": job.status, "media": pr.media.model_dump()}

    @mcp.tool(annotations=ro)
    @_wrap
    def get_processing_status(job_id: str) -> dict[str, Any]:
        """Progress and status of a job (analysis, preview render or export). When an export succeeds this
        lists output files (local paths, or download URLs on the remote server)."""
        return job_view(service.job(job_id, owner()))

    @mcp.tool(annotations=rw)
    @_wrap
    def cancel_processing(job_id: str) -> dict[str, Any]:
        """Request cancellation of a running job. Cancellation happens at the next chunk boundary
        (usually within ~30 s of audio processing)."""
        return job_view(service.cancel(job_id, owner()))

    @mcp.tool(annotations=ro)
    @_wrap
    def get_analysis(project_id: str) -> dict[str, Any]:
        """Analysis summary: media info, loudness, detected problems (with suggested fixes), which models ran,
        how much of the timeline each category covers, and limitations of the result."""
        pr = service.get(project_id, owner())
        counts = {c: sum(1 for s in pr.segments if s.category == c) for c in CATEGORIES}
        return {
            "project_id": pr.id, "name": pr.name, "status": pr.status, "error": pr.error,
            "media": pr.media.model_dump(),
            "analysis": pr.analysis.model_dump() if pr.analysis else None,
            "segment_counts": counts, "removed_categories": pr.mix.removed,
            "uncertain_segments": sum(1 for s in pr.segments if s.uncertain),
            "flagged_unintelligible": sum(1 for s in pr.segments if "unintelligible" in s.flags),
            "presets": {k: v["label"] for k, v in PRESETS.items()},
        }

    @mcp.tool(annotations=ro)
    @_wrap
    def list_audio_segments(
        project_id: str,
        category: Annotated[Category | None, Field(description="Filter by category")] = None,
        min_confidence: Annotated[float, Field(ge=0, le=1)] = 0.0,
        only_uncertain: bool = False,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        """List detected segments with category, start/end seconds, confidence, uncertainty flags,
        description and (for dialogue) transcript + intelligibility. Segments can overlap."""
        segs = [s for s in service.segments(project_id, owner(), category)
                if s.confidence >= min_confidence and (s.uncertain or not only_uncertain)]
        page = segs[offset: offset + limit]
        return {"total": len(segs), "offset": offset, "segments": [s.model_dump() for s in page]}

    @mcp.tool(annotations=rw)
    @_wrap
    def update_segment_category(
        project_id: str, segment_id: str,
        category: Annotated[Category | None, Field(description="New category")] = None,
        muted: Annotated[bool | None, Field(description="Mute this segment in the cleaned mix")] = None,
        start: Annotated[float | None, Field(ge=0)] = None,
        end: Annotated[float | None, Field(ge=0)] = None,
    ) -> dict[str, Any]:
        """Correct a segment's category (e.g. BGM ↔ Music), adjust its boundaries, or mute it. Edits are
        undoable and never touch the original media; stems are re-rendered in the background."""
        return service.update_segment(project_id, segment_id, owner(), category=category, muted=muted,
                                      start=start, end=end)

    @mcp.tool(annotations=rw)
    @_wrap
    def remove_audio_category(
        project_id: str,
        categories: Annotated[list[Category] | None, Field(description="Categories to remove from the mix")] = None,
        preset: Annotated[str | None, Field(description="Or a cleanup preset: " + ", ".join(PRESETS))] = None,
        render_preview: Annotated[bool, Field(description="Also render a preview WAV of the cleaned mix")] = True,
    ) -> dict[str, Any]:
        """Select which categories are removed from the cleaned mix (non-destructive, replaces the previous
        selection) or apply a preset such as remove_music, music_and_effects, remove_gibberish. Returns the
        known limitations for this removal and, optionally, a preview render job."""
        if preset:
            pr = service.apply_preset(project_id, preset, owner())
        elif categories is not None:
            pr = service.set_removed(project_id, list(categories), owner())
        else:
            raise AudioLayerError("Provide categories or preset")
        out: dict[str, Any] = {"removed_categories": pr.mix.removed,
                               "muted_segments": [s.id for s in pr.segments if s.muted],
                               "limitations": service.limitations_for(pr, list(pr.mix.removed))}
        if render_preview and pr.analysis and pr.analysis.separated:
            out["preview_job"] = job_view(service.render_preview(project_id, owner()))
        return out

    @mcp.tool(annotations=rw)
    @_wrap
    def separate_audio_category(
        project_id: str, category: Category,
        format: Literal["wav", "flac", "mp3"] = "wav",
    ) -> dict[str, Any]:
        """Isolate one category as its own full-length, timeline-synchronised stem (silence where the
        category is absent). Returns an export job_id; poll get_processing_status for the file."""
        job = service.export(project_id, ExportOptions(items=[category], format=format, zip=False), owner())
        return job_view(job)

    @mcp.tool(annotations=rw)
    @_wrap
    def export_audio_stems(
        project_id: str,
        categories: Annotated[list[Category] | None, Field(description="Default: all five categories")] = None,
        format: Literal["wav", "flac", "mp3"] = "wav",
        include_original: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Export category stems (WAV = 48 kHz / 24-bit) plus optional original audio and metadata
        (JSON, CSV, SRT, Audacity labels) as files and a ZIP package for external editors."""
        items: list[Any] = list(categories or CATEGORIES)
        if include_original:
            items.append("original")
        if include_metadata:
            items.append("metadata")
        return job_view(service.export(project_id, ExportOptions(items=items, format=format), owner()))

    @mcp.tool(annotations=rw)
    @_wrap
    def export_cleaned_audio(
        project_id: str,
        remove: Annotated[list[Category] | None, Field(description="Categories to remove; default: the "
                                                         "project's current selection")] = None,
        format: Literal["wav", "flac", "mp3"] = "wav",
        loudness_preset: Literal["none", "youtube", "tiktok", "podcast", "broadcast"] = "none",
        include_video: Annotated[bool, Field(description="Also write the original video with the cleaned "
                                                         "audio (video stream copied, not re-encoded)")] = False,
    ) -> dict[str, Any]:
        """Export the reconstructed mix without the removed categories (and muted segments), optionally
        loudness-normalised for a platform and muxed back into the video."""
        items: list[Any] = ["mix", "metadata"] + (["video"] if include_video else [])
        opts = ExportOptions(items=items, format=format, remove=remove, loudness_preset=loudness_preset)
        return job_view(service.export(project_id, opts, owner()))

    @mcp.tool(annotations=ro)
    @_wrap
    def get_quality_report(project_id: str) -> dict[str, Any]:
        """Creator-focused problem report: silent/missing audio, clipping, hum, hiss, rumble, muffled
        (band-limited) audio, edge clicks, stereo phase, loudness/true-peak, gibberish dialogue — each with
        a suggested fix."""
        return service.quality_report(project_id, owner())

    @mcp.tool(annotations=ro)
    @_wrap
    def list_projects() -> dict[str, Any]:
        """List projects in the workspace (id, name, status, duration)."""
        return {"projects": [{"project_id": p.id, "name": p.name, "status": p.status,
                              "duration": p.media.audio_duration, "has_video": p.media.has_video}
                             for p in service.list_projects(owner())]}

    @mcp.tool(annotations=ro)
    @_wrap
    def get_system_info() -> dict[str, Any]:
        """Installed models, devices, licenses and limits — use it to explain what the server can do."""
        info = service.system()
        info["input_dirs"] = [] if remote else [str(p) for p in settings.input_dirs]
        return info

    if remote:
        @mcp.tool(annotations=rw)
        @_wrap
        def create_upload_url() -> dict[str, Any]:
            """Create a single-use upload URL (valid 1 hour). The user uploads the media with
            `curl -F file=@video.mp4 <upload_url>`; the response contains the project_id to analyze."""
            from ..api.app import create_upload_token

            tok = create_upload_token(owner())
            return {"upload_url": f"{settings.public_base_url}/api/uploads/{tok}", "expires_in_s": 3600,
                    "method": "POST multipart/form-data, field name 'file'"}

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AudioLayer Studio MCP server")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("AUDIOLAYER_LOG_LEVEL", "INFO"), stream=sys.stderr)
    settings = get_settings()
    if args.transport == "stdio":
        if not settings.input_dirs:
            log.warning("AUDIOLAYER_INPUT_DIRS is not set: analyze_media(file_path=...) will be refused.")
        server = build_server(AudioLayerService(settings), remote=False)
        server.run("stdio")
    else:
        # Remote mode is served by the API app (auth, uploads and downloads live there).
        import uvicorn

        os.environ.setdefault("AUDIOLAYER_ENABLE_REMOTE_MCP", "true")
        uvicorn.run("audiolayer.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
