"""FastAPI application: REST API for the web UI, file serving, and the remote MCP endpoint."""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..core import pipeline
from ..core.errors import AudioLayerError, InvalidMedia, LimitExceeded, NotFound
from ..core.paths import SUPPORTED_EXTENSIONS, safe_filename
from ..core.schemas import AnalyzeOptions, Category, ExportOptions
from ..core.service import AudioLayerService
from .auth import Unauthorized, current_principal, principal_for_token, require_principal, token_from_headers

log = logging.getLogger("audiolayer.api")

# ---- single-use upload tokens (remote MCP workflow) ------------------------------------
_UPLOADS: dict[str, tuple[str, float]] = {}
_UPLOAD_LOCK = threading.Lock()


def create_upload_token(owner: str, ttl: float = 3600) -> str:
    tok = secrets.token_urlsafe(24)
    with _UPLOAD_LOCK:
        now = time.time()
        for k in [k for k, (_, exp) in _UPLOADS.items() if exp < now]:
            _UPLOADS.pop(k, None)
        _UPLOADS[tok] = (owner, now + ttl)
    return tok


def _consume_upload_token(tok: str) -> str:
    with _UPLOAD_LOCK:
        item = _UPLOADS.pop(tok, None)
    if not item or item[1] < time.time():
        raise NotFound("Upload link is invalid or expired")
    return item[0]


class SegmentPatch(BaseModel):
    category: Category | None = None
    muted: bool | None = None
    solo: bool | None = None
    start: float | None = None
    end: float | None = None


class MixPatch(BaseModel):
    removed: list[Category] | None = None
    preset: str | None = None


class PreviewRequest(BaseModel):
    remove: list[Category] | None = None


Owner = Annotated[str, Depends(require_principal)]


def create_app(settings: Settings | None = None, service: AudioLayerService | None = None) -> FastAPI:
    settings = settings or get_settings()
    svc = service or AudioLayerService(settings)
    enable_mcp = os.environ.get("AUDIOLAYER_ENABLE_REMOTE_MCP", "false").lower() in ("1", "true", "yes")
    mcp_app = None
    mcp_server = None
    if enable_mcp:
        from urllib.parse import urlparse

        from mcp.server.transport_security import TransportSecuritySettings

        from ..mcp_server.server import build_server

        hosts = {"127.0.0.1", "localhost", "127.0.0.1:*", "localhost:*", *settings.allowed_hosts}
        origins = {"http://127.0.0.1:*", "http://localhost:*"}
        if settings.public_base_url:
            u = urlparse(settings.public_base_url)
            hosts.add(u.netloc)
            origins.add(f"{u.scheme}://{u.netloc}")
        security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=sorted(hosts),
                                             allowed_origins=sorted(origins))
        mcp_server = build_server(svc, remote=True)
        mcp_app = mcp_server.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True,
                                                 transport_security=security)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if mcp_app is not None:
            async with mcp_app.router.lifespan_context(mcp_app):
                yield
        else:
            yield
        svc.jobs.shutdown()

    app = FastAPI(title="AudioLayer Studio API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = svc
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                           allow_headers=["*"], allow_credentials=False)

    @app.exception_handler(AudioLayerError)
    async def _err(_: Request, e: AudioLayerError) -> JSONResponse:
        return JSONResponse(status_code=e.status, content={"error": e.to_dict()})

    # ---- system ------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/system")
    def system(owner: Owner) -> dict[str, Any]:
        return svc.system()

    # ---- projects ----------------------------------------------------------------------
    def _save_upload(file: UploadFile) -> Path:
        name = safe_filename(file.filename or "upload")
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidMedia(f"Unsupported file type {Path(name).suffix!r}",
                               hint="Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        limit = settings.max_upload_mb * 1024 * 1024
        fd, tmp = tempfile.mkstemp(dir=settings.tmp_dir, suffix=Path(name).suffix.lower())
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise LimitExceeded(f"Upload exceeds {settings.max_upload_mb} MB")
                    out.write(chunk)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return Path(tmp)

    def _ingest(file: UploadFile, owner: str, auto_analyze: bool) -> dict[str, Any]:
        tmp = _save_upload(file)
        try:
            pr = svc.create_from_file(tmp, safe_filename(file.filename or "upload"), owner, move=True)
        finally:
            tmp.unlink(missing_ok=True)
        out: dict[str, Any] = {"project": pr.model_dump()}
        if auto_analyze:
            out["job"] = svc.analyze(pr.id, AnalyzeOptions(), owner).model_dump()
        return out

    @app.post("/api/projects")
    def upload(owner: Owner, file: UploadFile = File(...), analyze: bool = Form(False)) -> dict[str, Any]:
        return _ingest(file, owner, analyze)

    @app.post("/api/uploads/{upload_token}")
    def upload_with_token(upload_token: str, file: UploadFile = File(...)) -> dict[str, Any]:
        owner = _consume_upload_token(upload_token)
        res = _ingest(file, owner, False)
        return {"project_id": res["project"]["id"], "media": res["project"]["media"],
                "next": "Ask Claude to run analyze_media with this project_id."}

    @app.get("/api/projects")
    def list_projects(owner: Owner) -> list[dict[str, Any]]:
        return [{"id": p.id, "name": p.name, "status": p.status, "created_at": p.created_at,
                 "duration": p.media.audio_duration, "has_video": p.media.has_video} for p in svc.list_projects(owner)]

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str, owner: Owner) -> dict[str, Any]:
        pr = svc.get(project_id, owner)
        d = pr.model_dump(exclude={"history", "redo"})
        d["can_undo"], d["can_redo"] = bool(pr.history), bool(pr.redo)
        d["active_jobs"] = [j.model_dump() for j in svc.jobs.active_for(project_id)]
        d["stems_available"] = pipeline.stream_paths(svc.store, project_id) is not None
        d["limitations_for_selection"] = svc.limitations_for(pr, list(pr.mix.removed)) if pr.analysis else []
        return d

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str, owner: Owner) -> dict[str, Any]:
        svc.delete(project_id, owner)
        return {"deleted": project_id}

    @app.post("/api/projects/{project_id}/analyze")
    def analyze(project_id: str, owner: Owner, opts: AnalyzeOptions = Body(default_factory=lambda: AnalyzeOptions())) -> dict[str, Any]:
        return svc.analyze(project_id, opts, owner).model_dump()

    @app.get("/api/projects/{project_id}/peaks")
    def peaks(project_id: str, owner: Owner) -> dict[str, Any]:
        svc.get(project_id, owner)
        return pipeline.load_peaks(svc.store, project_id)

    @app.get("/api/projects/{project_id}/quality")
    def quality(project_id: str, owner: Owner) -> dict[str, Any]:
        return svc.quality_report(project_id, owner)

    # ---- segments & mix ------------------------------------------------------------------
    @app.get("/api/projects/{project_id}/segments")
    def segments(project_id: str, owner: Owner, category: str | None = None) -> list[dict[str, Any]]:
        return [s.model_dump() for s in svc.segments(project_id, owner, category)]

    @app.patch("/api/projects/{project_id}/segments/{segment_id}")
    def patch_segment(project_id: str, segment_id: str, body: SegmentPatch, owner: Owner) -> dict[str, Any]:
        return svc.update_segment(project_id, segment_id, owner, **body.model_dump())

    @app.post("/api/projects/{project_id}/mix")
    def set_mix(project_id: str, body: MixPatch, owner: Owner) -> dict[str, Any]:
        if body.preset:
            pr = svc.apply_preset(project_id, body.preset, owner)
        else:
            pr = svc.set_removed(project_id, list(body.removed or []), owner)
        return {"removed": pr.mix.removed, "segments": [s.model_dump() for s in pr.segments],
                "limitations": svc.limitations_for(pr, list(pr.mix.removed))}

    @app.post("/api/projects/{project_id}/undo")
    def undo(project_id: str, owner: Owner) -> dict[str, Any]:
        return svc.undo(project_id, owner).model_dump(exclude={"history", "redo"})

    @app.post("/api/projects/{project_id}/redo")
    def redo(project_id: str, owner: Owner) -> dict[str, Any]:
        return svc.undo(project_id, owner, redo=True).model_dump(exclude={"history", "redo"})

    @app.post("/api/projects/{project_id}/reset")
    def reset(project_id: str, owner: Owner) -> dict[str, Any]:
        return svc.reset_edits(project_id, owner).model_dump(exclude={"history", "redo"})

    @app.post("/api/projects/{project_id}/preview")
    def preview(project_id: str, owner: Owner, body: PreviewRequest = Body(default_factory=lambda: PreviewRequest())) -> dict[str, Any]:
        return svc.render_preview(project_id, owner, body.remove).model_dump()

    @app.post("/api/projects/{project_id}/export")
    def export(project_id: str, opts: ExportOptions, owner: Owner) -> dict[str, Any]:
        return svc.export(project_id, opts, owner).model_dump()

    # ---- jobs ------------------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, owner: Owner) -> dict[str, Any]:
        return svc.job(job_id, owner).model_dump()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str, owner: Owner) -> dict[str, Any]:
        return svc.cancel(job_id, owner).model_dump()

    # ---- files (Range requests supported for audio/video seeking) --------------------------
    @app.get("/api/projects/{project_id}/files/{rel:path}")
    def files(project_id: str, rel: str, owner: Owner, download: bool = Query(False)) -> FileResponse:
        p = svc.resolve_file(project_id, rel, owner)
        return FileResponse(p, filename=p.name if download else None,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/projects/{project_id}/media")
    def media(project_id: str, owner: Owner) -> FileResponse:
        pr = svc.get(project_id, owner)
        return FileResponse(svc.store.path(project_id, pr.original_file))

    # ---- remote MCP --------------------------------------------------------------------------
    if mcp_app is not None:
        async def mcp_asgi(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                await mcp_app(scope, receive, send)
                return
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            who = principal_for_token(settings, token_from_headers(headers.get("authorization")))
            if who is None:
                resp = JSONResponse({"error": Unauthorized("Missing or invalid bearer token").to_dict()},
                                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await resp(scope, receive, send)
                return
            tok = current_principal.set(who)
            try:
                await mcp_app(scope, receive, send)
            finally:
                current_principal.reset(tok)

        app.mount("/mcp", mcp_asgi)

    # ---- built frontend ---------------------------------------------------------------------
    dist = Path(os.environ.get("AUDIOLAYER_FRONTEND_DIST", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
    if dist.is_dir() and (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="AudioLayer Studio API server")
    ap.add_argument("--host", default=os.environ.get("AUDIOLAYER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("AUDIOLAYER_PORT", "8000")))
    ap.add_argument("--remote-mcp", action="store_true", help="Also serve MCP (Streamable HTTP) at /mcp")
    a = ap.parse_args()
    if a.remote_mcp:
        os.environ["AUDIOLAYER_ENABLE_REMOTE_MCP"] = "true"
    logging.basicConfig(level=os.environ.get("AUDIOLAYER_LOG_LEVEL", "INFO"))
    uvicorn.run("audiolayer.api.app:create_app", factory=True, host=a.host, port=a.port)

