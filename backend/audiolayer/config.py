"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]


def _tokens(raw: str | None) -> dict[str, str]:
    """Parse ``name:token,name2:token2`` into ``{token: name}``."""
    out: dict[str, str] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, token = item.partition(":")
        if not token:
            name, token = "default", name
        out[token.strip()] = name.strip()
    return out


@dataclass(frozen=True)
class Settings:
    workspace: Path
    model_dir: Path
    device: str = "auto"
    max_upload_mb: int = 2048
    max_duration_s: float = 1800.0
    classifier: str = "auto"  # panns | heuristic | auto
    separator: str = "auto"  # demucs | none | auto
    demucs_model: str = "htdemucs"
    asr: str = "auto"  # faster-whisper | none | auto
    whisper_model: str = "base"
    allow_model_download: bool = True
    input_dirs: list[Path] = field(default_factory=list)
    auth_tokens: dict[str, str] = field(default_factory=dict)
    workers: int = 1
    chunk_seconds: float = 30.0
    public_base_url: str = ""
    allowed_hosts: list[str] = field(default_factory=list)
    cors_origins: list[str] = field(default_factory=list)

    @property
    def projects_dir(self) -> Path:
        return self.workspace / "projects"

    @property
    def jobs_dir(self) -> Path:
        return self.workspace / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.workspace / "tmp"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_tokens)

    def ensure_dirs(self) -> None:
        for p in (self.workspace, self.projects_dir, self.jobs_dir, self.tmp_dir, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    # One shared default so the web app, the CLI and Claude (MCP) all see the same projects.
    ws = Path(os.environ.get("AUDIOLAYER_WORKSPACE", str(Path.home() / "AudioLayer" / "workspace"))).expanduser().resolve()
    md = Path(
        os.environ.get("AUDIOLAYER_MODEL_DIR", str(Path.home() / ".cache" / "audiolayer"))
    ).expanduser().resolve()
    cors = os.environ.get("AUDIOLAYER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return Settings(
        workspace=ws,
        model_dir=md,
        device=os.environ.get("AUDIOLAYER_DEVICE", "auto"),
        max_upload_mb=int(os.environ.get("AUDIOLAYER_MAX_UPLOAD_MB", "2048")),
        max_duration_s=float(os.environ.get("AUDIOLAYER_MAX_DURATION_S", "1800")),
        classifier=os.environ.get("AUDIOLAYER_CLASSIFIER", "auto"),
        separator=os.environ.get("AUDIOLAYER_SEPARATOR", "auto"),
        demucs_model=os.environ.get("AUDIOLAYER_DEMUCS_MODEL", "htdemucs"),
        asr=os.environ.get("AUDIOLAYER_ASR", "auto"),
        whisper_model=os.environ.get("AUDIOLAYER_WHISPER_MODEL", "base"),
        allow_model_download=_bool("AUDIOLAYER_ALLOW_MODEL_DOWNLOAD", True),
        input_dirs=_paths(os.environ.get("AUDIOLAYER_INPUT_DIRS")),
        auth_tokens=_tokens(os.environ.get("AUDIOLAYER_AUTH_TOKENS")),
        workers=max(1, int(os.environ.get("AUDIOLAYER_WORKERS", "1"))),
        chunk_seconds=float(os.environ.get("AUDIOLAYER_CHUNK_SECONDS", "30")),
        public_base_url=os.environ.get("AUDIOLAYER_PUBLIC_BASE_URL", "").rstrip("/"),
        allowed_hosts=[h.strip() for h in os.environ.get("AUDIOLAYER_ALLOWED_HOSTS", "").split(",") if h.strip()],
        cors_origins=[c.strip() for c in cors.split(",") if c.strip()],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = load_settings()
    s.ensure_dirs()
    return s


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: S110 # pragma: no cover - torch missing or broken -> CPU
        pass
    return "cpu"
