"""Validation for user-supplied local file paths (used by the local MCP server).

Only files inside explicitly configured directories (``AUDIOLAYER_INPUT_DIRS`` and the
workspace) may be read. Symlinks are resolved before the check, so a link pointing outside
the allowlist is rejected.
"""

from __future__ import annotations

from pathlib import Path

from .errors import Forbidden, InvalidMedia, NotFound

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".wav", ".mp3", ".aac", ".m4a", ".flac", ".webm"}


def validate_input_path(raw: str, allowed_roots: list[Path]) -> Path:
    if not raw or "\x00" in raw:
        raise Forbidden("Empty or invalid path")
    if not allowed_roots:
        raise Forbidden(
            "Local file access is disabled.",
            hint="Set AUDIOLAYER_INPUT_DIRS to one or more folders the server may read from.",
        )
    p = Path(raw).expanduser()
    if not p.is_absolute():
        # Relative paths are resolved against the first allowed root.
        p = allowed_roots[0] / p
    p = p.resolve()
    if not any(p == r or r in p.parents for r in allowed_roots):
        raise Forbidden(
            f"Path is outside the allowed input folders: {raw}",
            hint="Allowed folders: " + ", ".join(str(r) for r in allowed_roots),
        )
    if not p.exists() or not p.is_file():
        raise NotFound(f"File not found: {raw}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidMedia(
            f"Unsupported file type {p.suffix!r}",
            hint="Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
    return p


def safe_filename(name: str) -> str:
    base = Path(name).name
    keep = "".join(c if (c.isalnum() or c in "._- ") else "_" for c in base).strip(" .")
    return keep[:120] or "upload"
