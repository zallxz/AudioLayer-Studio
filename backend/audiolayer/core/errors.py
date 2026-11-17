"""Domain errors with stable machine-readable codes (used by REST and MCP layers)."""

from __future__ import annotations


class AudioLayerError(Exception):
    code = "error"
    status = 400

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class InvalidMedia(AudioLayerError):
    code = "invalid_media"
    status = 415


class LimitExceeded(AudioLayerError):
    code = "limit_exceeded"
    status = 413


class NotFound(AudioLayerError):
    code = "not_found"
    status = 404


class Forbidden(AudioLayerError):
    code = "forbidden"
    status = 403


class Conflict(AudioLayerError):
    code = "conflict"
    status = 409


class ModelUnavailable(AudioLayerError):
    code = "model_unavailable"
    status = 503


class Cancelled(AudioLayerError):
    code = "cancelled"
    status = 409


class ProcessingFailed(AudioLayerError):
    code = "processing_failed"
    status = 500
