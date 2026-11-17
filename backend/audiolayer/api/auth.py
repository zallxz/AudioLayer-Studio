"""Bearer-token authentication shared by the REST API and the remote MCP endpoint.

* No tokens configured (default, local use): every request is the ``local`` principal.
* ``AUDIOLAYER_AUTH_TOKENS="alice:tok1,bob:tok2"``: requests must send
  ``Authorization: Bearer <token>``; each token maps to a principal that owns its projects.
  Media URLs used by <audio>/<video> elements may pass ``?token=`` instead of the header.
"""

from __future__ import annotations

import contextvars
import hmac

from fastapi import Request

from ..config import Settings
from ..core.errors import AudioLayerError

current_principal: contextvars.ContextVar[str] = contextvars.ContextVar("current_principal", default="local")


class Unauthorized(AudioLayerError):
    code = "unauthorized"
    status = 401


def principal_for_token(settings: Settings, token: str | None) -> str | None:
    if not settings.auth_enabled:
        return "local"
    if not token:
        return None
    for known, name in settings.auth_tokens.items():
        if hmac.compare_digest(known.encode(), token.encode()):
            return name
    return None


def token_from_headers(auth_header: str | None, query_token: str | None = None) -> str | None:
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return query_token


def require_principal(request: Request) -> str:
    settings: Settings = request.app.state.settings
    tok = token_from_headers(request.headers.get("authorization"), request.query_params.get("token"))
    who = principal_for_token(settings, tok)
    if who is None:
        raise Unauthorized("Missing or invalid bearer token")
    return who
