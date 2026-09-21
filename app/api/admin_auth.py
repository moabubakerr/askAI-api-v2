"""
Login, logout and the guard every /admin route sits behind.

ONE account, from configuration. The default password is ADMIN123, which is a
placeholder for getting the dashboard built and is not a credential: it is
public in this repository, it is the first thing any scanner tries, and these
routes return every question users have typed. Before anyone outside the team
can reach this service, ADMIN_PASSWORD must be set to something else in .env.
Nothing here enforces that, because a service that refuses to start is worse
than one that warns — but the warning is logged on every startup.

Two ways in, deliberately:

  X-Admin-Key     a shared secret for service-to-service calls, unchanged
  Bearer <token>  a session from POST /admin/login, for a person at a browser

Tokens live in memory, like the conversation store. A restart signs everyone
out, and a second replica would not recognise the first's tokens. Both are
acceptable for a single-process on-prem deployment and both stop being
acceptable the moment it is scaled, which is the point at which this should
move to a signed token or Redis.
"""
import logging
import secrets
import threading
import time
from typing import Optional

from fastapi import Header, HTTPException

from app.core.config import settings

log = logging.getLogger(__name__)

# The placeholder that must not reach production.
_PLACEHOLDER_PASSWORD = "ADMIN123"

# Slows a password-guessing loop from thousands per second to a handful,
# without locking a legitimate admin out of their own dashboard after a typo.
_FAILED_ATTEMPT_DELAY_SECONDS = 1.0


class _Sessions:
    """Opaque tokens with an expiry, held in memory."""

    def __init__(self) -> None:
        self._tokens: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = time.time()
        for token in [t for t, v in self._tokens.items() if v["expires_at"] <= now]:
            del self._tokens[token]

    def create(self, username: str) -> dict:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + settings.ADMIN_SESSION_HOURS * 3600
        with self._lock:
            self._purge()
            self._tokens[token] = {"username": username, "expires_at": expires_at}
        return {"token": token, "expires_at": expires_at, "username": username}

    def get(self, token: str) -> Optional[dict]:
        with self._lock:
            self._purge()
            session = self._tokens.get(token)
            return dict(session) if session else None

    def revoke(self, token: str) -> bool:
        with self._lock:
            return self._tokens.pop(token, None) is not None

    def count(self) -> int:
        with self._lock:
            self._purge()
            return len(self._tokens)


sessions = _Sessions()


def warn_if_default_password() -> None:
    """Says so, loudly, on every startup. Called from the app factory."""
    if settings.ADMIN_PASSWORD == _PLACEHOLDER_PASSWORD:
        log.warning(
            "ADMIN_PASSWORD is still the placeholder %r. Anyone who can reach "
            "this service can read every question users have asked. Set "
            "ADMIN_PASSWORD in .env before this is exposed.", _PLACEHOLDER_PASSWORD)


def authenticate(username: str, password: str) -> Optional[dict]:
    """Checks credentials and starts a session, or returns None.

    compare_digest on both fields: a plain == on a secret leaks its length and
    its matching prefix through timing. That matters little behind a VPN and
    costs nothing to do properly.
    """
    if not settings.ADMIN_PASSWORD:
        return None
    user_ok = secrets.compare_digest((username or ""), settings.ADMIN_USERNAME)
    pass_ok = secrets.compare_digest((password or ""), settings.ADMIN_PASSWORD)
    # Both compared before returning, so a wrong username and a wrong password
    # take the same time and cannot be told apart.
    if not (user_ok and pass_ok):
        time.sleep(_FAILED_ATTEMPT_DELAY_SECONDS)
        return None
    return sessions.create(settings.ADMIN_USERNAME)


def require_admin(x_admin_key: Optional[str] = Header(default=None),
                   authorization: Optional[str] = Header(default=None)) -> dict:
    """The dependency every /admin route uses.

    Accepts either credential. Returns who the caller is, so a route can say so
    and a future audit log has something to record.
    """
    if not settings.ADMIN_API_KEY and not settings.ADMIN_PASSWORD:
        # Nothing the caller sends can fix this, so 503 rather than 401 —
        # "unauthorized" would send them looking for a credential that does not
        # exist yet.
        raise HTTPException(
            status_code=503,
            detail="Admin API is disabled. Set ADMIN_API_KEY or ADMIN_PASSWORD.")

    if x_admin_key and settings.ADMIN_API_KEY and secrets.compare_digest(
            x_admin_key, settings.ADMIN_API_KEY):
        return {"via": "api_key", "username": "service"}

    if authorization and authorization.lower().startswith("bearer "):
        session = sessions.get(authorization[7:].strip())
        if session:
            return {"via": "session", "username": session["username"],
                    "expires_at": session["expires_at"]}

    raise HTTPException(
        status_code=401,
        detail="Sign in at POST /admin/login, or send a valid X-Admin-Key.")
