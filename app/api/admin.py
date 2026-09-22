"""
/admin — read-only views over the message log, the feedback and the catalogue.

Closed by default. These endpoints return whatever users typed into the chat,
which is personal data under Qatar's PDPL, so they refuse to serve until a
credential is configured.

Two ways in, and the difference matters:

  POST /admin/login  ->  Authorization: Bearer <token>   a person, signed in
  X-Admin-Key: ...                                       a service, shared key

The key is a shared secret and says nothing about who is holding it. A login
session at least names an account, and today there is exactly one. Neither is
individual accountability: if SCAI needs to know WHICH person read the logs,
this needs real identity in front of it.

See app/api/admin_auth.py for the session model and its limits.

Every route is a SELECT. The app connects as scai_ro, which holds INSERT on two
application tables and nothing else, so nothing here can change SCAI's data.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from app.api.admin_auth import authenticate, require_admin, sessions
from app.db import admin_queries

router = APIRouter(prefix="/admin", tags=["admin"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    ok: bool
    token: Optional[str] = None
    expires_at: Optional[float] = None
    username: Optional[str] = None


@router.post("/login", response_model=LoginResponse, response_model_exclude_none=True)
def login(req: LoginRequest):
    """Exchanges a username and password for a session token.

    Send it back as `Authorization: Bearer <token>` on every other /admin call.
    The same 401 and the same delay for a wrong username as for a wrong
    password, so neither can be probed independently.
    """
    session = authenticate(req.username, req.password)
    if not session:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return LoginResponse(ok=True, **session)


@router.post("/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    """Ends this session.

    Idempotent on purpose: logging out twice, or with a token that has already
    expired, is a success. The caller wanted to be signed out and is.
    """
    token = authorization[7:].strip() if (authorization or "").lower().startswith("bearer ") else ""
    return {"ok": True, "ended": sessions.revoke(token) if token else False}


@router.get("/me", dependencies=[Depends(require_admin)])
def whoami(caller: dict = Depends(require_admin)):
    """Who the current credential belongs to. Lets a dashboard check on load
    whether its stored token is still good, without fetching data to find out."""
    return caller


@router.get("/messages", dependencies=[Depends(require_admin)])
def list_messages(
    from_: Optional[str] = Query(None, alias="from",
                                  description="ISO timestamp, inclusive"),
    to: Optional[str] = Query(None, description="ISO timestamp, exclusive"),
    language: Optional[str] = Query(None, pattern="^(en|ar)$"),
    answered: Optional[bool] = None,
    shape: Optional[str] = Query(None, description="answer_shape, e.g. value, series, refusal"),
    q: Optional[str] = Query(None, description="substring of the question or the answer"),
    limit: Optional[int] = None,
    offset: Optional[int] = None,
):
    """Every logged exchange, newest first, with its rating when it has one."""
    return admin_queries.messages(from_ts=from_, to_ts=to, language=language,
                                   answered=answered, shape=shape, q=q,
                                   limit=limit, offset=offset)


@router.get("/messages/{message_id}/provenance", dependencies=[Depends(require_admin)])
def message_provenance(message_id: str):
    """One answer and the exact rows it was built from.

    The chain the panel exists for: question → answer → cited record → that
    data point's own numbers → the indicator → the export file it was loaded
    from. Nothing here is reconstructed after the fact; the citations were
    computed when the answer was, in app/compute/citations.py, which is also
    what rendered the "Sources:" footer the user saw.

    An empty citation list is a real result, not an error: greetings and
    refusals cite nothing. A 404 means the message was never logged, which the
    best-effort logger permits.
    """
    result = admin_queries.message_provenance(message_id)
    if not result:
        raise HTTPException(status_code=404, detail="No logged message with that id.")
    return result


@router.get("/lineage", dependencies=[Depends(require_admin)])
def data_lineage():
    """Which export file each table was loaded from, and when.

    Recorded by etl/load_data.py from the files it actually opened, so it
    describes the load that produced the data currently in the database.

    Read `rows_in_file` against `rows_in_table`: they differ by design in
    several places, and a gap appearing where there was not one before is the
    earliest sign an export changed shape.

    `not_loaded_by_etl` lists the tables no export feeds — the application's
    own, and the embeddings and article chunks the index scripts write. Their
    absence is the point: nothing upstream will ever refresh them.
    """
    return admin_queries.lineage()


@router.get("/feedback", dependencies=[Depends(require_admin)])
def list_feedback(
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    max_rating: Optional[int] = Query(None, ge=1, le=5),
    has_comment: Optional[bool] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
):
    """Ratings, joined to the message each is about.

    min_rating=1&max_rating=2 is the one worth watching: every low score
    carries a required comment saying what was wrong.
    """
    return admin_queries.feedback(min_rating=min_rating, max_rating=max_rating,
                                   has_comment=has_comment,
                                   limit=limit, offset=offset)


@router.get("/sessions", dependencies=[Depends(require_admin)])
def list_sessions(limit: Optional[int] = None, offset: Optional[int] = None):
    """One row per conversation, newest activity first."""
    return admin_queries.sessions(limit=limit, offset=offset)


@router.get("/stats", dependencies=[Depends(require_admin)])
def usage_stats():
    """Volume, refusal rate and ratings — overall, by language, by answer shape."""
    return admin_queries.stats()


@router.get("/catalogue", dependencies=[Depends(require_admin)])
def list_catalogue(
    q: Optional[str] = Query(None, description="substring of the indicator name"),
    limit: Optional[int] = None,
    offset: Optional[int] = None,
):
    """The published indicators the assistant can answer from, with coverage.

    data_points, first_period and last_period are the useful columns here: they
    are what tells you an indicator exists but has nothing recent, which is the
    commonest reason a reasonable question gets refused.
    """
    return admin_queries.catalogue(q=q, limit=limit, offset=offset)


@router.get("/catalogue/{indicator_id}", dependencies=[Depends(require_admin)])
def catalogue_item(indicator_id: str):
    """One indicator and its full published series."""
    item = admin_queries.catalogue_item(indicator_id)
    if not item:
        raise HTTPException(status_code=404, detail="No published indicator with that id.")
    return item
