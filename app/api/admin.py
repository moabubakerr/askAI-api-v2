"""
/admin — read-only views over the message log, the feedback and the catalogue.

Behind a key, and closed by default. These endpoints return whatever users
typed into the chat, which is personal data under Qatar's PDPL; an unset key is
a missing decision rather than a decision to publish, so the routes refuse to
serve until one is configured. Set ADMIN_API_KEY and send it as X-Admin-Key.

This is a shared secret, not identity: it says the caller is the dashboard, not
WHO is using the dashboard. Adequate for a service-to-service call from an
admin UI that does its own sign-in; not adequate as the only thing between a
person and the logs. If individual accountability is needed — who read what —
this needs real auth in front of it.

Every route is a SELECT. The app connects as scai_ro, which holds INSERT on two
application tables and nothing else, so nothing here can change SCAI's data.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.core.config import settings
from app.db import admin_queries

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(x_admin_key: Optional[str] = Header(default=None)) -> None:
    """Refuses unless a key is configured AND matches.

    503 when unconfigured, not 401: nothing the caller sends can fix it, and
    saying "unauthorized" would send them looking for a credential that does
    not exist yet.
    """
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin API is disabled. Set ADMIN_API_KEY to enable it.")
    if not x_admin_key or x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key.")


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
