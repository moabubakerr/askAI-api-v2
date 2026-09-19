"""
FastAPI gateway — v2, wired to the deterministic pipeline (app/core/graph_v2.py).

Session state (for follow-up questions, fixing F-030 "context lost") is
kept server-side, keyed by session_id, in a plain in-memory dict here.
Swap for Redis/Postgres before running multiple app instances — this
in-memory version is single-process only.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.core.graph_v2 import handle_message
from app.core.reading import read_message

app = FastAPI(title="SCAI Economic Data Assistant", version="2.0.0")

# session_id -> last resolved indicator/country/period, for follow-ups.
# Single-process only — see docstring above before scaling out.
_SESSION_STORE: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    conversation_context: Optional[str] = ""


class ChatResponse(BaseModel):
    answer: str
    facts_payload: dict
    chart: Optional[dict] = None
    verified: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


class ReadResponse(BaseModel):
    """Same readings as /chat, split by who wrote each piece of text.

    council_analysis is SCAI's, verbatim. narration is generated. They are
    separate fields rather than one rendered string precisely so the frontend
    cannot blur them and a reader can always tell which is which.
    """
    ok: bool
    message: Optional[str] = None       # only when ok is false
    headline: Optional[dict] = None     # single-reading answers only
    one_liner: Optional[str] = None
    council_analysis: list = []
    evidence: list = []
    narration: Optional[str] = None
    disclaimer: Optional[str] = None
    chart: Optional[dict] = None
    facts_payload: dict = {}
    verified: bool = True


@app.post("/read", response_model=ReadResponse)
def read_endpoint(req: ChatRequest):
    """"Read this for me" — the readings formatted for a person, rather than as
    one paragraph of prose. Takes the same request as /chat."""
    prior_state = _SESSION_STORE.get(req.session_id, {})

    result = read_message(
        user_message=req.message,
        conversation_context=req.conversation_context or "",
        session_state=prior_state,
    )

    _SESSION_STORE[req.session_id] = result["session_state"]
    return ReadResponse(**{k: v for k, v in result.items() if k != "session_state"})


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    prior_state = _SESSION_STORE.get(req.session_id, {})

    result = handle_message(
        user_message=req.message,
        conversation_context=req.conversation_context or "",
        session_state=prior_state,
    )

    _SESSION_STORE[req.session_id] = result["session_state"]

    payload = result["facts_payload"]
    return ChatResponse(
        answer=result["answer"],
        facts_payload=payload,
        chart=payload.get("chart"),
        verified="_verifier_rejected_numbers" not in payload,
    )
