"""
Server-side conversation history, per session.

Before this, each request was answered in isolation: the only thing carried
between turns was the last indicator name, and the transcript itself had to be
supplied by the caller on every request as `conversation_context`. A frontend
that didn't send it — or sent it in a shape the prompt didn't expect — got a
system with no memory at all, which is QC finding F-030 ("What does
non-hydrocarbon GDP mean?" / "WHAT IS THE LATEST VALUE?" / "WHAT about the last
year?", where each follow-up lost the thread).

Three deliberate choices:

Bounded, not complete. max_model_len on the serving side is 16384 tokens for
prompt AND completion together, shared with the indicator candidate list and
the composer's facts. History is therefore capped by turns and by characters,
and the cap is enforced here rather than hoped for — an unbounded transcript
would start failing requests only once a conversation got long, which is the
worst time to discover it.

Evicted, not kept forever. A process-lifetime dict keyed by session id is an
unbounded memory leak in a long-running service: every distinct session id ever
seen stays resident. Sessions expire by idle time and the store is capped by
count, oldest-touched first.

Still in-memory, and still single-process. Restarting the API loses every
conversation, and running two replicas gives each its own half of the history.
That is a real limit, not a hidden one — moving this to Redis or Postgres is a
drop-in replacement of this class, and nothing above it needs to change.
"""
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Optional

from app.resolvers.period_resolver import parse_period_expression

# Roughly six exchanges of context. Enough for "and the year before that?"
# chains, short enough to leave the prompt budget to the actual question.
MAX_TURNS = 12
MAX_CONTEXT_CHARS = 2000
SESSION_IDLE_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 1000

# An answer can run to several paragraphs plus a Sources footer. Stored turns
# keep only the opening of each, since history exists to resolve references
# like "that" and "the year before", not to re-read past answers in full.
MAX_STORED_TEXT = 400


class ConversationStore:
    def __init__(self, max_turns: int = MAX_TURNS, idle_seconds: int = SESSION_IDLE_SECONDS,
                 max_sessions: int = MAX_SESSIONS):
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._max_turns = max_turns
        self._idle_seconds = idle_seconds
        self._max_sessions = max_sessions
        # FastAPI serves sync endpoints from a threadpool, so two requests for
        # the same session really can land concurrently.
        self._lock = threading.Lock()

    def _evict(self) -> None:
        cutoff = time.time() - self._idle_seconds
        for sid in [s for s, v in self._sessions.items() if v["touched"] < cutoff]:
            del self._sessions[sid]
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def _session(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            session = {"turns": deque(maxlen=self._max_turns), "state": {}, "touched": time.time()}
            self._sessions[session_id] = session
        session["touched"] = time.time()
        self._sessions.move_to_end(session_id)
        return session

    def get_state(self, session_id: str) -> dict:
        with self._lock:
            self._evict()
            return dict(self._session(session_id)["state"])

    def record(self, session_id: str, user_message: str, answer: str, state: dict) -> None:
        with self._lock:
            self._evict()
            session = self._session(session_id)
            session["turns"].append({"user": (user_message or "").strip()[:MAX_STORED_TEXT],
                                      "assistant": (answer or "").strip()[:MAX_STORED_TEXT]})
            session["state"] = dict(state or {})

    def render_context(self, session_id: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """A plain transcript, newest turns prioritised.

        Trimming takes whole turns from the FRONT, so the most recent exchange
        is never the one cut in half — a follow-up refers to what just happened,
        and half a turn is worse than one fewer turn.
        """
        with self._lock:
            self._evict()
            session = self._sessions.get(session_id)
            turns = list(session["turns"]) if session else []

        if not turns:
            return ""

        rendered, total = [], 0
        for turn in reversed(turns):
            block = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            if total + len(block) > max_chars and rendered:
                break
            rendered.append(block)
            total += len(block)
        return "\n\n".join(reversed(rendered))

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> dict:
        with self._lock:
            self._evict()
            return {
                "sessions": len(self._sessions),
                "turns": sum(len(s["turns"]) for s in self._sessions.values()),
                "max_turns_per_session": self._max_turns,
                "idle_expiry_seconds": self._idle_seconds,
            }


conversations = ConversationStore()


# "the latest", "currently", "right now" — a period is being specified, just not
# by name. Without this the previous turn's period is inherited and "what is the
# latest value?" after a question about 2024 answers with 2024's last reading,
# which is the opposite of what was asked. Exactly the F-030 sequence
# ("WHAT IS THE LATEST VALUE?" / "WHAT about the last year?").
_ASKS_FOR_LATEST = re.compile(
    r"\b(latest|most recent|current(ly)?|right now|as of today|up to date|newest)\b"
    r"|أحدث|آخر|الحالي",
    re.IGNORECASE,
)


def carry_forward(intent: dict, state: dict, user_message: str = "") -> dict:
    """Fills slots the follow-up left unsaid from the previous turn.

    "How many tourists arrived in May 2025?" then "and for Saudi Arabia?" —
    the second names a country but no indicator and no period, and means both
    of the previous ones. Equally "and the year before that?" names a period
    and means the previous indicator and countries.

    So each slot is inherited only when the new message does not fill it, and
    only on a follow-up. Anything the user did state always wins: inheriting
    over an explicit value would silently answer a different question from the
    one asked, which is the failure this whole pipeline exists to avoid.
    """
    if not intent.get("is_followup") or not state:
        return intent

    carried = dict(intent)
    if not carried.get("indicator_phrase") and state.get("last_indicator_name"):
        carried["indicator_phrase"] = state["last_indicator_name"]
    # last_metrics (a multi-metric answer) is deliberately NOT restored here.
    # It is handled in handle_message, which rejoins the list so the normal
    # split path re-runs the snapshot; flattening it into indicator_phrase at
    # this point would make it look like one indicator name.
    if not carried.get("countries_mentioned") and not carried.get("country_group_mentioned"):
        if state.get("last_countries"):
            carried["countries_mentioned"] = list(state["last_countries"])
        if state.get("last_country_group"):
            carried["country_group_mentioned"] = state["last_country_group"]
    asks_latest = bool(_ASKS_FOR_LATEST.search(user_message or ""))
    # A follow-up that states its own period must not inherit the old one. The
    # model often leaves period_expression empty even when the message says
    # "what about in 2024", and inheriting then pinned the answer to the
    # previous turn's year: the metrics changed correctly and the period never
    # did, so three indicators came back for 2023 again.
    states_own_period = parse_period_expression(user_message or "").kind != "unspecified"
    if (not carried.get("period_expression") and state.get("last_period_expression")
            and not asks_latest and not states_own_period):
        carried["period_expression"] = state["last_period_expression"]
    if not carried.get("explicit_frequency") and state.get("last_explicit_frequency"):
        carried["explicit_frequency"] = state["last_explicit_frequency"]
    return carried
