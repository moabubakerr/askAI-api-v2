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

# The user's own words are kept in full up to this, and the cap is separate from
# the one above on purpose. The two sides of a turn are not worth the same per
# character: the question is short, is what every reference points back at, and
# is the only part the model did not write. Truncating both at 400 spent the
# budget symmetrically on the half that needed it least.
MAX_STORED_QUESTION = 700

# How many exchanges the Composer is shown. Three, because a reference can reach
# two turns back — "inflation" / "and 2024?" / "which was higher?" — and because
# these lines share a prompt budget with the facts payload, the indicator
# candidates and twenty-eight rules.
COMPOSER_HISTORY_TURNS = 3

# After this many turns, a slot left over from an earlier question is not
# inherited by a follow-up. Nothing used to expire: last_indicator_name sat in
# the session until it timed out two hours later, so a single is_followup
# misfire could answer a fresh question with an indicator from ten turns ago.
# Three matches the depth a reference can actually reach — beyond that "that"
# does not mean the old value to the reader either.
STALE_AFTER_TURNS = 3

# Slots whose meaning decays with distance, so they are stamped and checked. A
# slot NOT listed here keeps the old behaviour of living until the session ends
# — last_catalog_scope and last_ambiguous_candidates are conversation modes
# rather than answers to one question, and expiring them mid-clarification would
# break the exchange they exist to hold open.
DECAYING_SLOTS = ("last_indicator_name", "last_indicator_detail_id", "last_countries",
                  "last_country_group", "last_period_expression",
                  "last_explicit_frequency", "last_extremum", "last_metrics",
                  "answer_length")


def turn_index(state: dict) -> int:
    return int((state or {}).get("turn_index") or 0)


def remember(state: dict, key: str, value) -> None:
    """Writes a slot and records WHICH TURN wrote it.

    The stamp is what makes staleness visible. Without it a slot carries no
    information about whether it belongs to the question being asked or to one
    the user finished with several turns ago, and `carry_forward` has to treat
    both the same.
    """
    state[key] = value
    if key in DECAYING_SLOTS:
        state.setdefault("_slot_turns", {})[key] = turn_index(state)


def is_fresh(state: dict, key: str) -> bool:
    """Whether a slot is recent enough for a follow-up to inherit it.

    Unstamped slots count as fresh. Sessions that predate stamping, and slots
    written by a path that has not been converted to `remember`, must keep
    working — a memory feature that silently drops context is worse than the
    unbounded one it replaced.
    """
    stamped = ((state or {}).get("_slot_turns") or {}).get(key)
    if stamped is None:
        return True
    return turn_index(state) - int(stamped) <= STALE_AFTER_TURNS


def forget_stale(state: dict) -> dict:
    """Drops every decaying slot the current turn is too far from.

    Applied once per turn rather than checked at each read, so a slot that has
    aged out is gone from the state a later stage inspects too — several places
    in the pipeline read `last_indicator_name` directly, outside carry_forward,
    and a freshness test that only guarded one of them would leave the others
    reaching past the window.
    """
    for key in DECAYING_SLOTS:
        if key in state and not is_fresh(state, key):
            state.pop(key, None)
            (state.get("_slot_turns") or {}).pop(key, None)
    return state


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

    def record(self, session_id: str, user_message: str, answer: str, state: dict,
               message_id: Optional[str] = None, slots: Optional[dict] = None,
               language: Optional[str] = None) -> None:
        with self._lock:
            self._evict()
            session = self._session(session_id)
            session["turns"].append({"user": (user_message or "").strip()[:MAX_STORED_QUESTION],
                                      "assistant": (answer or "").strip()[:MAX_STORED_TEXT],
                                      "message_id": message_id,
                                      # What this turn was actually ABOUT, as the
                                      # pipeline resolved it. See render_context.
                                      "slots": dict(slots or {}),
                                      "language": language})
            session["state"] = dict(state or {})

    def find_turn(self, session_id: str, message_id: str) -> Optional[dict]:
        """The exchange a message id refers to, if it is still in memory.

        Used so a rating can be stored with the question and answer it is
        about. Turns fall out of the window after twelve exchanges and sessions
        after two hours of silence, so this returns None routinely — a rating
        arriving without them is still recorded, just with less around it.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            for turn in reversed(session["turns"]):
                if turn.get("message_id") == message_id:
                    return dict(turn)
        return None

    # The resolved facts worth writing into the transcript, and the order they
    # read in. Not every slot a turn produced — only the ones a follow-up refers
    # back to.
    _SLOT_ORDER = ("indicator", "period", "countries", "group", "computation")

    @staticmethod
    def _render_slots(slots: dict) -> str:
        """The turn's resolved subject, as one annotation line.

        The transcript used to be prose alone, which asks the router to re-derive
        from an English sentence what the pipeline had already established. "The
        latest reading for Real GDP was 185.17 Bn QAR" does say the indicator,
        but it says it in the same voice as every other noun in the sentence, and
        the period it reports may not be the period that was asked for.

        Written as a compact annotation rather than sentences because it is read
        by a model under a token budget, and because it must not be mistaken for
        something the user said.
        """
        parts = []
        for key in ConversationStore._SLOT_ORDER:
            value = (slots or {}).get(key)
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value if v)
            if value:
                parts.append(f"{key}={value}")
        return f"[resolved: {'; '.join(parts)}]" if parts else ""

    def render_context(self, session_id: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """A transcript, newest turns prioritised, each annotated with what it
        resolved to.

        Trimming takes whole turns from the FRONT, so the most recent exchange
        is never the one cut in half — a follow-up refers to what just happened,
        and half a turn is worse than one fewer turn.

        Turns are numbered so "the year before that" has something to point at,
        and so the router can tell adjacent from distant: a question that only
        makes sense against turn 5 while turn 1 is on screen is a different kind
        of follow-up from one continuing the exchange just above it.
        """
        with self._lock:
            self._evict()
            session = self._sessions.get(session_id)
            turns = list(session["turns"]) if session else []

        if not turns:
            return ""

        rendered, total = [], 0
        for offset, turn in enumerate(reversed(turns)):
            number = len(turns) - offset
            block = f"[turn {number}] User: {turn['user']}\nAssistant: {turn['assistant']}"
            annotation = self._render_slots(turn.get("slots") or {})
            if annotation:
                block += f"\n{annotation}"
            if total + len(block) > max_chars and rendered:
                break
            rendered.append(block)
            total += len(block)
        return "\n\n".join(reversed(rendered))

    def recent_turns(self, session_id: str, count: int = COMPOSER_HISTORY_TURNS) -> list:
        """The last few exchanges, for the Composer rather than the router.

        A different view of the same turns from render_context: no slot
        annotations and no turn numbers, because the Composer is not resolving
        references — it is avoiding repeating itself, and machine annotations in
        that prompt were narrated back at the reader the first time an internal
        field reached it (see _public in composer_agent).
        """
        with self._lock:
            self._evict()
            session = self._sessions.get(session_id)
            turns = list(session["turns"])[-count:] if session else []
        return [{"user": t.get("user"), "assistant": t.get("assistant")} for t in turns]

    def last_language(self, session_id: str) -> Optional[str]:
        """The language of the most recent turn that had one.

        Language is detected per message from its script, which is right for a
        message that HAS script to read and wrong for "2024", "top 3" or "GDP" —
        a follow-up in an Arabic conversation with no Arabic characters in it.
        Those were answered in English, mid-thread.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            turns = list(session["turns"]) if session else []
        for turn in reversed(turns):
            if turn.get("language"):
                return turn["language"]
        return None

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


# Openers that can only be continuations — a message starting with one of these
# is asking about something already on screen, whatever the router decided.
_CONTINUATION_OPENER = re.compile(
    r"^\s*(and|what about|how about|ok(ay)?[,\s]+and|but|also|then|same for|"
    r"now|just|only|make it|show me (just|only)|give me (just|only))\b"
    r"|^\s*(و|وماذا عن|ماذا عن|وكيف|وما|طيب|فقط|اختصر|وبالنسبة)",
    re.IGNORECASE,
)

# Wording that reaches back at something rather than naming it. Deliberately
# narrow: "that", "those", "it", "the same", "the previous one".
_ANAPHORIC = re.compile(
    r"\b(that|those|these|the same|the previous|the one (before|above)|"
    r"it|them|the above)\b"
    r"|نفس|ذلك|تلك|السابق|أعلاه",
    re.IGNORECASE,
)


def looks_like_followup(user_message: str) -> bool:
    """A deterministic second opinion on is_followup.

    carry_forward and the limit branch in decide_route both hang off a single
    boolean an LLM produced, with nothing corroborating it. When the router
    misses — and it does, on exactly the terse messages follow-ups are made of
    — the question is answered as though it named its own subject, which for
    "and 2024?" means no subject at all.

    This only ever votes YES. It cannot mark a message as a fresh question,
    because the shapes it recognises are sufficient for a follow-up and not
    necessary for one: plenty of follow-ups open with none of them. A rule that
    could also veto would start overriding the router on questions it reads
    correctly, which is the mistake decide_route was cleaned up to stop making.
    """
    text = (user_message or "").strip()
    if not text:
        return False
    if _CONTINUATION_OPENER.search(text):
        return True
    if _ANAPHORIC.search(text):
        return True
    # A message of a few words carrying no verb is a fragment answering the
    # previous turn — "Q2 2025", "Saudi Arabia too", "monthly".
    return len(text.split()) <= 4 and "?" not in text


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
    if not state:
        return intent
    if not intent.get("is_followup"):
        # The router said no. It gets a second opinion, and only on shapes that
        # cannot be anything else — see looks_like_followup.
        if not looks_like_followup(user_message):
            return intent
        intent = dict(intent)
        intent["is_followup"] = True
        # Recorded so a reader of the logs can tell which of the two decided it,
        # rather than having to reproduce the router's output to find out.
        intent["_followup_inferred"] = True

    carried = dict(intent)
    # Every inherited slot below is additionally gated on freshness: a value
    # written more than STALE_AFTER_TURNS ago belongs to a question the user has
    # moved on from, and inheriting it answers that one instead of this one.
    if (not carried.get("indicator_phrase") and state.get("last_indicator_name")
            and is_fresh(state, "last_indicator_name")):
        carried["indicator_phrase"] = state["last_indicator_name"]
    # last_metrics (a multi-metric answer) is deliberately NOT restored here.
    # It is handled in handle_message, which rejoins the list so the normal
    # split path re-runs the snapshot; flattening it into indicator_phrase at
    # this point would make it look like one indicator name.
    if not carried.get("countries_mentioned") and not carried.get("country_group_mentioned"):
        if state.get("last_countries") and is_fresh(state, "last_countries"):
            carried["countries_mentioned"] = list(state["last_countries"])
        if state.get("last_country_group") and is_fresh(state, "last_country_group"):
            carried["country_group_mentioned"] = state["last_country_group"]
    asks_latest = bool(_ASKS_FOR_LATEST.search(user_message or ""))
    # A follow-up that states its own period must not inherit the old one. The
    # model often leaves period_expression empty even when the message says
    # "what about in 2024", and inheriting then pinned the answer to the
    # previous turn's year: the metrics changed correctly and the period never
    # did, so three indicators came back for 2023 again.
    states_own_period = parse_period_expression(user_message or "").kind != "unspecified"
    if (not carried.get("period_expression") and state.get("last_period_expression")
            and is_fresh(state, "last_period_expression")
            and not asks_latest and not states_own_period):
        carried["period_expression"] = state["last_period_expression"]
    if (not carried.get("explicit_frequency") and state.get("last_explicit_frequency")
            and is_fresh(state, "last_explicit_frequency")):
        carried["explicit_frequency"] = state["last_explicit_frequency"]
    return carried
