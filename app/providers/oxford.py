"""Oxford Economics, reached through their MCP server.

MCP is JSON-RPC 2.0 over one HTTP POST per call. This server needs no session:
`initialize` returns no Mcp-Session-Id, and tools/call works on a bare request,
so there is no handshake to keep alive and no state to lose between turns. That
is why this is ~100 lines of httpx and not an MCP SDK dependency — pulling one
in would buy transport features this endpoint does not use.

Verified against the live service on 2026-09-26:

    POST https://services.oxfordeconomics.com/mcp
    Api-Key: <key>                          # NOT x-api-key, which 401s
    Accept: application/json, text/event-stream
    -> tools: EconomicData, EconomicAnalysis

TWO TOOLS, and the split is the useful part:

  EconomicData      numbers. Returns a markdown table (region, year, variable,
                    value, units). Capped at 10 datapoints per call, one
                    indicator per call — cross-region comparisons of the SAME
                    indicator are fine in one question.
  EconomicAnalysis  prose. Summaries and quotations from Oxford's published
                    research — assumptions, risks, methodology, outlook.

PASS-THROUGH, DELIBERATELY. Whatever Oxford returns is shown to the user as
Oxford said it — no verification, no filtering, no correcting a figure that
looks wrong. This is the opposite of the rule for SCAI's own answers, and the
difference is the point: our pipeline states a number only after checking it
against the rows it retrieved, because the number is ours to stand behind.
Oxford's number is Oxford's, presented under their name, and a reader
comparing the two houses needs to see what each actually said.

So do not add a verifier here. Silently dropping or adjusting one of their
figures would leave the user comparing our answer against an edited version of
theirs, without knowing it had been edited — and it would hide exactly the
disagreements a side-by-side comparison exists to surface. If one of their
values looks wrong, that is a finding to raise with Oxford, not a bug to patch
in this module.

Nothing here is verified against the facts payload, because there is no payload
to verify against: the answer arrives as text Oxford composed. Callers must
label it as Oxford's and must not report it as `verified`. See the note on
AskResponse in app/api/main.py.
"""
import json
import re
import uuid
from typing import Optional

import httpx

from app.core.config import settings
from app.core.llm_client import chat, router_client

DATA_TOOL = "EconomicData"
ANALYSIS_TOOL = "EconomicAnalysis"

# One stable conversation id per SCAI session, so Oxford's own follow-up
# handling sees a conversation rather than a stream of unrelated first
# questions. uuid5 rather than a stored mapping: the id has to survive a
# restart, and deriving it costs nothing.
_CONVERSATION_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Picking between the two tools is a reading-comprehension task, so the router
# model does it rather than a word list. Keyword matching cannot separate "what
# is the GDP growth outlook" (a number) from "explain the GDP growth outlook"
# (prose) — both contain "outlook" — and it has to be maintained twice over,
# once per language this product answers in.
_TOOL_CHOICE_PROMPT = f"""You route an economics question to one of two Oxford Economics tools.

- "{DATA_TOOL}": the question asks for figures — a value, a rate, a level, a
  count, a ranking, or the same indicator across several regions or years. The
  answer would be a table of numbers.
- "{ANALYSIS_TOOL}": the question asks for reasoning — why something happened,
  what is expected and on what assumptions, risks, drivers, methodology, or a
  summary of published research. The answer would be prose.
- "both": the question genuinely needs figures AND an explanation of them to be
  answered at all. Use this sparingly: it costs a second slow call to the
  vendor, so do not pick it merely because an explanation might be a nice
  addition to a number.

The question may be in English or Arabic. Judge what the asker wants, not which
words appear — "what is the growth outlook" wants a number, while "explain the
growth outlook" wants prose.

Respond ONLY with JSON: {{"tool": "{DATA_TOOL}" | "{ANALYSIS_TOOL}" | "both"}}"""


class OxfordError(RuntimeError):
    """A call that did not produce an answer. Carries text fit to show a user."""


def available() -> bool:
    """False when no key is configured, which is the default.

    Checked before every call so that a deployment which never sets
    OXFORD_API_KEY cannot accidentally send a user's question off-premises.
    """
    return bool(settings.OXFORD_API_KEY and settings.OXFORD_MCP_URL)


def conversation_id(session_id: str) -> str:
    return str(uuid.uuid5(_CONVERSATION_NAMESPACE, session_id or "default"))


def tools_for(question: str, mode: str = "auto") -> list[str]:
    """Which of the two tools to call.

    "auto" asks the router model to read the question. An explicit mode skips
    that call entirely — a caller that already knows what it wants should not
    pay a classification round-trip to be told.

    Falls back to the data tool when the model is unreachable or answers with
    something unexpected. That direction is deliberate: the data tool returns a
    sourced table, so a misroute shows the reader a number they can check,
    where the wrong way round returns commentary that answers a question about
    figures with none in it.
    """
    if mode == "data":
        return [DATA_TOOL]
    if mode == "analysis":
        return [ANALYSIS_TOOL]
    if mode == "both":
        return [DATA_TOOL, ANALYSIS_TOOL]

    try:
        raw = chat(
            client=router_client,
            model=settings.ROUTER_MODEL_NAME,
            system=_TOOL_CHOICE_PROMPT,
            user=question or "",
            temperature=0.0,
            max_tokens=50,
        )
        choice = json.loads(raw.strip().strip("```json").strip("```")).get("tool")
    except Exception:  # noqa: BLE001 — model down or unparseable; see docstring
        return [DATA_TOOL]

    if choice == "both":
        return [DATA_TOOL, ANALYSIS_TOOL]
    return [choice] if choice in (DATA_TOOL, ANALYSIS_TOOL) else [DATA_TOOL]


def _rpc(method: str, params: dict, timeout: float) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        resp = httpx.post(
            settings.OXFORD_MCP_URL,
            json=payload,
            timeout=timeout,
            headers={
                # Confirmed with the vendor. `x-api-key` returns 401.
                "Api-Key": settings.OXFORD_API_KEY,
                "Content-Type": "application/json",
                # The spec allows a server to answer either way. This one
                # replies with plain JSON, but the header is required.
                "Accept": "application/json, text/event-stream",
            },
        )
    except httpx.TimeoutException as exc:
        raise OxfordError(
            f"Oxford Economics did not respond within {int(timeout)}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise OxfordError(f"Could not reach Oxford Economics: {exc}") from exc

    if resp.status_code == 401 or resp.status_code == 403:
        raise OxfordError("Oxford Economics rejected the API key.")
    if resp.status_code >= 400:
        raise OxfordError(f"Oxford Economics returned HTTP {resp.status_code}.")

    try:
        body = resp.json()
    except ValueError as exc:
        raise OxfordError("Oxford Economics returned a response that was not JSON.") from exc

    # JSON-RPC puts application errors in a 200 body, so a status check alone
    # would report a refused query as a successful empty answer.
    if body.get("error"):
        raise OxfordError(str(body["error"].get("message") or "Oxford Economics returned an error."))
    return body.get("result") or {}


def _text_of(result: dict) -> str:
    """The text blocks of an MCP tool result, joined.

    The ONLY thing this module drops, and it is a rendering limit rather than a
    judgement about content: non-text blocks (images, embedded resources) have
    no place to go in a text answer, and neither tool has yet returned one. The
    text itself is passed through byte for byte — see the pass-through note at
    the top of this module.

    If these tools do start returning images, this is where to render them, not
    where to decide they were not worth showing.
    """
    parts = [
        _plain_text(block.get("text", ""))
        for block in (result.get("content") or [])
        if block.get("type") == "text"
    ]
    return "\n\n".join(p for p in parts if p).strip()


# Oxford composes its prose for a web client, so it arrives with markup in it —
# "<br /> Qatar: Fiscal deterioration highlights near-term challenge, Sep, 2026."
# is how it separates an answer from the report it came from. This surface
# renders Markdown, not HTML, so those tags reached the reader as literal text.
_HTML_BREAK = re.compile(r"(?i)<\s*br\s*/?\s*>|</\s*(?:p|div|li|tr|h[1-6])\s*>")
_HTML_TAG = re.compile(r"<[^>]+>")


def _plain_text(text: str) -> str:
    """Oxford's answer with its markup rendered rather than printed.

    A break tag is where Oxford ended a line, so it becomes a line break — the
    citation it separates is meant to sit on its own. Every other tag is
    dropped, not escaped: the text inside it is the answer, and the tag is
    presentation for a viewer this one is not.

    Deliberately narrow. This is the one thing besides non-text blocks that the
    module does not pass through byte for byte, and it stops at markup — no
    rewording, no trimming, no reflowing of what Oxford actually wrote.
    """
    if not text or "<" not in text:
        return text or ""
    cleaned = _HTML_BREAK.sub("\n", text)
    cleaned = _HTML_TAG.sub("", cleaned)
    # The tag often sits mid-sentence with a space either side, which leaves a
    # line beginning with one once the tag becomes a newline.
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# Whether the question already says which country it is about. Only the names
# that appear in these questions: Qatar itself, the GCC neighbours it is
# usually compared with, and the Arabic forms.
_NAMES_A_COUNTRY = re.compile(
    r"\b(qatar\w*|saudi|uae|emirat\w*|kuwait\w*|bahrain\w*|oman\w*|gcc|gulf|"
    r"singapore\w*|norway|world\w*|glob\w*|region\w*|countr\w*|compare\w*|"
    r"economies|nations?|international|everywhere|across)\b"
    r"|قطر|السعودي|الإمارات|الامارات|الكويت|البحرين|عمان|الخليج|سنغافورة|العالم|الدول",
    re.IGNORECASE)

# Appended when it does not. Oxford covers every economy it models, so "what's
# the real GDP value" is a question it can answer about two hundred countries —
# and it did, returning a table led by Paraguay and Uzbekistan. This is a SCAI
# product: a question that names no country is a question about Qatar, and the
# asker should not have to say so every time.
#
# Worded as a default rather than a filter, and only added when the question is
# silent, so "compare Qatar with Saudi Arabia" and "which economies grew
# fastest" are left exactly as they were typed.
_DEFAULT_COUNTRY_NOTE = (
    " (If no country is specified in this question, answer about Qatar — "
    "report Qatar's figures and Qatar's outlook, not a cross-country list.)"
)


def _with_default_country(question: str) -> str:
    """Qatar, unless the asker named somewhere else."""
    if not question or _NAMES_A_COUNTRY.search(question):
        return question or ""
    return question.strip() + _DEFAULT_COUNTRY_NOTE


def call_tool(tool: str, question: str, language: str = "en",
              session_id: str = "default", timeout: Optional[float] = None) -> str:
    result = _rpc(
        "tools/call",
        {
            "name": tool,
            "arguments": {
                "question": _with_default_country(question),
                "language": (language or "en")[:2],
                "conversationId": conversation_id(session_id),
            },
        },
        timeout if timeout is not None else settings.OXFORD_TIMEOUT_S,
    )
    text = _text_of(result)
    if result.get("isError"):
        raise OxfordError(text or "Oxford Economics could not answer that question.")
    if not text:
        raise OxfordError("Oxford Economics returned no answer for that question.")
    return text


def ask(question: str, language: str = "en", session_id: str = "default",
        mode: str = "auto") -> dict:
    """One question, one answer, plus which tools produced it.

    Raises OxfordError with a message fit for a reader. The caller decides
    whether that becomes an error response or one half of a combined answer
    that still has SCAI's side to show.
    """
    if not available():
        raise OxfordError("Oxford Economics is not configured on this deployment.")

    tools, sections, failures = tools_for(question, mode), [], []
    for tool in tools:
        try:
            sections.append((tool, call_tool(tool, question, language, session_id)))
        except OxfordError as exc:
            failures.append(str(exc))

    if not sections:
        # Every tool failed for the same reason when there was one tool; with
        # two, say both rather than pick one and hide the other.
        raise OxfordError(" ".join(dict.fromkeys(failures)))

    if len(sections) == 1:
        answer = sections[0][1]
    else:
        # Headed, because the two tools answer different questions and a reader
        # needs to know whether they are looking at a measured figure or at
        # Oxford's commentary on it.
        heads = {DATA_TOOL: "**Data**", ANALYSIS_TOOL: "**Analysis**"}
        answer = "\n\n".join(f"{heads[t]}\n\n{body}" for t, body in sections)

    return {"answer": answer, "tools_used": [t for t, _ in sections],
            "partial_errors": failures}
