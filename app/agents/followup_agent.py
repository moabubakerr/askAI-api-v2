"""
Follow-up suggestions — up to three questions the reader might ask next.

Offered only when they help. After a greeting, a refusal, a clarifying
question or an instruction like "in short", a row of suggested questions is
noise: the reader either has nothing to follow up on or has just told us what
they want. So there are two gates, and both have to pass:

  1. should_suggest — deterministic. No data answered, no suggestions.
  2. The model itself, which is told to return an empty list whenever the
     answer already closes the subject or the message was not a question.

A suggestion is a promise that the question can be answered, so the prompt
names what ASK AI can answer and forbids everything else — forecasts above
all, which SCAI does not publish and this product refuses. Nothing here states
a figure: suggestions are questions, and a number in one would be a claim no
verifier has checked, so any suggestion stating a value — a decimal, a
percentage, a number of 100 or more — is dropped. Periods and small counts
("Q4 2025", "the last 5 years") are fine.

Never fatal. A failure here costs the reader three buttons, not the answer.
"""
import json
import re
from typing import Optional

from app.core.config import settings
from app.core.llm_client import chat, router_client

MAX_SUGGESTIONS = 3
MAX_CHARS = 120

# Routes whose answer is not something to follow up on. Refusals and
# clarifications are caught separately, by payload["ok"] being false.
_NO_FOLLOW_UPS = {"general_chat", "out_of_scope"}

SYSTEM_PROMPT = """You are part of ASK AI, the AI assistant of Qatar's Supreme Council for
Economic Affairs and Investment (SCAI). The user just asked a question and got an
answer. Suggest what they are most likely to ask NEXT.

Every suggestion MUST take one of these shapes — they are the only questions
ASK AI can answer. Fill in <X> with the indicator and <period> with a year,
quarter or month; do not invent a new shape.
- "What is the latest <X>?"
- "What was <X> in <period>?"
- "How has <X> changed over the last N years?" / "Show the trend of <X> since <period>"
- "Compare <X> in <period> and <period>"
- "What was the growth rate of <X> from <period> to <period>?"
- "Is <X> improving?"
- "What was the highest (or lowest) <X> since <period>?" / "When was <X> highest?"
- "What does <X> mean?"
- "What is SCAI's analysis of <X>?"
The same shapes in Arabic are fine when the answer is in Arabic.

ASK AI CANNOT answer, so never suggest:
- a threshold or condition: "when was X last higher than <period>", "how many
  times did X exceed…", "how long has X been above…"
- a cause or a reason: "why did X fall", "what drove X"
- a forecast: what X WILL be, next year, by 2030
- a comparison with another country — and never name one: most indicators are
  reported for Qatar alone.

Rules:
1. At most 3 suggestions, each a short question (under 12 words) the USER would
   type, in the answer's language. Put the most likely one first.
2. Each must follow naturally from THIS question and answer — the next step a
   curious analyst would take: another period, a comparison, a related indicator. Not a random topic.
3. Name the indicator in each one, as the answer names it. Never write "it".
4. No figures and no invented facts. Years and quarters are fine; values are not.
5. Do not suggest anything already asked in the conversation.
6. Return an EMPTY list when suggestions would not help: the message was a
   greeting, thanks or an instruction about the answer (language, length,
   format); the answer is a refusal, an error or a request for clarification;
   or the answer already fully closes the subject. When in doubt, return [].

Respond ONLY with JSON: {"follow_ups": ["...", "..."]}"""


def should_suggest(payload: dict, computation: Optional[str]) -> bool:
    """The deterministic half of "only when needed"."""
    if not payload or payload.get("ok") is False:
        return False
    return (computation or "") not in _NO_FOLLOW_UPS


# Periods are allowed ("2024", "Q4 2025", "2025-04"), and so are small counts
# ("the last 5 years", "top 3"). What is left — a decimal, a percentage, a
# number of 100 or more — is a value, and a value in a suggestion is a claim
# no verifier has checked.
_PERIOD = re.compile(r"\b(?:19|20)\d{2}(?:-Q[1-4]|-\d{2})?\b|\bQ[1-4]\b", re.IGNORECASE)
_VALUE = re.compile(r"\d+[.,]\d+|\d+\s*[%٪]|\b\d{3,}\b")


def _states_a_value(text: str) -> bool:
    return bool(_VALUE.search(_PERIOD.sub(" ", text)))


# Other countries and regions. Suggestions never offer a country comparison:
# most indicators are reported for Qatar alone, and "how does Real GDP compare
# to Saudi Arabia" was suggested three times in one conversation and refused
# every time. The prompt says so; this is the backstop when the model does
# it anyway.
_OTHER_COUNTRY = re.compile(
    r"\b(saudi|ksa|uae|emirat\w*|kuwait\w*|bahrain\w*|oman\w*|singapore\w*|norway|"
    r"switzerland|gcc|gulf|other countries|countries|international|global|world)\b"
    r"|السعودي|الإمارات|الامارات|الكويت|البحرين|عمان|سنغافورة|النرويج|سويسرا|الخليج|الدول",
    re.IGNORECASE)


# Question shapes the pipeline has no route for. "When was the last time
# Government Revenues were higher than 2026-Q1?" was suggested, clicked, sent
# to the period comparison and refused with "I need two specific periods to
# compare" — a suggestion the product then could not answer. Also causes and
# forecasts, which the prompt forbids for the same reason.
_UNANSWERABLE = re.compile(
    r"\blast\s+time\b|\b(higher|lower|greater|less|more|above|below)\s+than\b"
    r"|\bhow\s+many\s+times\b|\bhow\s+long\b|\bexceed\w*\b|\bfall\s+below\b"
    r"|\bwhy\b|\bwhat\s+(caused|drove|drives|causes)\b|\breason\b"
    r"|\bwill\b|\bforecast\w*\b|\bproject\w*\b|\bnext\s+(year|quarter|month)\b|\bby\s+20\d{2}\b"
    r"|آخر\s+مرة|أعلى\s+من|أقل\s+من|أكثر\s+من|كم\s+مرة|لماذا|ما\s+سبب|توقع\w*|العام\s+المقبل",
    re.IGNORECASE)


def _clean(items, question: str, asked: set) -> list[str]:
    out = []
    for item in items if isinstance(items, list) else []:
        text = str(item or "").strip().strip('"').strip()
        key = text.lower().rstrip("?؟ ")
        if (not text or len(text) > MAX_CHARS or key in asked
                or key == (question or "").strip().lower().rstrip("?؟ ")
                or _states_a_value(text)
                or _OTHER_COUNTRY.search(text)
                or _UNANSWERABLE.search(text)):
            continue
        if key not in {o.lower().rstrip("?؟ ") for o in out}:
            out.append(text)
    return out[:MAX_SUGGESTIONS]


def suggest_follow_ups(question: str, answer: str, language: str = "en",
                       history: Optional[list] = None,
                       subject: Optional[dict] = None) -> list[str]:
    """Up to three next questions, or [] when none would help."""
    earlier = [t.get("user") for t in (history or []) if t.get("user")]
    user_prompt = (
        f"Language for suggestions: {language}\n\n"
        f"Earlier questions in this conversation: {json.dumps(earlier, ensure_ascii=False)}\n\n"
        f"What this answer was about: {json.dumps(subject or {}, ensure_ascii=False)}\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n{(answer or '')[:1500]}"
    )
    try:
        raw = chat(
            client=router_client,
            model=settings.ROUTER_MODEL_NAME,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.2,
            max_tokens=200,
        )
        parsed = json.loads(raw.strip().strip("```json").strip("```"))
    except Exception:  # noqa: BLE001 — model down or unparseable; see module docstring
        return []
    asked = {str(q).strip().lower().rstrip("?؟ ") for q in earlier}
    return _clean(parsed.get("follow_ups") if isinstance(parsed, dict) else None,
                  question, asked)
