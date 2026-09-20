"""
Last-resort disambiguation: the model CHOOSES from the real catalogue.

Calibration against the live bge-m3 settled what the resolver could not do on
its own. Two questions whose right answer the embeddings ranked FIRST were
still refused, because the score sat under MIN_CONFIDENCE:

    "public debt"                             -> Public Debt as a Percentage
                                                 of GDP (%)      0.573
    "How much of Qatar's exports are non-oil" -> Non-Hydrocarbon Exports
                                                 (share of total exports) 0.542

and the highest score any nonsense question reached was 0.602, for "number of
penguins in Qatar". Lowering the threshold far enough to admit 0.542 would
admit the penguins first. There is no threshold that separates them, so no
amount of tuning fixes this — the ranking is right and the confidence is not.

That is the narrow gap this fills. It is deliberately a CHOICE, never a
generation: the model is shown names that exist and must answer with one of
them verbatim or with NONE, and the answer is checked against the list before
it is used. The alternative — asking a model what an indicator is called — is
how a system starts citing indicators that do not exist.

Only runs when deterministic resolution has already failed, so it costs a call
on the refusal path and nothing on the path that works.
"""
import re
from typing import Optional

from app.core.llm_client import chat, llm_client
from app.core.config import settings

SYSTEM_PROMPT = """You match a user's wording to an economic indicator, from a fixed list.

You are given the user's phrase and a numbered list of indicator names that
exist in Qatar's SCAI database. Reply with ONE of:
  - the indicator name copied EXACTLY as it appears in the list, character for
    character, including any bracketed text
  - the single word NONE

Reply with nothing else. No explanation, no punctuation around it, no number.

Choose a name ONLY when the user's phrase means that indicator. Everyday
wording counts: "non-oil" means non-hydrocarbon, "public debt" means a public
debt measure, "cost of living" means inflation, "how fast prices are rising"
means inflation.

Answer NONE whenever the list does not contain what was asked for. The list is
what happened to score closest, NOT a set of plausible answers — it always has
entries, including for questions this database cannot answer at all. A user
asking about penguins, solar power or the weather must get NONE, even though
the list in front of you will look full of economic indicators. Picking the
nearest one is the failure this exists to prevent: a confident wrong indicator
is worse than no answer, because the figure it returns is real and the reader
has no way to tell it answers a different question.

If two entries both fit, choose the more specific one. If the phrase names a
concept the list only relates to loosely, answer NONE."""


def pick_indicator(phrase: str, candidates: list[str],
                    language: str = "en") -> Optional[str]:
    """Returns a name from `candidates`, or None. Never returns anything else.

    The validation is the point, not a formality. The model's reply is matched
    back against the list it was given, so a paraphrase, a near-miss or an
    invented name all resolve to None and the caller keeps its refusal.
    """
    if not phrase or not candidates:
        return None

    listing = "\n".join(f"{i}. {name}" for i, name in enumerate(candidates, 1))
    try:
        reply = chat(
            client=llm_client,
            model=settings.LLM_MODEL_NAME,
            system=SYSTEM_PROMPT,
            user=f"User's phrase: {phrase}\n\nIndicators:\n{listing}\n\nAnswer:",
            temperature=0.0,
        )
    except Exception:
        # A disambiguation that cannot run is a refusal, not an error: the
        # caller already has a perfectly good "I could not tell which
        # indicator you meant" to fall back on.
        return None

    answer = (reply or "").strip().strip('"').strip("'")
    # Models like to prefix a list item with its number even when told not to.
    answer = re.sub(r"^\s*\d+[.)]\s*", "", answer).strip()
    if not answer or answer.upper() == "NONE":
        return None

    # Exact first, then case/whitespace-insensitive. Nothing looser: a
    # "close enough" rule here would let a paraphrase through, which is the
    # one thing this must not do.
    if answer in candidates:
        return answer
    folded = {c.casefold().strip(): c for c in candidates}
    return folded.get(answer.casefold().strip())
