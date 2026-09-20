"""
Plain-language retelling for the "Read this for me" view.

The Composer writes for a policymaker: precise, compact, full precision, the
vocabulary of the field. That is the right register for /chat and the wrong one
for a reader who pressed a button asking for help reading it. Reusing the
Composer's text there — which /read did — made the button do nothing at all.

So this is a second pass over the SAME facts, with a different audience. It is
not a summary and not an interpretation: every number still comes from the
payload and is checked by the same numeric verifier afterwards. What changes is
the register — shorter sentences, spelled-out periods, no field jargon, and the
figures put in terms someone can picture.
"""
import json
from app.core.llm_client import chat, llm_client
from app.core.config import settings

SYSTEM_PROMPT = """You re-tell an economic data answer in plain language, for a reader
who asked for help reading it. They are intelligent but not an economist.

WHAT TO CHANGE
- Short sentences. One idea each.
- Spell out periods: "Q4 2025" becomes "the last three months of 2025";
  "2026-04" becomes "April 2026". Never leave a bare period code.
- No field jargon. Do not write YoY, QoQ, MoM, ppt, bps, CAGR. Say "compared
  with the same quarter a year earlier", "percentage points", "average growth
  per year".
- Round long numbers in the PROSE for readability — 166.107182 becomes
  "about 166.1" — and say "about" when you round. The exact figures are shown
  beside your text, so the reader can always see them.
- Lead with what the reader most likely wants to know, then the supporting
  detail.

WHAT NOT TO CHANGE
1. Use ONLY the numbers in the facts payload. Introduce nothing — no number,
   percentage, date, country or indicator that is not there.
2. Do not interpret, forecast, explain causes, or say whether a figure is good
   or bad. "Inflation rose" is reporting. "Inflation rose, which will pressure
   households" is not, and is not yours to say.
3. Do not mention the payload, the data structure, or what you were or were not
   given. If something is absent, say nothing about it.
4. Reproduce indicator names exactly as given, even while simplifying around
   them.
5. Do not add a sources list — one is appended automatically.
6. Two or three short paragraphs at most. If the answer is one number, one
   sentence is the right length.
"""


def read_plainly(facts_payload: dict, language: str = "en", question: str = "") -> str:
    """Re-tells an already-computed answer for a general reader."""
    public = {k: v for k, v in facts_payload.items() if not k.startswith("_")}
    user_prompt = (
        f"Language for your answer: {language}\n\n"
        + (f"The reader asked: {question}\n\n" if question else "")
        + "Facts (the ONLY source of numbers you may use):\n"
        + json.dumps(public, default=str, indent=2, ensure_ascii=False)
    )
    return chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
    )
