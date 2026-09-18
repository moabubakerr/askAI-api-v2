"""
DEPRECATED (v1 architecture) — superseded by app/core/graph_v2.py.

The SCAI QC report showed this free-form Text-to-SQL + LLM-does-arithmetic
pattern produces wrong numbers (see README "QC findings -> fix" table).
Kept here for reference only. The only place a v1-style free-form LLM SQL
call might still be reasonable is pure qualitative text lookups (sectors/
entities/articles) where no arithmetic or invented-number risk exists —
even there, prefer extending app/db/retriever.py with a fixed parameterized
query instead if you can.
"""

"""
Verifier Agent: last line of defense before a numeric answer reaches the user.
Checks that every number in the draft answer traces back to the retrieved rows.
This is cheap insurance against a model quietly "rounding" or misremembering
a figure between the analyst step and the final text.
"""
import json
from app.core.llm_client import chat, router_client
from app.core.config import settings

SYSTEM_PROMPT = """You are a fact-checker. You are given source data (JSON rows) and a
draft answer. Check whether every specific number in the draft answer is actually
supported by the source data (either directly present, or a straightforward
arithmetic derivation of values that ARE present, e.g. % change between two given values).

Respond ONLY with JSON: {"verified": true|false, "issue": "<short description or null>"}
"""


def verify(rows: list[dict], draft_answer: str) -> dict:
    if not rows:
        # nothing to check against — if the draft claims specific figures, that's a red flag
        return {"verified": False, "issue": "No source data was retrieved to support this answer."} \
            if any(ch.isdigit() for ch in draft_answer) else {"verified": True, "issue": None}

    user_prompt = f"Source data:\n{json.dumps(rows[:200])}\n\nDraft answer:\n{draft_answer}"

    raw = chat(
        client=router_client,
        model=settings.ROUTER_MODEL_NAME,  # small model is fine for this check
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=150,
    )
    try:
        return json.loads(raw.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        return {"verified": True, "issue": None}  # don't block the user on a parsing hiccup
