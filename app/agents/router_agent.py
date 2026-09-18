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
Router Agent: classifies the incoming question so the graph can branch.
Uses the small/fast model — this is a cheap classification task, no need
to burn the 70B model's latency budget on it.
"""
import json
from app.core.llm_client import chat, router_client
from app.core.config import settings

INTENTS = ["data_lookup", "trend_analysis", "comparison", "chart_request", "general_chat", "out_of_scope"]

SYSTEM_PROMPT = f"""You are an intent classifier for a Qatari economic-data assistant (SCAI).
Classify the user's message into exactly one of: {INTENTS}.

- data_lookup: asking for a single figure/value (e.g. "what was GDP growth in 2023?")
- trend_analysis: asking about change over time (e.g. "how has inflation moved since 2020?")
- comparison: comparing indicators, sectors, or periods (e.g. "compare 2022 vs 2023 GDP")
- chart_request: explicitly wants a chart/graph/plot
- general_chat: greetings, meta questions about the assistant
- out_of_scope: anything unrelated to Qatari economic data (e.g. sports, other countries' data not in our DB)

Respond ONLY with JSON: {{"intent": "<intent>", "wants_chart": true|false}}
wants_chart should be true if a chart/visualization would help, regardless of the primary intent.
"""


def classify(user_message: str) -> dict:
    raw = chat(
        client=router_client,
        model=settings.ROUTER_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_message,
        temperature=0.0,
        max_tokens=100,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fail safe: treat as data_lookup rather than crash
        return {"intent": "data_lookup", "wants_chart": False}
