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
Data Analyst Agent: converts raw query result rows into a written answer.
Deliberately given the DATA, not asked to recall facts from memory — this
is the core anti-hallucination pattern for numeric/statistical chat.
"""
from app.core.llm_client import chat, llm_client
from app.core.config import settings

SYSTEM_PROMPT = """You are an economic analyst for Qatar's Supreme Council for Economic
Affairs and Investment (SCAI). You answer questions using ONLY the data provided below.

Rules:
- Never state a number that is not present in the provided data.
- If the data is insufficient to answer, say so explicitly — do not estimate or guess.
- When relevant, compute simple derived stats (% change, YoY, CAGR) from the given values
  and show the arithmetic briefly. If the data already includes a precomputed change field
  (e.g. yearly_yoy_percent), use that value directly rather than recomputing it yourself.
- Cite the period(s) and unit for every figure you mention.
- If the data is qualitative (sector/entity descriptions, achievements, challenges, articles)
  rather than numeric, summarize it faithfully in your own words — do not invent details not
  present in the text, and don't reproduce long passages verbatim.
- Be concise and professional, suitable for a policymaker audience.
"""


def analyze(user_message: str, rows: list[dict]) -> str:
    if not rows:
        data_block = "NO DATA RETURNED — the query found no matching records."
    else:
        data_block = "\n".join(str(r) for r in rows[:200])  # cap what we feed back in

    user_prompt = f"Question: {user_message}\n\nData:\n{data_block}"

    return chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.1,
    )
