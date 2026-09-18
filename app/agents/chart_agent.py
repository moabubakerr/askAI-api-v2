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
Chart Agent: given query result rows, decides chart type and shape,
and returns a small declarative spec the frontend renders (e.g. with
Plotly.js or Recharts). The LLM never writes chart-rendering code —
it only picks type/fields, which removes an entire class of bugs.
"""
import json
from app.core.llm_client import chat, llm_client
from app.core.config import settings

SYSTEM_PROMPT = """You choose how to visualize economic data. Given a user question and
a list of data rows (as JSON), output ONLY a JSON object of this exact shape:

{
  "chart_type": "line" | "bar" | "grouped_bar" | "pie",
  "title": "<short chart title>",
  "x_field": "<column name to use for x-axis / categories>",
  "y_field": "<column name to use for y-axis / values>",
  "series_field": "<column name to split multiple series by, or null>"
}

Pick "line" for time series, "bar" for single-period comparisons across categories,
"grouped_bar" for comparing categories across a couple of periods, "pie" only for
composition/share-of-total questions with a handful of categories.
Use exact column names as given in the data. Output ONLY the JSON, nothing else.
"""


def build_chart_spec(user_message: str, rows: list[dict]) -> dict | None:
    if not rows:
        return None

    sample = rows[:50]
    user_prompt = f"Question: {user_message}\n\nData columns and sample rows:\n{json.dumps(sample)}"

    raw = chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=300,
    )
    try:
        spec = json.loads(raw.strip().strip("```json").strip("```"))
        spec["data"] = rows
        return spec
    except json.JSONDecodeError:
        return None
