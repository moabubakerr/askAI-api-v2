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
Text-to-SQL Agent: turns a natural language question + conversation context
into a single safe SELECT statement against the indicators DB.
"""
from app.core.llm_client import chat, llm_client
from app.core.config import settings
from app.db.schema import get_schema_prompt
from app.db.executor import run_query, UnsafeSQLError

SYSTEM_TEMPLATE = """You are a SQL generator for a PostgreSQL database of Qatari economic
indicators maintained by SCAI. Given a user question, output ONE valid SELECT statement.

{schema}

Output ONLY the SQL. No explanation, no markdown fences, no comments.
If the question cannot be answered from this schema, output exactly: NO_QUERY
"""


def generate_sql(user_message: str, conversation_context: str = "") -> str:
    system = SYSTEM_TEMPLATE.format(schema=get_schema_prompt(settings.MAX_SQL_ROWS))
    user_prompt = f"Conversation so far:\n{conversation_context}\n\nQuestion: {user_message}"

    sql = chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system=system,
        user=user_prompt,
        temperature=0.0,
    )
    return sql.strip().strip("```sql").strip("```").strip()


def generate_and_run(user_message: str, conversation_context: str = "") -> dict:
    """Returns {sql, rows, error}. Retries once with the DB error fed back
    to the model — this recovers a large fraction of first-try SQL mistakes."""
    sql = generate_sql(user_message, conversation_context)

    if sql == "NO_QUERY":
        return {"sql": None, "rows": [], "error": "NO_QUERY"}

    try:
        rows = run_query(sql)
        return {"sql": sql, "rows": rows, "error": None}
    except (UnsafeSQLError, Exception) as e:
        # one retry with the error message included
        retry_prompt = (
            f"The previous query failed.\nQuery: {sql}\nError: {e}\n"
            f"Fix it. Output ONLY the corrected SQL."
        )
        fixed_sql = chat(
            client=llm_client,
            model=settings.LLM_MODEL_NAME,
            system=SYSTEM_TEMPLATE.format(schema=get_schema_prompt(settings.MAX_SQL_ROWS)),
            user=retry_prompt,
            temperature=0.0,
        ).strip().strip("```sql").strip("```").strip()

        try:
            rows = run_query(fixed_sql)
            return {"sql": fixed_sql, "rows": rows, "error": None}
        except Exception as e2:
            return {"sql": fixed_sql, "rows": [], "error": str(e2)}
