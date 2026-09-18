"""
Executes LLM-generated SQL safely.

Key protections (non-negotiable for a data-integrity-sensitive chatbot):
  1. DB connection uses a READ-ONLY role at the Postgres level (defense in depth,
     don't rely on app-layer checks alone — create the role with GRANT SELECT only).
  2. Reject any statement that isn't a single SELECT.
  3. Hard row limit enforced even if the model forgets LIMIT.
  4. Query timeout so a bad query can't hang the connection pool.
"""
import re
import sqlalchemy
from sqlalchemy import create_engine, text
from app.core.config import settings

engine = create_engine(settings.POSTGRES_DSN, pool_pre_ping=True)

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|COPY)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(Exception):
    pass


def validate_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(";")

    if not stripped.upper().startswith("SELECT"):
        raise UnsafeSQLError("Only SELECT statements are permitted.")

    if FORBIDDEN_KEYWORDS.search(stripped):
        raise UnsafeSQLError("Query contains a forbidden write/DDL keyword.")

    if ";" in stripped:
        raise UnsafeSQLError("Multiple statements are not permitted.")

    # Enforce LIMIT if missing
    if "LIMIT" not in stripped.upper():
        stripped += f" LIMIT {settings.MAX_SQL_ROWS}"

    return stripped


def run_query(sql: str, timeout_seconds: int = 10) -> list[dict]:
    safe_sql = validate_sql(sql)

    with engine.connect() as conn:
        conn.execute(text(f"SET statement_timeout = {timeout_seconds * 1000}"))
        result = conn.execute(text(safe_sql))
        rows = [dict(row._mapping) for row in result]

    return rows
