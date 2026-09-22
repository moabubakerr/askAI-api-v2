"""
Read-only queries behind the /admin endpoints.

SQL lives here rather than in the route handlers, the same way the retriever
holds the app's own queries: every statement is a fixed, parameterised SELECT
written by a person. Filters arrive as bound parameters, never as string
interpolation, so a filter value cannot become SQL.

Every filter is written as "parameter IS NULL OR <condition>", so one query
serves every combination of filters. Assembling a WHERE clause from whichever
filters happened to be supplied is how injection gets written by accident, and
it also means each combination is a differently-shaped query nobody has run.

Nothing here writes. The app connects as scai_ro, which holds INSERT on two
application tables and nothing else, so an admin route physically cannot alter
SCAI's data even if one of these were wrong.
"""
from typing import Optional

from sqlalchemy import text
from app.core.config import settings
from app.db.executor import engine


def _page(limit: Optional[int], offset: Optional[int]) -> tuple:
    """A sane window, whatever the caller asked for.

    An unbounded list over a table that grows with every question is a slow
    query waiting to happen, so the ceiling is enforced here rather than
    trusted to the caller.
    """
    size = settings.ADMIN_PAGE_SIZE if limit is None else int(limit)
    size = max(1, min(size, settings.ADMIN_MAX_PAGE_SIZE))
    return size, max(0, int(offset or 0))


def _rows(sql: str, params: dict) -> list:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params).fetchall()]


def _scalar(sql: str, params: dict) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(sql), params).scalar() or 0)


def _one(sql: str, params: dict) -> dict:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


_MESSAGE_FILTERS = """
  WHERE (:from_ts  IS NULL OR m.asked_at >= CAST(:from_ts AS timestamptz))
    AND (:to_ts    IS NULL OR m.asked_at <  CAST(:to_ts   AS timestamptz))
    AND (:language IS NULL OR m.language = :language)
    AND (:answered IS NULL OR m.answered = CAST(:answered AS boolean))
    AND (:shape    IS NULL OR m.answer_shape = :shape)
    AND (:q_like   IS NULL OR m.question ILIKE :q_like OR m.answer ILIKE :q_like)
"""


def messages(from_ts=None, to_ts=None, language=None, answered=None,
             shape=None, q=None, limit=None, offset=None) -> dict:
    size, start = _page(limit, offset)
    params = {"from_ts": from_ts, "to_ts": to_ts, "language": language,
              "answered": answered, "shape": shape,
              "q_like": f"%{q}%" if q else None,
              "limit": size, "offset": start}
    total = _scalar(f"SELECT count(*) FROM chat_messages m {_MESSAGE_FILTERS}", params)
    rows = _rows(f"""
        SELECT m.message_id, m.session_id, m.asked_at, m.endpoint, m.language,
               m.question, m.answer, m.answered, m.answer_shape, m.indicator,
               m.period_label, m.verified, m.readable, m.latency_ms,
               f.rating, f.comment
        FROM chat_messages m
        LEFT JOIN message_feedback f ON f.message_id = m.message_id
        {_MESSAGE_FILTERS}
        ORDER BY m.asked_at DESC
        LIMIT :limit OFFSET :offset
    """, params)
    return {"total": total, "limit": size, "offset": start, "rows": rows}


_FEEDBACK_FILTERS = """
  WHERE (:min_rating  IS NULL OR f.rating >= CAST(:min_rating AS int))
    AND (:max_rating  IS NULL OR f.rating <= CAST(:max_rating AS int))
    AND (:has_comment IS NULL
         OR (CAST(:has_comment AS boolean) IS TRUE  AND f.comment IS NOT NULL)
         OR (CAST(:has_comment AS boolean) IS FALSE AND f.comment IS NULL))
"""


def feedback(min_rating=None, max_rating=None, has_comment=None,
             limit=None, offset=None) -> dict:
    size, start = _page(limit, offset)
    params = {"min_rating": min_rating, "max_rating": max_rating,
              "has_comment": has_comment, "limit": size, "offset": start}
    total = _scalar(
        f"SELECT count(*) FROM message_feedback f {_FEEDBACK_FILTERS}", params)
    rows = _rows(f"""
        SELECT f.feedback_id, f.message_id, f.session_id, f.rating, f.comment,
               f.created_at, f.question, f.answer,
               m.answer_shape, m.indicator, m.language, m.answered, m.latency_ms
        FROM message_feedback f
        LEFT JOIN chat_messages m ON m.message_id = f.message_id
        {_FEEDBACK_FILTERS}
        ORDER BY f.created_at DESC
        LIMIT :limit OFFSET :offset
    """, params)
    return {"total": total, "limit": size, "offset": start, "rows": rows}


def sessions(limit=None, offset=None) -> dict:
    """One row per conversation.

    A session is not a table — it is whatever messages share a session_id — so
    it is derived rather than stored, which avoids two sources of truth that
    can disagree.
    """
    size, start = _page(limit, offset)
    total = _scalar("SELECT count(DISTINCT session_id) FROM chat_messages", {})
    rows = _rows("""
        SELECT m.session_id,
               min(m.asked_at)                              AS started_at,
               max(m.asked_at)                              AS last_seen_at,
               count(*)                                     AS messages,
               count(*) FILTER (WHERE m.answered IS FALSE)  AS refusals,
               max(m.language)                              AS language,
               round(avg(m.latency_ms))                     AS avg_latency_ms,
               count(f.rating)                              AS ratings,
               round(avg(f.rating), 2)                      AS avg_rating
        FROM chat_messages m
        LEFT JOIN message_feedback f ON f.message_id = m.message_id
        GROUP BY m.session_id
        ORDER BY max(m.asked_at) DESC
        LIMIT :limit OFFSET :offset
    """, {"limit": size, "offset": start})
    return {"total": total, "limit": size, "offset": start, "rows": rows}


def stats() -> dict:
    """Headline numbers, and the same numbers split.

    Split by language deliberately: an Arabic failure rate averaged into one
    overall figure is how "SCAI's commentary comes back in English" went
    unnoticed. Split by answer shape because "which kinds of answer do people
    rate badly" is actionable in a way that one average is not.
    """
    return {
        "overall": _one("""
            SELECT count(*)                                      AS messages,
                   count(DISTINCT session_id)                    AS sessions,
                   count(*) FILTER (WHERE answered IS FALSE)     AS refusals,
                   round(100.0 * count(*) FILTER (WHERE answered IS FALSE)
                         / nullif(count(*), 0), 1)               AS refusal_rate_pct,
                   count(*) FILTER (WHERE verified IS FALSE)     AS unverified,
                   round(avg(latency_ms))                        AS avg_latency_ms,
                   max(asked_at)                                 AS last_message_at
            FROM chat_messages
        """, {}),
        "ratings": _one("""
            SELECT count(*)                                      AS ratings,
                   round(avg(rating), 2)                         AS avg_rating,
                   count(*) FILTER (WHERE rating <= 2)           AS low_ratings,
                   count(*) FILTER (WHERE comment IS NOT NULL)   AS with_comment
            FROM message_feedback
        """, {}),
        "rating_distribution": _rows("""
            SELECT rating, count(*) AS n FROM message_feedback
            GROUP BY rating ORDER BY rating
        """, {}),
        "by_language": _rows("""
            SELECT language,
                   count(*)                                      AS messages,
                   count(*) FILTER (WHERE answered IS FALSE)     AS refusals,
                   round(100.0 * count(*) FILTER (WHERE answered IS FALSE)
                         / nullif(count(*), 0), 1)               AS refusal_rate_pct,
                   round(avg(latency_ms))                        AS avg_latency_ms
            FROM chat_messages GROUP BY language ORDER BY messages DESC
        """, {}),
        "by_answer_shape": _rows("""
            SELECT m.answer_shape,
                   count(*)                                      AS messages,
                   count(f.rating)                               AS ratings,
                   round(avg(f.rating), 2)                       AS avg_rating
            FROM chat_messages m
            LEFT JOIN message_feedback f ON f.message_id = m.message_id
            GROUP BY m.answer_shape ORDER BY messages DESC
        """, {}),
        "top_indicators": _rows("""
            SELECT indicator, count(*) AS n FROM chat_messages
            WHERE indicator IS NOT NULL
            GROUP BY indicator ORDER BY n DESC LIMIT 20
        """, {}),
    }


# The published layer only, and one row per indicator: is_main picks the
# headline detail, without which Real GDP appears three times — once for each
# of its hydrocarbon and non-hydrocarbon components.
_CATALOGUE_SELECT = """
    SELECT i.indicator_id, d.indicator_detail_id, d.published_detail_id,
           i.name_en, i.name_ar, d.name_en AS detail_name_en,
           i.indicator_type_en, d.unit_en, d.unit_ar, d.format, d.polarity_en,
           d.target_value, d.target_year, d.baseline_value, d.baseline_year,
           d.data_source_en, d.definition_en,
           (SELECT count(*) FROM published_data_points p
             WHERE p.published_indicator_detail_id = d.published_detail_id
               AND p.actual IS NOT NULL)                       AS data_points,
           (SELECT min(p.period_label) FROM published_data_points p
             WHERE p.published_indicator_detail_id = d.published_detail_id
               AND p.actual IS NOT NULL)                       AS first_period,
           (SELECT max(p.period_label) FROM published_data_points p
             WHERE p.published_indicator_detail_id = d.published_detail_id
               AND p.actual IS NOT NULL)                       AS last_period
    FROM indicator_details d
    JOIN indicators i ON i.indicator_id = d.indicator_id
    WHERE d.is_published IS TRUE AND d.is_main IS TRUE
"""


def catalogue(q=None, limit=None, offset=None) -> dict:
    size, start = _page(limit, offset)
    params = {"q_like": f"%{q}%" if q else None, "limit": size, "offset": start}
    filter_sql = """
      AND (:q_like IS NULL OR i.name_en ILIKE :q_like OR i.name_ar ILIKE :q_like)
    """
    total = _scalar(f"""
        SELECT count(*) FROM indicator_details d
        JOIN indicators i ON i.indicator_id = d.indicator_id
        WHERE d.is_published IS TRUE AND d.is_main IS TRUE {filter_sql}
    """, params)
    rows = _rows(f"""
        {_CATALOGUE_SELECT} {filter_sql}
        ORDER BY i.name_en
        LIMIT :limit OFFSET :offset
    """, params)
    return {"total": total, "limit": size, "offset": start, "rows": rows}


def catalogue_item(indicator_id: str, max_points: int = 1000) -> Optional[dict]:
    """One indicator with its published series.

    Accepts any of the three ids the catalogue exposes. They are different
    GUIDs for the same thing — a caller holding one should not have to know
    which one this endpoint wanted.
    """
    row = _one(f"""
        {_CATALOGUE_SELECT}
          AND (i.indicator_id = :id OR d.indicator_detail_id = :id
               OR d.published_detail_id = :id)
    """, {"id": indicator_id, "q_like": None})
    if not row:
        return None
    row["series"] = _rows("""
        SELECT published_data_point_id AS record_id,
               period_label, period_date, granularity, country_en,
               actual, target, outlook
        FROM published_data_points
        WHERE published_indicator_detail_id = :pdid AND actual IS NOT NULL
        ORDER BY period_date, country_en NULLS FIRST
        LIMIT :limit
    """, {"pdid": row.get("published_detail_id"), "limit": max_points})
    # Where these numbers came from. The panel shows it beside the series so
    # "which spreadsheet is this from?" is answered on the same screen as the
    # figure being questioned, rather than being a separate investigation.
    row["provenance"] = _rows("""
        SELECT table_name, source_file, file_sha256, rows_in_file,
               rows_in_table, finished_at AS loaded_at
        FROM v_etl_current_load
        WHERE table_name IN ('published_data_points', 'indicator_details',
                             'indicators')
        ORDER BY table_name, source_file
    """, {})
    return row


def lineage() -> dict:
    """Every table, the export file it was loaded from, and when.

    Written by etl/load_data.py from the files it actually opened, so this is a
    record of the load that produced the data now in the database — not a
    mapping document that someone has to remember to update.

    `rows_in_file` minus `rows_in_table` is the interesting column. They differ
    on purpose in several places (61 nameless indicator stubs are dropped, as
    are child rows whose parent did not load), and a gap that appears where
    there was not one before is the first sign an export changed shape.
    """
    current = _rows("""
        SELECT load_id, table_name, source_file, loader, file_sha256,
               file_bytes, file_modified_at, rows_in_file, rows_in_table,
               started_at, finished_at
        FROM v_etl_current_load
        ORDER BY table_name, source_file
    """, {})
    # Tables the ETL does not load: the application's own, and the two index
    # scripts' output. Listing them as "no source file" is more useful than
    # omitting them, because their absence from the lineage view is itself the
    # thing worth knowing — nothing upstream will ever refresh them.
    unmanaged = _rows("""
        SELECT c.relname AS table_name, c.reltuples::bigint AS approx_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
          AND c.relname NOT IN (SELECT table_name FROM v_etl_current_load)
        ORDER BY c.relname
    """, {})
    return {
        "load": _one("""
            SELECT load_id, min(started_at) AS started_at,
                   max(finished_at) AS finished_at,
                   count(DISTINCT table_name) AS tables,
                   count(DISTINCT source_file) AS source_files,
                   sum(rows_in_file) AS rows_in_files
            FROM v_etl_current_load GROUP BY load_id
        """, {}),
        "tables": current,
        "not_loaded_by_etl": unmanaged,
        "history": _rows("""
            SELECT load_id, max(finished_at) AS finished_at,
                   count(DISTINCT table_name) AS tables,
                   sum(rows_in_table) AS rows_loaded
            FROM etl_load_log
            GROUP BY load_id ORDER BY max(finished_at) DESC LIMIT 20
        """, {}),
    }


def message_provenance(message_id: str) -> Optional[dict]:
    """One answer and the exact rows it was built from.

    The chain this whole feature exists for: question → answer → cited record →
    the data point's own numbers → the indicator → the export file it was
    loaded from.

    Returns None only when the message itself was never logged. A logged
    message with no citations is a real and ordinary result — greetings and
    refusals cite nothing — and comes back with an empty list rather than a 404,
    because "this answer used no data" is an answer to the question asked.
    """
    message = _one("""
        SELECT m.message_id, m.session_id, m.asked_at, m.endpoint, m.language,
               m.question, m.answer, m.answered, m.answer_shape, m.indicator,
               m.period_label, m.verified, m.readable, m.latency_ms,
               f.rating, f.comment
        FROM chat_messages m
        LEFT JOIN message_feedback f ON f.message_id = m.message_id
        WHERE m.message_id = :mid
    """, {"mid": message_id})
    if not message:
        return None
    citations = _rows("""
        SELECT position, source_table, record_id, indicator, data_source,
               period_label, country, record_found,
               published_indicator_detail_id, indicator_id, indicator_detail_id,
               indicator_name_en, indicator_name_ar, unit_en, is_main,
               actual, target, outlook, period_date, granularity,
               source_file, file_sha256, loaded_at
        FROM v_message_provenance
        WHERE message_id = :mid
        ORDER BY position
    """, {"mid": message_id})
    # `record_found` is false when a cited row no longer exists. Almost always
    # a reload: the ETL truncates the source layer, so ids do not survive it.
    # Reported rather than hidden — an admin checking an old answer against
    # today's data needs to know the row it quoted is gone, not be shown a
    # blank cell. Only meaningful for published_data_points, the one table the
    # view resolves against; for the rest it is always false and says nothing.
    unresolved = sum(1 for c in citations
                     if c["source_table"] == "published_data_points"
                     and c["record_id"] and not c["record_found"])
    return {"message": message, "citations": citations,
            "unresolved_records": unresolved}
