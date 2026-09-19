"""
Deterministic orchestration graph — v2 architecture, built in response to the
SCAI QC report. The old graph.py (Text-to-SQL agent + LLM analyst doing
arithmetic) is superseded by this pipeline for anything numeric.

Flow: Intent (LLM, extraction only) -> Resolve (indicator/country/period,
all deterministic) -> Retrieve (parameterized SQL) -> Compute (pure Python,
see app/compute/engine.py) -> Compose (LLM, phrasing only) -> Verify
(non-LLM regex check; falls back to a template if the LLM slipped a number
that isn't in the facts payload) -> Attach sources (non-LLM, always runs).

Every answer gets a "Sources:" footer built deterministically from the
citations attached during dispatch — this is NOT something the Composer is
merely asked to include, since an instruction the LLM might skip on some
phrasing isn't a real guarantee. See app/compute/citations.py.

Conversation state (for follow-ups, fixing F-030) is a small dict the caller
persists per session and passes back in as `session_state`; this function
returns an updated one.
"""
from typing import TypedDict, Optional

from app.nlu.intent_agent import extract_intent
from app.resolvers.indicator_resolver import resolve_indicator
from app.resolvers.country_resolver import resolve_countries
from app.resolvers.period_resolver import parse_period_expression, parse_explicit_frequency, choose_granularity
from app.db import retriever
from app.compute import engine as compute
from app.compute.verifier import verify_numbers, render_template_fallback
from app.compute.citations import Citation, citations_for_rows, render_sources_footer, citations_to_dicts
from app.compute.chart_builder import build_chart_spec
from app.compute.chart_request import wants_chart
from app.agents.composer_agent import compose_answer


PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4, "yearly": 1}


class SessionState(TypedDict, total=False):
    last_indicator_detail_id: str
    last_indicator_name: str
    last_countries: list
    last_period_expression: str


def _find_row_by_period(rows: list[dict], period_label: Optional[str]) -> Optional[dict]:
    if period_label is None:
        return None
    return next((r for r in rows if r.get("period_label") == period_label), None)


def _capabilities_answer(language: str) -> tuple[dict, list[Citation]]:
    from sqlalchemy import text
    with retriever.engine.connect() as conn:
        n_ind = conn.execute(text("SELECT COUNT(*) FROM indicators WHERE is_published")).scalar()
        sector_rows = conn.execute(text(
            "SELECT name_en, sector_id FROM sectors WHERE is_active ORDER BY name_en"
        )).fetchall()
    sectors = [r[0] for r in sector_rows]
    citations = [Citation(indicator=None, data_source=None, table="indicators", record_id=None)]
    citations += [Citation(indicator=r[0], data_source=None, table="sectors", record_id=r[1]) for r in sector_rows]
    payload = {
        "ok": True,
        "facts": {
            "published_indicator_count": n_ind,
            "sectors_covered": sectors,
            "capability_note": (
                "I answer questions about Qatar's published economic indicators "
                "(latest values, trends, comparisons across periods or benchmark "
                "countries, rankings) using only SCAI's approved data. I retrieve "
                "and compute from that data — I do not estimate or forecast."
            ),
        },
    }
    return payload, citations


def handle_message(user_message: str, conversation_context: str = "",
                    session_state: Optional[SessionState] = None) -> dict:
    session_state = dict(session_state or {})
    intent = extract_intent(user_message, conversation_context)
    ctype = intent.get("computation_type", "out_of_scope")
    language = intent.get("language", "en")

    # --- non-data intents ---
    if ctype == "general_chat":
        payload = {"ok": True, "facts": {"note": "Greeting — no data needed."}}
        return _finish(payload, language, session_state, [], skip_compose=True,
                        canned="Hello! Ask me about Qatar's published economic indicators — "
                               "latest values, trends, or comparisons.")

    if ctype == "capabilities":
        payload, citations = _capabilities_answer(language)
        return _finish(payload, language, session_state, citations)

    if ctype == "out_of_scope":
        payload = {"ok": False, "message": "That's outside what I can help with — I only cover "
                                            "Qatar's published SCAI economic indicators."}
        return _finish(payload, language, session_state, [])

    if ctype == "count_list":
        indicator_type_hint = intent.get("indicator_phrase") or ""
        rows = retriever.count_indicators_by_type(indicator_type_hint)
        citations = [Citation(indicator=r.get("name_en"), data_source=None, table="indicators",
                               record_id=r.get("record_id")) for r in rows]
        if not rows:
            rows = retriever.get_sector_indicators(indicator_type_hint)
            citations = [Citation(indicator=r.get("name_en"), data_source=None, table="sectors",
                                   record_id=r.get("sector_record_id")) for r in rows]
        payload = {"ok": True, "facts": {"count": len(rows), "names": [r.get("name_en") for r in rows]}} \
            if rows else {"ok": False, "message": f"No indicators found matching \"{indicator_type_hint}\"."}
        return _finish(payload, language, session_state, citations)

    if ctype == "macro_overview":
        payload, citations = _macro_overview()
        if payload.get("ok") and wants_chart(user_message, ctype):
            payload["chart"] = build_chart_spec(ctype, payload["facts"], "Economic Overview", None, None)
        return _finish(payload, language, session_state, citations)

    # --- everything below needs an indicator resolved first ---
    indicator_phrase = intent.get("indicator_phrase")
    if not indicator_phrase and intent.get("is_followup") and session_state.get("last_indicator_name"):
        indicator_phrase = session_state["last_indicator_name"]

    resolution = resolve_indicator(indicator_phrase or "")
    if resolution.status != "resolved" and resolution.status != "inactive":
        payload = {"ok": False, "message": resolution.message}
        return _finish(payload, language, session_state, [])

    match = resolution.match
    session_state["last_indicator_detail_id"] = match.indicator_detail_id
    session_state["last_indicator_name"] = match.name_en

    # Must be P02's PublishedIndicatorDetailId, NOT indicator_detail_id — they are
    # different GUIDs, and published_data_points keys on the former
    # (schema.sql:86). Passing the source id here matched zero rows for every
    # published indicator, and because the retriever only falls back to
    # indicator_values when this is None, a non-empty wrong id also suppressed
    # the fallback: every data question answered "No approved data points were
    # found", regardless of indicator or period.
    published_detail_id = match.published_detail_id if match.is_published else None
    available_gran = retriever.get_available_granularities(published_detail_id, match.indicator_detail_id)
    explicit_gran = parse_explicit_frequency(intent.get("explicit_frequency"))
    granularity = choose_granularity(explicit_gran, available_gran)

    if granularity is None:
        wanted = intent.get("explicit_frequency")
        payload = {"ok": False, "message": (
            f"\"{match.name_en.strip()}\" has no {wanted} data in the approved dataset "
            f"(available: {', '.join(sorted(available_gran)) or 'none'})."
        )}
        return _finish(payload, language, session_state, [])

    period = parse_period_expression(intent.get("period_expression"))
    countries_res = resolve_countries(intent.get("countries_mentioned", []), intent.get("country_group_mentioned"))

    payload, citations = _dispatch_computation(ctype, intent, match, published_detail_id, granularity, period, countries_res)

    if payload.get("ok") and wants_chart(user_message, ctype):
        chart = build_chart_spec(ctype, payload["facts"], match.name_en, match.unit_en, match.format)
        if chart:
            payload["chart"] = chart

    return _finish(payload, language, session_state, citations)


def _dispatch_computation(ctype, intent, match, published_detail_id, granularity, period, countries_res):
    unit = match.unit_en or ""
    indicator_name = match.name_en
    data_source_en = match.data_source_en

    if ctype in ("country_comparison", "country_ranking"):
        series_by_country = retriever.get_series_multi_country(
            match.indicator_detail_id, published_detail_id, granularity,
            countries_res.resolved_countries,
            start_date=period.start_date, end_date=period.end_date,
            include_qatar=True,
        )
        period_label = None
        note = (f"No approved data source recognized for: {', '.join(countries_res.unresolved_names)}."
                if countries_res.unresolved_names else None)
        result = (compute.country_ranking(series_by_country, period_label, ascending=True)
                  if ctype == "country_ranking"
                  else compute.country_comparison(series_by_country, period_label))

        citations = []
        if result.ok:
            entries = result.facts.get("ranked") or result.facts.get("rows") or []
            for entry in entries:
                country_rows = series_by_country.get(entry["country"], [])
                row = _find_row_by_period(country_rows, entry.get("period_label"))
                if row:
                    citations.append(Citation(indicator_name, data_source_en, row.get("source_table", "unknown"),
                                               row.get("record_id"), row.get("period_label"), entry["country"]))
        return _wrap(result, unit, extra_note=note), citations

    rows = retriever.get_series(match.indicator_detail_id, published_detail_id, granularity,
                                 start_date=period.start_date, end_date=period.end_date)

    if ctype == "latest_value":
        result = compute.latest_value(rows)
        used = [rows[-1]] if rows else []
        return _wrap(result, unit), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "trend":
        result = compute.trend(rows)
        return _wrap(result, unit), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "min_max":
        which = intent.get("extremum") or "max"
        result = compute.min_max(rows, which)
        used = [_find_row_by_period(rows, result.facts.get("period_label"))] if result.ok else []
        used = [r for r in used if r]
        return _wrap(result, unit), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "difference":
        result = compute.difference(rows)
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("high_period")),
                                 _find_row_by_period(rows, result.facts.get("low_period"))) if r]
        return _wrap(result, unit), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "growth_rate":
        if len(rows) < 2:
            return {"ok": False, "message": "Not enough data points to compute a growth rate."}, []
        method = (intent.get("growth_method_hint") or "cagr").lower()
        ppy = PERIODS_PER_YEAR.get(granularity, 1)
        result = compute.growth_rate(rows, rows[0]["period_label"], rows[-1]["period_label"], method, ppy)
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("period_start")),
                                 _find_row_by_period(rows, result.facts.get("period_end"))) if r]
        return _wrap(result, unit), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_comparison":
        if len(rows) < 2:
            return {"ok": False, "message": "Could not find two distinct periods to compare."}, []
        result = compute.period_to_period_change(rows, rows[0]["period_label"], rows[-1]["period_label"])
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("period_a")),
                                 _find_row_by_period(rows, result.facts.get("period_b"))) if r]
        return _wrap(result, unit), citations_for_rows(used, indicator_name, data_source_en)

    return {"ok": False, "message": "I couldn't determine what computation this question needs."}, []


def _wrap(result: compute.ComputeResult, unit: str, extra_note: Optional[str] = None) -> dict:
    payload = {"ok": result.ok, "facts": {**result.facts, "unit": unit}} if result.ok else \
              {"ok": False, "message": result.message}
    if extra_note and payload.get("ok"):
        payload["facts"]["note"] = extra_note
    return payload


def _macro_overview():
    """Fixes F-007/F-018: a deterministic curated overview instead of
    silently falling back to a single indicator. HEADLINE_NAMES is an
    editorial list — refine it once you know which indicators SCAI wants
    surfaced as the standard macro snapshot."""
    HEADLINE_NAMES = ["Real GDP", "Inflation", "Trade Balance", "Government Revenues"]
    facts, citations = [], []
    for name in HEADLINE_NAMES:
        res = resolve_indicator(name)
        if res.status != "resolved":
            continue
        # Same distinction as in handle_message: published_data_points keys on
        # P02's PublishedIndicatorDetailId, not on indicator_detail_id.
        published_id = res.match.published_detail_id if res.match.is_published else None
        avail = retriever.get_available_granularities(published_id, res.match.indicator_detail_id)
        gran = choose_granularity(None, avail)
        if not gran:
            continue
        rows = retriever.get_series(res.match.indicator_detail_id, published_id, gran)
        latest = compute.latest_value(rows)
        if latest.ok:
            facts.append({"indicator": res.match.name_en, "unit": res.match.unit_en,
                          "granularity": gran, **latest.facts})
            if rows:
                citations += citations_for_rows([rows[-1]], res.match.name_en, res.match.data_source_en)
    if not facts:
        return {"ok": False, "message": "No headline indicators were available to build an overview."}, []
    return {"ok": True, "facts": {"overview": facts}}, citations


def _finish(payload: dict, language: str, session_state: dict, citations: list[Citation],
            skip_compose: bool = False, canned: Optional[str] = None) -> dict:
    if skip_compose:
        answer = canned or ""
    else:
        draft = compose_answer(payload, language)
        clean, offending = verify_numbers(draft, payload)
        answer = draft if clean else render_template_fallback(payload, language)
        if not clean:
            payload["_verifier_rejected_numbers"] = offending  # for logging/debugging only

    # Sources footer: always appended deterministically, never left to the
    # LLM's discretion. Only skipped when there's genuinely nothing to cite
    # (greetings, or a refusal where no data was retrieved at all).
    footer = render_sources_footer(citations)
    if footer:
        answer = f"{answer}\n\n{footer}"

    payload["citations"] = citations_to_dicts(citations)

    return {
        "answer": answer,
        "facts_payload": payload,
        "session_state": session_state,
    }
