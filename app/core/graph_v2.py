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
from datetime import date
from typing import TypedDict, Optional

from app.nlu.intent_agent import extract_intent
from app.core.conversation import carry_forward
from app.core.messages import msg, detect_language, is_greeting
from app.resolvers.indicator_resolver import resolve_indicator, has_usable_definition
from app.resolvers.country_resolver import resolve_countries
from app.resolvers.period_resolver import parse_period_expression, parse_explicit_frequency, choose_granularity
from app.db import retriever
from app.compute import engine as compute
from app.compute.verifier import verify_numbers, render_template_fallback
from app.compute.citations import Citation, citations_for_rows, render_sources_footer, citations_to_dicts
from app.compute.chart_builder import build_chart_spec
from app.compute.chart_request import wants_chart
from app.agents.composer_agent import compose_answer
# Definitions in indicator_details are inconsistently wrapped in HTML
# ("<p>The increase in the general level of prices...</p>") because they come
# from a rich-text CMS field. Reusing the ETL's stripper rather than writing a
# second one — it is a pure regex helper with no pandas/DB dependency.
from etl.utils import strip_html


PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4, "yearly": 1}

# Above this, a match is treated as certain enough to answer without comment.
# Between MIN_CONFIDENCE (0.55, in indicator_resolver) and this, the answer is
# still given but the match is disclosed. Provisional: it needs calibrating
# against real bge-m3 scores over a question set, not guessing.
CONFIDENT_MATCH = 0.75


class SessionState(TypedDict, total=False):
    """What a follow-up can inherit. Written every turn, read by
    conversation.carry_forward() on the next one."""
    last_indicator_detail_id: str
    last_indicator_name: str
    last_countries: list
    last_country_group: str
    last_period_expression: str
    last_explicit_frequency: str


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
    # A follow-up states only what changed. Inherit the rest from the previous
    # turn before anything is resolved, so "and for Saudi Arabia?" keeps the
    # earlier indicator AND period instead of only the indicator.
    intent = carry_forward(intent, session_state, user_message)
    ctype = intent.get("computation_type", "out_of_scope")

    # Greetings are decided here, not by the model. It routed "اهلا" and
    # "كيف حالك" to general_chat but sent "سلام" and "سلام علبكم" down the data
    # path, where they hit indicator resolution and were answered "No indicator
    # was mentioned." A closed set of short phrases does not need a model, and
    # this way the behaviour is the same every time.
    if is_greeting(user_message):
        ctype = "general_chat"
    # Arabic script is unambiguous; the model's language field is a guess. An
    # Arabic greeting was being answered in English, which is the product simply
    # not working in one of its two languages.
    language = detect_language(user_message, intent.get("language"))

    # Remember what this turn was about, whatever happens below, so the next
    # follow-up has something to refer back to even if this one fails to
    # resolve an indicator.
    if intent.get("period_expression"):
        session_state["last_period_expression"] = intent["period_expression"]
    if intent.get("explicit_frequency"):
        session_state["last_explicit_frequency"] = intent["explicit_frequency"]
    if intent.get("countries_mentioned"):
        session_state["last_countries"] = list(intent["countries_mentioned"])
    if intent.get("country_group_mentioned"):
        session_state["last_country_group"] = intent["country_group_mentioned"]

    # --- non-data intents ---
    if ctype == "general_chat":
        payload = {"ok": True, "facts": {"note": "Greeting — no data needed."}}
        return _finish(payload, language, session_state, [], skip_compose=True,
                        canned=msg("greeting", language))

    if ctype == "capabilities":
        payload, citations = _capabilities_answer(language)
        return _finish(payload, language, session_state, citations)

    if ctype == "out_of_scope":
        payload = {"ok": False, "message": msg("out_of_scope", language)}
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
            if rows else {"ok": False, "message": msg("no_indicators_matching", language,
                                                        query=indicator_type_hint)}
        return _finish(payload, language, session_state, citations)

    if ctype == "macro_overview":
        payload, citations = _macro_overview(language)
        if payload.get("ok") and wants_chart(user_message, ctype):
            payload["chart"] = build_chart_spec(ctype, payload["facts"], "Economic Overview", None, None)
        return _finish(payload, language, session_state, citations)

    # --- everything below needs an indicator resolved first ---
    indicator_phrase = intent.get("indicator_phrase")
    if not indicator_phrase and intent.get("is_followup") and session_state.get("last_indicator_name"):
        indicator_phrase = session_state["last_indicator_name"]

    # A definition question is answered from catalog text, so a stub with no
    # data points is still a legitimate match. Every other path needs figures.
    resolution = resolve_indicator(indicator_phrase or "", require_data=(ctype != "definition"),
                                    language=language)
    if resolution.status != "resolved" and resolution.status != "inactive":
        payload = {"ok": False, "message": resolution.message}
        return _finish(payload, language, session_state, [])

    match = resolution.match
    session_state["last_indicator_detail_id"] = match.indicator_detail_id
    session_state["last_indicator_name"] = match.name_en

    # Definitional questions ("what is inflation?", F-030's "what does
    # non-hydrocarbon GDP mean?") are answered from indicator_details.definition_en,
    # which is SCAI's own wording — never from the model's general knowledge, and
    # never by falling through to a latest-value lookup. Only 261 of 644 details
    # carry a real definition, so a missing one is reported honestly rather than
    # filled in.
    if ctype == "definition":
        # The top match may be an empty catalog stub that shadows the real
        # indicator: "GDP" (no data, definition "-") outranks "Real GDP" (70
        # points, a full definition) on name similarity alone. If so, retry
        # over only those entries that HAVE a definition.
        #
        # That retry is deliberately fenced: its result is accepted only when
        # the alternative's name actually contains the phrase the user asked
        # about. Without that guard, asking about an indicator that genuinely
        # has no definition would return a confident definition of some
        # unrelated indicator instead — a silent substitution, and a worse
        # failure than admitting the gap, because a plausible definition of the
        # wrong thing reads as correct. "GDP" -> "Real GDP" passes the guard;
        # "Tanker" -> some unrelated indicator does not.
        if not has_usable_definition(match):
            phrase_key = (indicator_phrase or "").strip().lower()
            retry = resolve_indicator(indicator_phrase or "", require_data=False,
                                       require_definition=True, language=language)
            if (retry.status == "resolved" and phrase_key
                    and phrase_key in retry.match.name_en.strip().lower()):
                match = retry.match
                session_state["last_indicator_detail_id"] = match.indicator_detail_id
                session_state["last_indicator_name"] = match.name_en

        definition = match.definition_ar if language == "ar" and match.definition_ar else match.definition_en
        definition = strip_html(definition or "").strip()
        if not definition or definition == "-":
            payload = {"ok": False, "message": msg("no_definition", language,
                                                     indicator=match.name_en.strip())}
            return _finish(payload, language, session_state, [])
        payload = {"ok": True, "facts": {
            "indicator": match.name_en.strip(),
            "definition": definition,
            "unit": match.unit_en,
        }}
        citations = [Citation(indicator=match.name_en.strip(), data_source=match.data_source_en,
                              table="indicator_details", record_id=match.indicator_detail_id)]
        return _finish(payload, language, session_state, citations)

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
        # `wanted` is None when the user named no frequency, which rendered as
        # the literal 'has no None data'. Distinguish the two real cases: the
        # indicator has data but not at the frequency asked for, versus it has
        # no data at all.
        wanted = intent.get("explicit_frequency")
        if available_gran:
            message = msg("no_data_at_frequency", language, indicator=match.name_en.strip(),
                           wanted=wanted, available=", ".join(sorted(available_gran)))
        else:
            message = msg("no_data_at_all", language, indicator=match.name_en.strip())
        payload = {"ok": False, "message": message}
        return _finish(payload, language, session_state, [])

    period = parse_period_expression(intent.get("period_expression"))
    countries_res = resolve_countries(intent.get("countries_mentioned", []), intent.get("country_group_mentioned"))

    payload, citations = _dispatch_computation(ctype, intent, match, published_detail_id, granularity,
                                                period, countries_res, language)

    # Flag a shaky resolution so _finish can disclose it. The threshold sits
    # above MIN_CONFIDENCE (0.55) deliberately: clearing the bar to answer at
    # all is not the same as being sure enough to answer silently.
    if payload.get("ok") and match.confidence < CONFIDENT_MATCH:
        payload["_low_confidence_match"] = {"indicator": match.name_en.strip(),
                                            "confidence": round(match.confidence, 3)}

    if payload.get("ok") and wants_chart(user_message, ctype):
        chart = build_chart_spec(ctype, payload["facts"], match.name_en, match.unit_en, match.format)
        if chart:
            payload["chart"] = chart

    return _finish(payload, language, session_state, citations)


def _dispatch_computation(ctype, intent, match, published_detail_id, granularity, period,
                           countries_res, language="en"):
    unit = match.unit_en or ""
    indicator_name = match.name_en
    data_source_en = match.data_source_en

    if ctype in ("country_comparison", "country_ranking"):
        countries = list(countries_res.resolved_countries)
        note = (f"No approved data source recognized for: {', '.join(countries_res.unresolved_names)}."
                if countries_res.unresolved_names else None)

        # "Which country has the lowest inflation" names no countries, so the
        # resolved set is empty and the ranking was Qatar against itself — a
        # one-row league table (F-027). Fall back to SCAI's own benchmark set
        # for this indicator.
        #
        # Only for RANKING. A comparison must return exactly the countries the
        # user named and never a default benchmark list — that is F-001, where
        # "compare Qatar and Singapore" came back with six countries.
        if ctype == "country_ranking" and not countries:
            countries = retriever.get_benchmark_countries(published_detail_id)
            if not countries:
                return {"ok": False, "message": msg("no_benchmarks", language,
                                                     indicator=indicator_name.strip())}, []

        series_by_country = retriever.get_series_multi_country(
            match.indicator_detail_id, published_detail_id, granularity,
            countries,
            start_date=period.start_date, end_date=period.end_date,
            include_qatar=True,
        )
        # With no period named, align every country on the latest period they
        # ALL have, instead of letting each contribute its own latest row.
        #
        # A ranking asserts comparability. Ranking UAE's 2025-12 against
        # everyone else's 2026-04 does not support that assertion, and a
        # caveat only moves the problem to the reader. Aligning costs
        # freshness — here, four months for six countries — and that is the
        # right trade: a true ranking slightly out of date beats a current one
        # that isn't like-for-like.
        period_label = None
        if not (period.start_date or period.end_date):
            period_label = _latest_common_period(series_by_country)

        # Ranking direction was hardcoded ascending, so "which country has the
        # HIGHEST x" was answered with the lowest. extremum is exactly the field
        # the intent agent fills for that distinction.
        ascending = intent.get("extremum") != "max"
        result = (compute.country_ranking(series_by_country, period_label, ascending=ascending)
                  if ctype == "country_ranking"
                  else compute.country_comparison(series_by_country, period_label))

        citations = []
        if result.ok:
            entries = result.facts.get("ranked") or result.facts.get("rows") or []

            periods = {e.get("period_label") for e in entries if e.get("period_label")}
            if len(periods) > 1:
                # Only reachable when the countries share no period at all, so
                # alignment was impossible and each contributed its own latest.
                mixed = (f"These countries report no period in common, so each figure below is "
                         f"that country's own latest ({min(periods)} to {max(periods)}). "
                         f"This is not a like-for-like comparison.")
                note = f"{note} {mixed}" if note else mixed
            elif period_label:
                # Say which period was used and why it may not be the newest
                # available for every country — the alignment is invisible
                # otherwise, and a reader could reasonably assume "latest".
                aligned = (f"Compared at {period_label}, the most recent period all of these "
                           f"countries report.")
                note = f"{note} {aligned}" if note else aligned
            for entry in entries:
                country_rows = series_by_country.get(entry["country"], [])
                row = _find_row_by_period(country_rows, entry.get("period_label"))
                if row:
                    citations.append(Citation(indicator_name, data_source_en, row.get("source_table", "unknown"),
                                               row.get("record_id"), row.get("period_label"), entry["country"]))
        return _wrap(result, unit, extra_note=note, indicator_name=indicator_name), citations

    rows = retriever.get_series(match.indicator_detail_id, published_detail_id, granularity,
                                 start_date=period.start_date, end_date=period.end_date)
    rows = _apply_last_n_years(rows, period)

    # "What is the Real GDP forecast for 2026" was answered "No approved data
    # points were found for this indicator/period" — true, but it leaves the
    # user unable to tell whether the indicator is missing, the period is, or
    # they phrased it wrong. If the indicator has data outside the window they
    # asked for, say what IS there. Costs one extra query, only on the
    # empty-result path.
    if not rows and (period.start_date or period.end_date):
        available = retriever.get_series(match.indicator_detail_id, published_detail_id, granularity)
        if available:
            return {"ok": False, "message": msg(
                "no_data_for_period", language, indicator=indicator_name.strip(),
                granularity=granularity, period=period.raw or "that period",
                first=available[0]["period_label"], last=available[-1]["period_label"],
            )}, []

    if ctype == "latest_value":
        result = compute.latest_value(rows)
        # Cite the row the computation actually used, not rows[-1]. Those are no
        # longer the same: latest_value now skips future target-only rows, so
        # citing the last row would have attributed a 2025-12 reading to a
        # 2030-12 record — a citation pointing at the wrong number is worse than
        # none, since it looks like provenance.
        used = []
        if result.ok:
            row = _find_row_by_period(rows, result.facts.get("period_label"))
            used = [row] if row else []
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_ranking":
        # extremum "min" means lowest-first; anything else (including the usual
        # "highest to lowest" phrasing, and the null default) means highest-first.
        descending = intent.get("extremum") != "min"
        result = compute.period_ranking(rows, descending=descending)
        # Cite every row that appears in the ranking, not just the winner —
        # each listed figure is a claim of its own.
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "trend":
        result = compute.trend(rows)
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "min_max":
        which = intent.get("extremum") or "max"
        result = compute.min_max(rows, which)
        used = [_find_row_by_period(rows, result.facts.get("period_label"))] if result.ok else []
        used = [r for r in used if r]
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "difference":
        result = compute.difference(rows)
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("high_period")),
                                 _find_row_by_period(rows, result.facts.get("low_period"))) if r]
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(used, indicator_name, data_source_en)

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
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_comparison":
        if len(rows) < 2:
            return {"ok": False, "message": "Could not find two distinct periods to compare."}, []
        result = compute.period_to_period_change(rows, rows[0]["period_label"], rows[-1]["period_label"])
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("period_a")),
                                 _find_row_by_period(rows, result.facts.get("period_b"))) if r]
        return _wrap(result, unit, indicator_name=indicator_name), citations_for_rows(used, indicator_name, data_source_en)

    return {"ok": False, "message": "I couldn't determine what computation this question needs."}, []


def _apply_last_n_years(rows: list[dict], period) -> list[dict]:
    """Applies a "last N years" window, which nothing previously did.

    parse_period_expression() recognises the phrase and records n_years, but it
    never sets start_date, and start_date is the only thing get_series filters
    on — so "the last 5 years" and "the last 3 years" both returned the entire
    history, identically. That is QC findings F-022/F-023 exactly: the tester
    asked both and got the same answer.

    The window is anchored on the newest retrieved period rather than on
    today's date. Anchoring on wall-clock time would silently empty the window
    whenever a series lags the calendar, which several of these do."""
    if period.kind != "last_n_years" or not period.n_years:
        return rows
    dated = [r for r in rows if r.get("period_date")]
    if not dated:
        return rows
    anchor = max(r["period_date"] for r in dated)
    cutoff = date(anchor.year - period.n_years, anchor.month, 1)
    return [r for r in dated if r["period_date"] >= cutoff]


def _latest_common_period(series_by_country: dict[str, list[dict]]) -> Optional[str]:
    """The newest period every country with data actually reports.

    Countries with no data at all are ignored when intersecting — otherwise a
    single empty series would wipe out the intersection and force the whole
    ranking back onto mismatched periods. They are still reported separately as
    having no approved data.

    Returns None when the countries share no period at all, in which case the
    caller falls back to per-country latest and says so."""
    period_sets = []
    for rows in series_by_country.values():
        periods = {r["period_label"] for r in rows
                   if r.get("period_label") and r.get("actual") is not None}
        if periods:
            period_sets.append(periods)
    if not period_sets:
        return None
    common = set.intersection(*period_sets)
    return max(common) if common else None


def _wrap(result: compute.ComputeResult, unit: str, extra_note: Optional[str] = None,
          indicator_name: Optional[str] = None) -> dict:
    payload = {"ok": result.ok, "facts": {**result.facts, "unit": unit}} if result.ok else \
              {"ok": False, "message": result.message}
    # Name the indicator in every successful payload. Without it the Composer
    # has nothing to name and writes "the specified economic indicator", which
    # hides a mis-resolution: a question about solar energy was answered with a
    # "Share of Qatari Teachers" target and read as plausible, because the
    # answer text never said which indicator it used.
    if payload.get("ok") and indicator_name:
        payload["facts"]["indicator"] = indicator_name.strip()
    if extra_note and payload.get("ok"):
        payload["facts"]["note"] = extra_note
    return payload


def _macro_overview(language="en"):
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
        return {"ok": False, "message": msg("no_overview", language)}, []
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
    # Low-confidence match disclosure. Appended structurally, for the same
    # reason as the Sources footer: an instruction the Composer might drop on
    # some phrasing is not a guarantee.
    #
    # The case this exists for: "What percentage of Qatar's energy is generated
    # by solar?" resolved to "Share of Qatari Teachers" — the lexical overlap
    # between "percentage of Qatar's..." and "Share of Qatari..." was enough to
    # clear MIN_CONFIDENCE — and answered with its 2030 target of 50%. Nothing
    # in the reply signalled that the indicator was not what was asked about.
    # This does not stop a wrong match; it stops a wrong match reading as a
    # confident one.
    weak = payload.pop("_low_confidence_match", None)
    if weak:
        answer = (f"{answer}\n\nI matched your question to \"{weak['indicator']}\" "
                  f"(approximate match). If that isn't the indicator you meant, "
                  f"please name it exactly.")

    footer = render_sources_footer(citations)
    if footer:
        answer = f"{answer}\n\n{footer}"

    payload["citations"] = citations_to_dicts(citations)

    return {
        "answer": answer,
        "facts_payload": payload,
        "session_state": session_state,
    }
