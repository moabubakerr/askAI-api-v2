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
import re
from datetime import date
from typing import TypedDict, Optional

from app.nlu.intent_agent import extract_intent
from app.core.conversation import carry_forward
from app.core.messages import msg, detect_language, is_greeting
from app.resolvers.indicator_resolver import (resolve_indicator, has_usable_definition,
                                               resembles_catalogue_name, has_identifying_content)
from app.resolvers.country_resolver import resolve_countries
from app.resolvers.period_resolver import (parse_period_expression, parse_explicit_frequency,
                                            choose_granularity, parse_period_pair,
                                            parse_relative_pair, year_earlier_label,
                                            single_period_label, period_kind,
                                            parse_same_period_pair, validate_period_labels)
from app.db import retriever
from app.compute import engine as compute
from app.compute.verifier import verify_numbers, render_template_fallback
from app.compute.citations import Citation, citations_for_rows, render_sources_footer, citations_to_dicts
from app.compute.chart_builder import build_chart_spec
from app.compute.chart_request import wants_chart
from app.compute.formatting import display_unit, decimals_from_format
from app.agents.composer_agent import compose_answer
from app.agents.article_agent import answer_from_articles
from app.resolvers.embeddings import get_embedding
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

# How many article passages to hand the answering model. Six ~900-char excerpts
# is roughly 5k characters, which fits the 16384-token budget alongside the
# question and the answer.
ARTICLE_PASSAGES = 6

# Cosine DISTANCE (0 = identical), so lower is closer. A vector search always
# returns a nearest neighbour however unrelated the corpus is, so without a
# ceiling "what has SCAI written about penguins" would confidently answer from
# whichever article happened to be least distant. Provisional — calibrate it
# against real questions before trusting it; scripts/calibrate_resolver.py is
# the same idea for indicator matching.
ARTICLE_MAX_DISTANCE = 0.62

# A second article joins the answer only if its best chunk is within this much
# of the winner's. Beyond it, the two are about different things and including
# both dilutes the context rather than enriching it.
ARTICLE_SECOND_MARGIN = 0.04

# The chunk pool searched before grouping by article. Larger than the number of
# passages actually used, because the right article's best chunk can sit below
# several chunks of a thematically-adjacent one — which is exactly how "the
# trade war" lost to five passages about Strait of Hormuz tolls.
ARTICLE_SEARCH_POOL = 30


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
    # Named, because a citation with indicator=None rendered as a literal
    # "• None — SCAI Indicator Catalog" in the sources footer.
    citations = [Citation(indicator="Published indicator catalogue", data_source=None,
                           table="indicators", record_id=None)]
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
    # Set when "which performed best?" is redirected onto a country ranking, so
    # the direction comes from the indicator's polarity rather than from a
    # superlative the user never used.
    performance_followup = False

    # Greetings are decided here, not by the model. It routed "اهلا" and
    # "كيف حالك" to general_chat but sent "سلام" and "سلام علبكم" down the data
    # path, where they hit indicator resolution and were answered "No indicator
    # was mentioned." A closed set of short phrases does not need a model, and
    # this way the behaviour is the same every time.
    if is_greeting(user_message):
        ctype = "general_chat"

    # Catalogue questions are pinned here too. "List the indicators in Sectors"
    # worked, then broke when the intent prompt grew two more types — count_list
    # was the only type with no explicit rule and lost to the newer ones, so the
    # question fell through to indicator resolution and was answered "I couldn't
    # tell which indicator you're asking about". The phrasing is recognisable
    # (a list/count request naming a real sector or indicator type), so it does
    # not need to depend on the model getting it right.
    # "which are the best performing ones" after a sector listing. Pinned here
    # for the same reason and checked BEFORE count_list, because the phrasings
    # overlap ("what are the top performing indicators" is both a list request
    # and a ranking) and the ranking is the more specific reading.
    # "Is Qatar's economy growing?" was sent to indicator resolution and refused
    # with "name the metric more plainly" — asking the user to already know the
    # catalogue in order to ask the most natural question there is about an
    # economy. It is the same question as "how is Qatar's economy doing", which
    # already worked, and the difference between them was one word.
    elif _asks_about_the_economy(user_message):
        ctype = "macro_overview"
    elif _asks_for_direction_split(user_message):
        ctype = "scope_direction"
    elif _asks_for_performance_ranking(user_message):
        ctype = "scope_performance"
    elif _looks_like_catalog_request(user_message):
        ctype = "count_list"
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
                        canned=msg("greeting", language),
                        question=user_message)

    if ctype == "capabilities":
        payload, citations = _capabilities_answer(language)
        return _finish(payload, language, session_state, citations,
                        question=user_message)

    if ctype == "out_of_scope":
        payload = {"ok": False, "message": msg("out_of_scope", language)}
        return _finish(payload, language, session_state, [],
                        question=user_message)

    if ctype == "count_list":
        indicator_type_hint = intent.get("indicator_phrase") or user_message or ""
        # The hint is a fragment of a question ("indicators in diversification
        # target"), and it used to be used as a whole-string ILIKE pattern, so
        # it matched nothing — the catalog value is "Economic Diversification
        # Targets". Match on shared words against the real type and sector
        # names instead.
        kind, scope = _match_catalog_scope(indicator_type_hint)
        if kind == "sector":
            rows = retriever.get_sector_indicators(scope)
            citations = [Citation(indicator=r.get("name_en"), data_source=None, table="sectors",
                                   record_id=r.get("sector_record_id")) for r in rows]
        elif kind == "type":
            rows = retriever.count_indicators_by_type(scope)
            citations = [Citation(indicator=r.get("name_en"), data_source=None, table="indicators",
                                   record_id=r.get("record_id")) for r in rows]
        else:
            rows, citations = [], []
        # Remember WHICH group was listed. "what are the most performing ones"
        # is only answerable as a follow-up to this turn, and without the scope
        # it has nothing to rank.
        if kind:
            session_state["last_catalog_scope"] = {"kind": kind, "scope": scope}
        names = [r.get("name_en") for r in rows]
        payload = {"ok": True, "facts": {
            "count": len(names),
            # What the question actually matched, so the answer can say "101
            # published Sector Indicators" instead of the vaguer "101 indicators
            # related to various sectors" the model invented for itself.
            "scope": scope, "scope_kind": kind,
            # A short sample for the prose. The full list stays in `names` for
            # the frontend to render; asking the model to narrate 101 entries
            # produced a wall of text that then ran out of tokens mid-sentence.
            "names_sample": names[:8],
            "names": names,
        }} if rows else {"ok": False, "message": msg("no_indicators_matching", language,
                                                      query=indicator_type_hint)}
        return _finish(payload, language, session_state, citations,
                        question=user_message)

    if ctype in ("scope_performance", "scope_direction"):
        # The group can come from this message ("best performing education
        # indicators", "which national indicators are rising") or, far more
        # often, from the turn that just listed them.
        kind, scope = _match_catalog_scope(user_message)
        if not kind and session_state.get("last_catalog_scope"):
            kind = session_state["last_catalog_scope"]["kind"]
            scope = session_state["last_catalog_scope"]["scope"]
        # No group of indicators in play, but the previous turn ranked a set of
        # COUNTRIES on one indicator. "Which is the best performing one?" then
        # means best on that indicator, and the answer is already sitting in the
        # previous turn — asking it again as a fresh question is how this came
        # back "No indicator matches 'most preformed one'".
        # Only for a performance ranking. "Which are increasing and which are
        # declining" is a question about a group of indicators; redirecting it
        # onto one indicator's countries would answer something else entirely.
        if not kind and ctype == "scope_performance" and session_state.get("last_indicator_name"):
            ctype = "country_ranking"
            intent["indicator_phrase"] = (intent.get("indicator_phrase")
                                           or session_state["last_indicator_name"])
            # Which end of the ranking is "best" is not the user's to state and
            # not ours to assume — polarity_en says it, and for Inflation it
            # says Decrease, so best performing is the LOWEST. Resolved below,
            # once the indicator is.
            performance_followup = True
        elif not kind:
            payload = {"ok": False, "message": msg("performance_needs_scope", language)}
            return _finish(payload, language, session_state, [], question=user_message)

    if ctype in ("scope_performance", "scope_direction"):
        session_state["last_catalog_scope"] = {"kind": kind, "scope": scope}
        entries = retriever.get_scope_performance(kind, scope)
        # Each row is a different indicator with its own unit and precision, so
        # the Format column has to be applied per row rather than once for the
        # answer. Without this the ratio indicators print as "9.0 NA".
        for e in entries:
            e["unit_en"] = display_unit(e.get("unit_en"), e.get("format"))
        result = (compute.scope_direction(entries) if ctype == "scope_direction"
                  else compute.scope_performance(
                      entries, best_first=not _asks_for_worst(user_message)))
        payload = {"ok": result.ok, "facts": {**result.facts, "scope": scope, "scope_kind": kind}} \
            if result.ok else {"ok": False, "message": result.message}
        citations = [Citation(indicator=e.get("indicator"), data_source=None,
                               table="published_data_points", record_id=e.get("record_id"),
                               period_label=e.get("period_label"))
                     for e in entries if e.get("record_id")]
        return _finish(payload, language, session_state, citations, question=user_message)

    if ctype == "article_lookup":
        # Articles are the fallback for what the catalogue cannot answer, not a
        # parallel route to the same subjects. "هل هناك شركات تكنولوجيا مالية
        # جديدة تم افتتاحها في عام 2025؟" was answered from article prose —
        # correctly noting the excerpts did not cover it — while "Number of
        # licensed FinTech & InsurTech players" sat in the catalogue with the
        # figures that answer it outright.
        #
        # Unless the user explicitly asked what SCAI WROTE, a topic that
        # resolves confidently to an indicator with data is answered from the
        # data. "What has SCAI written about inflation?" still goes to the
        # articles, even though Inflation resolves perfectly, because that
        # question is about the writing.
        topic = intent.get("indicator_phrase") or user_message
        if not _explicitly_asks_for_writing(user_message):
            probe = resolve_indicator(topic, require_data=True, language=language)
            if (probe.status == "resolved" and probe.match
                    and probe.match.confidence >= CONFIDENT_MATCH):
                ctype = "latest_value"
                intent["indicator_phrase"] = topic
        if ctype == "article_lookup":
            return _article_answer(user_message, intent, language, session_state)

    if ctype == "macro_overview":
        payload, citations = _macro_overview(language)
        if payload.get("ok") and wants_chart(user_message, ctype):
            payload["chart"] = build_chart_spec(ctype, payload["facts"], "Economic Overview", None, None)
        return _finish(payload, language, session_state, citations,
                        question=user_message)

    # --- everything below needs an indicator resolved first ---
    indicator_phrase = intent.get("indicator_phrase")
    if not indicator_phrase and intent.get("is_followup") and session_state.get("last_indicator_name"):
        indicator_phrase = session_state["last_indicator_name"]
    if not indicator_phrase and session_state.get("last_metrics")             and not has_identifying_content(user_message):
        # The previous answer covered several metrics and this message names
        # none of its own — so it is asking about the same set, with something
        # changed. Rejoined as a list so the normal split path picks it up and
        # re-runs the snapshot against the new period.
        indicator_phrase = ", ".join(session_state["last_metrics"])
    if not indicator_phrase:
        # The intent agent dropped the field. It happens — that prompt now
        # carries seventeen computation types and fifteen rules — and when it
        # did, "What was inflation in May 2025?" was refused with "I couldn't
        # tell which indicator you're asking about", about a question that
        # names one plainly. The question itself is always available, and the
        # phrase normalizer strips the parts that are not the indicator, so
        # there is no reason to refuse without trying it.
        indicator_phrase = user_message

    # A definition question is answered from catalog text, so a stub with no
    # data points is still a legitimate match. Every other path needs figures.
    # Answering the previous turn's "which one did you mean?" by naming a
    # candidate is not a follow-up that inherits the old phrase — it IS the
    # indicator. Without this, clicking a chip re-sent the ambiguous phrase,
    # which carried forward and produced the identical ambiguity prompt again,
    # leaving the user with no way out of the loop.
    offered = session_state.get("last_ambiguous_candidates") or []
    if offered:
        typed = (user_message or "").strip().lower()
        chosen = next((c for c in offered if c.strip().lower() == typed), None)
        if chosen:
            indicator_phrase = chosen
        session_state.pop("last_ambiguous_candidates", None)

    resolution = resolve_indicator(indicator_phrase or "", require_data=(ctype != "definition"),
                                    language=language)

    # Several metrics in one question get one reading each. Attempted only when
    # the whole phrase did NOT resolve confidently as a single indicator, which
    # is what keeps "Crop Yield - Vegetables, Greenhouses" — a real indicator
    # whose own name contains a comma — from being torn in half. "GDP growth,
    # inflation, and government revenues" does not resolve as one name; it was
    # reported as ambiguous and offered a choice between three indicators, none
    # of which was the question.
    # Split on whether the phrase IS a catalogue name, not on how confidently it
    # resolved. Confidence stopped working the moment the alias table existed:
    # "GDP growth, inflation, and government revenues" contains "gdp", the alias
    # offered "Real GDP" as a query form, that scored near 1.0, and a
    # three-metric question was answered about one indicator with the other two
    # reported as unavailable — while asking for each separately worked.
    # Resolved BEFORE the multi-metric split, not after: the snapshot needs it.
    # "Show GDP growth, inflation, and government revenues for the year 2023"
    # returned each indicator's latest reading — 2025-Q4, 2026-04, 2026-Q1 —
    # because the period was parsed forty lines further down, long after the
    # snapshot had already answered. Asking for any one of the three on its own
    # honoured 2023 correctly, which is what made the omission visible.
    period = parse_period_expression(intent.get("period_expression"))
    if period.kind == "unspecified":
        # The model dropped the period. Read it from the question instead.
        #
        # This produced the worst failure yet: "What was inflation in May
        # 2025?" came back "There is no approved data for inflation in May
        # 2025" alongside April 2026's figure, while the identical question
        # without its opening words returned 0.08365% for May 2025. The data
        # was there the whole time; the filter was simply never applied, so the
        # series ran to its end and the newest row was reported.
        #
        # Same shape as the indicator_phrase fallback: the question always
        # carries what the model may or may not have extracted, and the parser
        # is deterministic, so there is no reason to depend on the extraction.
        from_message = parse_period_expression(user_message)
        if from_message.kind != "unspecified":
            period = from_message
            session_state["last_period_expression"] = user_message
    if ctype != "definition" and not resembles_catalogue_name(indicator_phrase or ""):
        parts = split_indicator_phrases(indicator_phrase or "")
        if len(parts) > 1:
            payload, citations = _indicator_snapshot(parts, language, period)
            if payload.get("ok") and len(payload["facts"]["overview"]) > 1:
                if wants_chart(user_message, "macro_overview"):
                    payload["chart"] = build_chart_spec("macro_overview", payload["facts"],
                                                         "Economic Snapshot", None, None)
                # Remember WHAT was answered about. The single-indicator path
                # records last_indicator_name and this one recorded nothing, so
                # "what about in 2024" after a three-metric answer had no
                # subject to inherit and was refused for naming no indicator —
                # after the system had just listed three.
                session_state["last_metrics"] = [e["indicator"]
                                                  for e in payload["facts"]["overview"]]
                session_state.pop("last_indicator_name", None)
                return _finish(payload, language, session_state, citations,
                        question=user_message)

    if resolution.status != "resolved" and resolution.status != "inactive":
        # "What is the GDP forecast 2026" was refused with "I couldn't find an
        # indicator matching 'GDP forecast'", which blames the wording. The
        # wording is fine; the gap is that this system does not forecast and
        # SCAI publishes none. Say that instead — and point at what the data
        # does hold, since a future TARGET is often what the asker wants.
        if _asks_for_forecast(user_message) and resolution.status == "not_found":
            payload = {"ok": False, "message": msg("no_forecast", language,
                                                    topic=(indicator_phrase or "").strip()
                                                          or user_message.strip())}
            return _finish(payload, language, session_state, [],
                            question=user_message)
        payload = {"ok": False, "message": resolution.message}
        if resolution.status == "ambiguous" and resolution.candidates:
            names = [c.name_en.strip() for c in resolution.candidates]
            payload["facts"] = {"candidates": names}
            session_state["last_ambiguous_candidates"] = names
            # Returned verbatim, Composer skipped. Asked to phrase an Arabic
            # refusal, the model TRANSLATED the candidate names — offering
            # "نسبة السمنة" where the catalogue says "Obesity Rate". The chips
            # are answer options: the user picks one and it is sent straight
            # back, so a translated name matches no catalogue row and the
            # choice leads nowhere. The message is already written in both
            # languages; the names inside it must stay exactly as stored.
            return _finish(payload, language, session_state, [],
                            skip_compose=True, canned=resolution.message,
                        question=user_message)
        return _finish(payload, language, session_state, [],
                        question=user_message)

    match = resolution.match
    session_state["last_indicator_detail_id"] = match.indicator_detail_id
    session_state["last_indicator_name"] = match.name_en
    session_state.pop("last_metrics", None)

    # Definitional questions ("what is inflation?", F-030's "what does
    # non-hydrocarbon GDP mean?") are answered from indicator_details.definition_en,
    # which is SCAI's own wording — never from the model's general knowledge, and
    # never by falling through to a latest-value lookup. Only 261 of 644 details
    # carry a real definition, so a missing one is reported honestly rather than
    # filled in.
    # F19 ("ما هو أخر تحليل للتضخم") and F-028: SCAI's own analyst commentary on
    # an indicator, returned VERBATIM. Previously this fell through to a value
    # lookup and answered with a number, which is not what "the latest analysis"
    # asks for. It also catches "why did X fall" — the user wants the written
    # explanation, not a definition and not a figure.
    if ctype == "analysis_lookup":
        rows = retriever.get_series(match.indicator_detail_id,
                                     match.published_detail_id if match.is_published else None,
                                     choose_granularity(parse_explicit_frequency(
                                         intent.get("explicit_frequency")),
                                         retriever.get_available_granularities(
                                             match.published_detail_id if match.is_published else None,
                                             match.indicator_detail_id)) or "yearly")
        with_actuals = [r for r in rows if r.get("actual") is not None]
        recent = list(reversed(with_actuals[-6:]))
        found = retriever.get_analysis_for_data_points([r["record_id"] for r in recent])
        entries = []
        for row in recent:
            analysis = found.get(row["record_id"])
            if not analysis:
                continue
            entry = {"period_label": row["period_label"], "value": row.get("actual")}
            for field, key in (("summary_en", "summary"), ("detailed_analysis_en", "detailed"),
                                ("npc_analysis_en", "npc_analysis"), ("benchmark_en", "benchmark")):
                value = (analysis.get(field) or "").strip()
                if value and value != "-":
                    entry[key] = value
            if len(entry) > 2:
                entries.append(entry)
            if entries:
                break   # the LATEST analysis, not a history of them

        if not entries:
            payload = {"ok": False, "message": msg("no_analysis", language,
                                                    indicator=match.name_en.strip())}
            return _finish(payload, language, session_state, [],
                        question=user_message)

        citations = [Citation(indicator=match.name_en.strip(), data_source=match.data_source_en,
                               table="indicator_analysis", record_id=None,
                               period_label=entries[0]["period_label"], country="Qatar")]
        payload = {"ok": True, "facts": {"indicator": match.name_en.strip(),
                                          "unit": match.unit_en, "analysis": entries}}
        # Verbatim, and skip the Composer entirely: this is the Council's own
        # wording, and a model asked to "phrase" it would paraphrase it.
        body = "\n\n".join(
            f"{match.name_en.strip()} — {e['period_label']}"
            + (f" ({e['value']} {match.unit_en or ''})".rstrip() if e.get("value") is not None else "")
            + "\n" + "\n\n".join(e[k] for k in ("summary", "detailed", "npc_analysis", "benchmark")
                                   if e.get(k))
            for e in entries
        )
        return _finish(payload, language, session_state, citations,
                        skip_compose=True, canned=body,
                        question=user_message)

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
            return _finish(payload, language, session_state, [],
                        question=user_message)
        payload = {"ok": True, "facts": {
            "indicator": match.name_en.strip(),
            "definition": definition,
            "unit": match.unit_en,
        }}
        citations = [Citation(indicator=match.name_en.strip(), data_source=match.data_source_en,
                              table="indicator_details", record_id=match.indicator_detail_id)]
        # Returned verbatim, skipping the Composer, for the same reason as the
        # analyst commentary: this is SCAI's own authored wording, and a model
        # asked to "phrase" it can only paraphrase it. Sending it through the
        # Composer also produced the definition twice — once reworded in the
        # prose and once as the raw fact — and made the model apologise for the
        # absence of figures a definition never has, in one case narrating its
        # own input: "(Note: The payload does not contain specific numerical
        # data points...)".
        body = definition
        if match.unit_en:
            body += msg("measured_in", language, unit=match.unit_en)
        return _finish(payload, language, session_state, citations,
                        skip_compose=True, canned=body,
                        question=user_message)

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
        return _finish(payload, language, session_state, [],
                        question=user_message)

    countries_res = resolve_countries(intent.get("countries_mentioned", []), intent.get("country_group_mentioned"))

    payload, citations = _dispatch_computation(ctype, intent, match, published_detail_id, granularity,
                                                period, countries_res, language,
                                                performance_followup=performance_followup,
                                                user_message=user_message,
                                                session_state=session_state)

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

    return _finish(payload, language, session_state, citations,
                        question=user_message)


def _dispatch_computation(ctype, intent, match, published_detail_id, granularity, period,
                           countries_res, language="en", performance_followup=False,
                           user_message="", session_state=None):
    # Unit AND scale, both of which live in the catalogue: unit_en says "QAR",
    # Format says "bn0.0". Reading only the first printed "185.17 QAR" beside
    # prose that said "QAR 185.2 billion".
    unit = display_unit(match.unit_en, match.format)
    decimals = decimals_from_format(match.format)
    indicator_name = match.name_en
    data_source_en = match.data_source_en

    # Single-series computations queried the Qatar series unconditionally and
    # ignored any country that had been resolved. So a follow-up of "and for
    # Saudi Arabia?" after an inflation question returned QATAR's figure,
    # labelled as the answer — a wrong number presented confidently, which is
    # the worst failure mode this pipeline has.
    #
    # One named country narrows the series to it. More than one is a comparison
    # however the question was phrased, so it is promoted rather than silently
    # answering about only the first.
    named_countries = [c for c in (countries_res.resolved_countries or [])]
    single_country = None
    if ctype not in ("country_comparison", "country_ranking") and named_countries:
        if len(named_countries) > 1 or countries_res.is_group:
            ctype = "country_comparison"
        else:
            single_country = named_countries[0]

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

        # An indicator with no per-country rows at all cannot answer a
        # per-country question in ANY period. Saying "no approved data found
        # for the requested countries" reads as "those countries are missing",
        # when the truth is that this series is only ever reported for Qatar as
        # a whole: "How many GCC tourists arrived into Qatar in 2025?" against
        # "Number of International Visitors", whose 125 points are every one a
        # national total.
        if countries and not retriever.has_country_breakdown(published_detail_id,
                                                              match.indicator_detail_id):
            asked_for = (countries_res.group_name if countries_res.is_group
                         else ", ".join(countries))
            return {"ok": False, "message": msg("no_country_breakdown", language,
                                                 indicator=indicator_name.strip(),
                                                 countries=asked_for)}, []

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
        # "Which performed best?" names no extremum, so the line above would
        # default it to ascending and call that an answer by luck. The
        # indicator decides: Inflation is polarity Decrease, so best is lowest;
        # for an Increase indicator like Real GDP best is highest. Getting this
        # from the data rather than the wording is the whole point — nobody
        # should have to phrase it as "lowest" to be understood.
        if performance_followup:
            decreasing = (match.polarity_en or "").strip().lower().startswith("decrease")
            ascending = decreasing if not _asks_for_worst(user_message) else not decreasing
            note_direction = ("Ranked best-performing first — for {ind}, "
                              "a {dir} figure is the better outcome.").format(
                ind=indicator_name.strip(), dir="lower" if decreasing else "higher")
            note = f"{note} {note_direction}" if note else note_direction
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
            elif period_label and len(entries) > 1:
                # Only when there is actually something to align. "Compare Real
                # GDP across Qatar and Saudi Arabia" has data for Qatar alone,
                # and telling the reader it was "compared at 2025-Q4, the most
                # recent period all of these countries report" described a
                # comparison that did not happen.
                aligned = (f"Compared at {period_label}, the most recent period all of these "
                           f"countries report.")
                note = f"{note} {aligned}" if note else aligned
            for entry in entries:
                country_rows = series_by_country.get(entry["country"], [])
                row = _find_row_by_period(country_rows, entry.get("period_label"))
                if row:
                    citations.append(Citation(indicator_name, data_source_en, row.get("source_table", "unknown"),
                                               row.get("record_id"), row.get("period_label"), entry["country"]))
        return _wrap(result, unit, extra_note=note, indicator_name=indicator_name, decimals=decimals), citations

    rows = retriever.get_series(match.indicator_detail_id, published_detail_id, granularity,
                                 start_date=period.start_date, end_date=period.end_date,
                                 country_en=single_country)
    rows = _apply_last_n_years(rows, period)

    # A named country with no data for this indicator must be said out loud,
    # not answered with Qatar's series as if it were theirs.
    if not rows and single_country:
        return {"ok": False, "message": msg("no_data_for_country", language,
                                             indicator=indicator_name.strip(),
                                             country=single_country)}, []

    # "What is the Real GDP forecast for 2026" was answered "No approved data
    # points were found for this indicator/period" — true, but it leaves the
    # user unable to tell whether the indicator is missing, the period is, or
    # they phrased it wrong. If the indicator has data outside the window they
    # asked for, say what IS there. Costs one extra query, only on the
    # empty-result path.
    if not rows and (period.start_date or period.end_date):
        available = retriever.get_series(match.indicator_detail_id, published_detail_id, granularity)
        if available:
            message = msg("no_data_for_period", language,
                           indicator=indicator_name.strip(), granularity=granularity,
                           period=period.raw or "that period",
                           first=available[0]["period_label"],
                           last=available[-1]["period_label"])
            # One limitation, one explanation, however the question was worded.
            latest_dated = max((r["period_date"] for r in available if r.get("period_date")),
                                default=None)
            asked_start = period.start_date
            if (latest_dated and asked_start and asked_start > latest_dated) or                _asks_for_forecast(user_message):
                message += msg("future_period_suffix", language)
            return {"ok": False, "message": message}, []

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
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_ranking":
        # extremum "min" means lowest-first; anything else (including the usual
        # "highest to lowest" phrasing, and the null default) means highest-first.
        descending = intent.get("extremum") != "min"
        result = compute.period_ranking(rows, descending=descending)
        # Cite every row that appears in the ranking, not just the winner —
        # each listed figure is a claim of its own.
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "trend":
        result = compute.trend(rows)
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "min_max":
        which = intent.get("extremum") or "max"
        result = compute.min_max(rows, which)
        used = [_find_row_by_period(rows, result.facts.get("period_label"))] if result.ok else []
        used = [r for r in used if r]
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "difference":
        result = compute.difference(rows)
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("high_period")),
                                 _find_row_by_period(rows, result.facts.get("low_period"))) if r]
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

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
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_comparison":
        # Use the two periods the user actually named. Previously this took the
        # first and last rows of whatever had been retrieved, but the retrieval
        # had already been narrowed to ONE period by the single-quarter branch
        # of the period parser — so "between Q1 2025 and Q4 2025" returned one
        # row and the answer was "could not find two distinct periods".
        # "...the same quarter of 2025" writes its second period as a bare
        # year, borrowing the quarter from the first. Checked alongside the
        # explicit pair because parse_period_pair reads it literally, as a
        # quarter next to a year, and declines the mixed granularity.
        pair = (parse_period_pair(period.raw)
                or parse_same_period_pair(period.raw or user_message))
        # A comparison stated relative to now — "this year and the year before
        # it", "compared with the previous year" — names no date for the token
        # scanner to find, so it was answered "I need two specific periods to
        # compare", asking the user to restate in the system's vocabulary
        # something they had already said clearly.
        relative = None if pair else parse_relative_pair(period.raw or user_message)
        if relative:
            rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                         granularity, country_en=single_country)
            actuals = [r for r in rows if r.get("actual") is not None]
            if len(actuals) >= 2:
                if relative == "prev_period":
                    # The reading immediately before the latest, whatever the
                    # series' own spacing is.
                    label_b, label_a = actuals[-1]["period_label"], actuals[-2]["period_label"]
                else:
                    # Anchored on the period the user named, when they named
                    # one. "Real GDP in Q4 2025, and what was it one year
                    # earlier" is a question about Q4 2025 and Q4 2024 — using
                    # the series' latest reading instead would answer about a
                    # quarter they did not mention.
                    anchor = (single_period_label(period.raw or user_message)
                              or actuals[-1]["period_label"])
                    if relative == "prev_two_years":
                        anchor = year_earlier_label(anchor)
                    label_b, label_a = anchor, year_earlier_label(anchor)
                pair = (label_a, label_b)
        # A follow-up that names ONE period — "compare it with Q4 2025" — means
        # that period against the one the previous turn reported. Both are known;
        # asking the user to restate a period they have already been shown is
        # the system losing its place in the conversation, not a real ambiguity.
        if not pair:
            named = single_period_label(period.raw or user_message)
            prior = (session_state or {}).get("last_period_used")
            if (named and prior and prior != named
                    and period_kind(named) == period_kind(prior)):
                rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                             granularity, country_en=single_country)
                pair = tuple(sorted((prior, named)))
                relative = "carried_forward"

        # Last resort: the intent agent's canonical labels, for a phrasing the
        # deterministic parser does not recognise. Never used as given — a
        # wrong period is the one error nothing downstream catches, because a
        # figure for the wrong quarter still traces to the data and passes the
        # numeric verifier. validate_period_labels requires it to be well
        # formed, like-for-like, and ACTUALLY PRESENT in this indicator's
        # series; anything else falls through to the refusal below, which the
        # user can see and correct.
        if not pair and intent.get("period_labels"):
            all_rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                             granularity, country_en=single_country)
            available = {r["period_label"] for r in all_rows if r.get("actual") is not None}
            checked = validate_period_labels(intent["period_labels"], available=available)
            if checked:
                rows, pair = all_rows, tuple(sorted(checked))
                relative = "model_labels"

        if pair:
            label_a, label_b = pair
            if not relative:
                # Re-fetch unfiltered: both named periods have to be present,
                # and the earlier query was scoped to just one of them.
                rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                             granularity, country_en=single_country)
        elif getattr(period, "kind", None) == "range" and len(rows) >= 2:
            # Only for an explicit span ("over 2024", "from 2022 to 2025"),
            # where the ends of the window ARE the two periods meant.
            #
            # This used to run for any retrieval holding two or more rows, so a
            # question whose periods had not been understood was answered from
            # whatever window happened to be in hand rather than refused. "Real
            # GDP in Q4 2025, and what was it one year earlier" was narrowed to
            # 2024 by the phrase "year earlier" and then answered "2024-Q1 vs
            # 2024-Q4" — a confident comparison between two quarters the user
            # never named. A refusal is recoverable; that is not.
            label_a, label_b = rows[0]["period_label"], rows[-1]["period_label"]
        else:
            return {"ok": False, "message": msg("need_two_periods", language)}, []

        result = compute.period_to_period_change(rows, label_a, label_b)
        used = []
        if result.ok:
            used = [r for r in (_find_row_by_period(rows, result.facts.get("period_a")),
                                 _find_row_by_period(rows, result.facts.get("period_b"))) if r]
            # Say WHICH two periods were compared, whenever the question did not
            # spell them out. A wrong period reads exactly like a right one —
            # the figure is real either way — so the only defence a reader has
            # is seeing what was actually compared. This is what would have made
            # "2024-Q1 vs 2024-Q4" obvious the moment it appeared.
            if relative:
                result.facts["periods_interpreted_as"] = f"{label_a} and {label_b}"
            # Where the deterministic parser and the model disagree, record it.
            # The parser wins — that ordering is the whole design — but it can
            # be confidently wrong on a phrasing it only partly recognises:
            # "الربع الافتتاحي من 2026 مقابل ما يقابله في 2025" scans as the
            # bare years 2026 and 2025, losing the quarter, and never reaches
            # the fallback. Flagged rather than acted on, so the size of that
            # gap can be measured before anything is changed on a hunch. The
            # leading underscore keeps it out of the Composer's payload.
            model_pair = validate_period_labels(intent.get("period_labels"))
            if model_pair and tuple(sorted(model_pair)) != tuple(sorted((label_a, label_b))):
                result.facts["_period_disagreement"] = {
                    "resolved": [label_a, label_b],
                    "model_said": list(model_pair),
                    "source": relative or "deterministic",
                }
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

    # Anything that reached here named an indicator, resolved it, and has rows.
    # Refusing at this point blames the user for the routing taxonomy — and
    # "I couldn't determine what computation this question needs" is internal
    # vocabulary that means nothing to a reader.
    #
    # It was reachable: "multi_indicator" is a real computation_type that is
    # handled only when the phrase splits into two or more metrics that both
    # resolve. "What was inflation in May 2025?" was classified as one, did not
    # split, and fell through — while the same question without the opening
    # words was answered fine. The latest value is the sensible default for a
    # question about an indicator and a period.
    result = compute.latest_value(rows)
    used = []
    if result.ok:
        row = _find_row_by_period(rows, result.facts.get("period_label"))
        used = [row] if row else []
    return (_wrap(result, unit, indicator_name=indicator_name, decimals=decimals),
            citations_for_rows(used, indicator_name, data_source_en))


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
    if period.kind not in ("last_n_years", "previous_year") or not period.n_years:
        return rows
    dated = [r for r in rows if r.get("period_date")]
    if not dated:
        return rows
    anchor = max(r["period_date"] for r in dated)

    if period.kind == "previous_year":
        # "the last year" means the year BEFORE the most recent data, not a
        # window ending at it. Treated as a window, the answer would be the
        # same latest reading the user had just been given.
        target = anchor.year - period.n_years
        in_year = [r for r in dated if r["period_date"].year == target]
        # If that year has no readings, keep everything rather than answer with
        # nothing — the caller reports an empty result honestly either way.
        return in_year or dated

    cutoff = date(anchor.year - period.n_years, anchor.month, 1)
    return [r for r in dated if r["period_date"] >= cutoff]


_ASKS_FOR_WRITING = re.compile(
    r"\b(wrote|written|writing|article|articles|paper|papers|publication|"
    r"published\s+(a|an|any)?\s*(article|piece|paper)|view|views|opinion|stance|"
    r"position|commentary|say\s+about|said\s+about|think\s+about)\b"
    r"|كتب|مقال|مقالات|رأي|وجهة نظر|موقف|تحليل المجلس",
    re.IGNORECASE,
)


_ASKS_FOR_FORECAST = re.compile(
    r"\b(forecast|forecasts?ed|projection|projected|predict(ion|ed)?|outlook|"
    r"expected\s+(to|value|level)|will\s+be|going\s+to\s+be)\b"
    r"|توقع|توقعات|تنبؤ|إسقاط|المتوقع",
    re.IGNORECASE,
)


def _asks_for_forecast(text: str) -> bool:
    return bool(_ASKS_FOR_FORECAST.search(text or ""))


def _explicitly_asks_for_writing(text: str) -> bool:
    """True when the question is about SCAI's published WRITING, not a number.

    The distinction decides whether the catalogue gets first refusal. Without
    it, preferring data would hijack "what has SCAI written about inflation?",
    which resolves to a real indicator but is plainly a question about an
    article.
    """
    return bool(_ASKS_FOR_WRITING.search(text or ""))


def _best_article_passages(passages: list[dict]) -> list[dict]:
    """Keeps the passages from the best-matching ARTICLE, not the best-matching
    chunks across the whole corpus.

    Chunk-level top-k was the reason English article questions failed. "The
    trade war" returned six passages that all cleared the distance threshold
    but came from five different articles, so the answering model was handed
    one relevant excerpt buried in five unrelated ones and concluded the
    articles didn't cover the topic. The Arabic answer showed the same effect
    from the other side: it found the right article and then noted that the
    others were about Strait of Hormuz tolls.

    A question about a topic is nearly always answered by ONE article, so the
    corpus is scored by article — an article's score is its best chunk — and
    the answer is grounded in the best article's passages. A second article is
    included only if it is genuinely close to the first, since two articles on
    the same theme is a real case and two on different themes is noise.
    """
    if not passages:
        return []

    by_article: dict[str, list[dict]] = {}
    for p in passages:
        by_article.setdefault(p["article_id"], []).append(p)

    def article_score(chunks: list[dict]) -> float:
        return min(float(c["distance"]) for c in chunks)

    ranked = sorted(by_article.values(), key=article_score)
    best = ranked[0]
    if article_score(best) > ARTICLE_MAX_DISTANCE and all(c["match"] == "semantic" for c in best):
        return []

    chosen = list(best)
    if len(ranked) > 1 and article_score(ranked[1]) <= article_score(best) + ARTICLE_SECOND_MARGIN:
        chosen += ranked[1]
    chosen.sort(key=lambda c: (float(c["distance"]), c["chunk_index"]))
    return chosen[:ARTICLE_PASSAGES]


def _article_answer(user_message: str, intent: dict, language: str, session_state: dict) -> dict:
    """Answers from SCAI's published articles instead of the indicator tables.

    The retrieval is semantic, so the user's wording need not match the
    article's — asking about "trade wars" finds a piece titled "Financial
    Stability Implications of Tariffs". That is the capability; the risk that
    comes with it is that a vector search ALWAYS returns a nearest neighbour,
    however unrelated, so relevance has to be checked rather than assumed.
    Hence the distance threshold: past it, the honest answer is that the
    articles do not cover this.
    """
    topic = (intent.get("indicator_phrase") or user_message or "").strip()
    if not topic:
        return _finish({"ok": False, "message": msg("no_article_topic", language)},
                        language, session_state, [],
                        question=user_message)

    if retriever.count_article_chunks() == 0:
        return _finish({"ok": False, "message": msg("articles_not_indexed", language)},
                        language, session_state, [],
                        question=user_message)

    query_embedding = get_embedding(topic)
    passages = retriever.search_article_chunks(query_embedding, language=language,
                                                limit=ARTICLE_SEARCH_POOL, query_text=topic)
    # Articles exist in both languages; if the question's language has no
    # indexed text, fall back rather than claim nothing was written.
    if not passages:
        other = "ar" if language == "en" else "en"
        passages = retriever.search_article_chunks(query_embedding, language=other,
                                                    limit=ARTICLE_SEARCH_POOL, query_text=topic)

    relevant = _best_article_passages(passages)
    if not relevant:
        nearest = f"{float(passages[0]['distance']):.3f}" if passages else "n/a"
        payload = {"ok": False, "message": msg("no_articles_found", language, topic=topic),
                   "facts": {"nearest_distance": nearest}}
        return _finish(payload, language, session_state, [],
                        question=user_message)

    answer = answer_from_articles(user_message, relevant, language)

    # The numeric verifier still applies, against the passages instead of a
    # facts payload: any figure in the answer must appear in the text it was
    # drawn from. It cannot check whether a CLAIM is faithful — that is what
    # the quote-and-cite prompt and the returned passages are for — but a
    # fabricated statistic is the failure that matters most here, and this
    # catches it.
    grounding = {"passages": [p["content"] for p in relevant]}
    clean, offending = verify_numbers(answer, grounding)
    if not clean:
        answer = (answer + "\n\n" + msg("ungrounded_numbers", language,
                                         numbers=", ".join(offending)))

    citations = [
        Citation(indicator=(p.get("title_en") or p.get("title_ar") or "").strip() or None,
                 data_source="SCAI", table="articles", record_id=p["article_id"],
                 period_label=str(p.get("published_date") or p.get("article_date") or "") or None,
                 country=None)
        for p in relevant
    ]
    # One citation per article, not per passage — several chunks of the same
    # piece are one source.
    seen, unique = set(), []
    for c in citations:
        if c.record_id not in seen:
            seen.add(c.record_id)
            unique.append(c)

    payload = {
        "ok": True,
        "facts": {
            "topic": topic,
            "passages": [
                {"article_title": (p.get("title_en") or p.get("title_ar") or "").strip(),
                 "article_id": p["article_id"],
                 "excerpt": p["content"],
                 "match": p["match"],
                 "distance": round(float(p["distance"]), 4)}
                for p in relevant
            ],
            "passage_count": len(relevant),
            "articles_considered": len({p["article_id"] for p in passages}),
            "best_distance": round(float(passages[0]["distance"]), 4) if passages else None,
        },
        # Same separation as /read: the source text stays distinguishable from
        # the generated summary of it.
        "answer_is_generated_from_articles": True,
    }
    if not clean:
        payload["_verifier_rejected_numbers"] = offending
    session_state["last_article_topic"] = topic
    return _finish(payload, language, session_state, unique,
                    skip_compose=True, canned=answer,
                        question=user_message)


_CATALOG_STOPWORDS = {
    "how", "many", "much", "list", "give", "me", "all", "the", "a", "an", "in", "of",
    "for", "show", "what", "are", "there", "is", "please", "and", "names", "name",
    "indicator", "indicators", "under", "within", "tell", "about", "which",
}


_CATALOG_REQUEST = re.compile(
    r"\b(list|how many|give me all|show all|show me all|name all|what are the|"
    r"names? of|all the)\b", re.IGNORECASE)


def _looks_like_catalog_request(text: str) -> bool:
    """A list/count request that names a real sector or indicator type.

    Both halves are required. "List the indicators in Sectors" qualifies;
    "list Qatar's quarterly GDP values" does not, because "GDP values" matches
    no sector or type — that one is a period ranking and must stay one.
    """
    if not text or not _CATALOG_REQUEST.search(text):
        return False
    if not re.search(r"\bindicator", text, re.IGNORECASE):
        return False
    kind, _ = _match_catalog_scope(text)
    return kind is not None


# "most preformed" is in here on purpose. This is a bilingual audience typing
# into a chat box; a spelling that a reader understands instantly should not
# decide whether the question gets answered.
_PERFORMANCE_RANKING = re.compile(
    r"\b(best|worst|top|strongest|weakest|highest|lowest|most|least)[\s-]*"
    r"(performing|performer|performed|preforming|preformed|performance)\b"
    r"|\b(performance|performing)\s+(ranking|ranked|comparison)\b"
    r"|\bwhich\s+(ones?|indicators?)\s+(are\s+)?(doing|performing|preforming)\b"
    r"|\brank\s+(them|these|the indicators)\b"
    r"|\b(on|against)\s+track\b"
    r"|أفضل\s*أداء|أسوأ\s*أداء|الأفضل\s*أداء|أداءً",
    re.IGNORECASE)

_WORST_FIRST = re.compile(
    r"\b(worst|weakest|lowest|least|behind|lagging|underperform\w*)\b|أسوأ|متأخر",
    re.IGNORECASE)


def _asks_for_performance_ranking(text: str) -> bool:
    """A request to rank a GROUP of indicators by how they are doing.

    Distinct from period_ranking, which orders one indicator's own readings
    over time. This orders different indicators against their own targets, and
    it only means anything when a group is in play — which is why the handler
    requires a sector or type, from this message or from the previous turn.
    """
    return bool(text and _PERFORMANCE_RANKING.search(text))


def _asks_for_worst(text: str) -> bool:
    return bool(text and _WORST_FIRST.search(text))


# A group split by direction of travel: which are up, which are down. Both
# halves must appear, so "which indicators are increasing?" alone still counts
# but "is inflation increasing?" — one named indicator — does not.
_DIRECTION_UP = re.compile(
    r"\b(increas\w*|rising|rise|risen|up|grow\w*|improv\w*|higher)\b|ارتفع\w*|ترتفع|تزايد|زيادة",
    re.IGNORECASE)
_DIRECTION_DOWN = re.compile(
    r"\b(declin\w*|decreas\w*|falling|fall\w*|down|drop\w*|shrink\w*|worsen\w*|lower)\b"
    r"|انخفض\w*|تنخفض|تراجع|هبوط",
    re.IGNORECASE)
_DIRECTION_PLURAL = re.compile(
    r"\bindicators\b|\bmetrics\b|\bones\b|\bwhich\s+are\b|المؤشرات", re.IGNORECASE)


def _asks_for_direction_split(text: str) -> bool:
    """"Which of these are increasing and which are declining?"

    Distinct from a two-period comparison of ONE indicator, which is what this
    was being read as — it asked the user to name two periods, then sent the
    whole sentence to the indicator resolver once they did. It is one question
    about a whole group, and each member's year-on-year change answers it.

    Both directions are required. A question with only one ("which indicators
    are rising?") still qualifies via the plural, but "is inflation rising?"
    names a single indicator and must stay a single-indicator question.
    """
    if not text:
        return False
    if not (_DIRECTION_UP.search(text) and _DIRECTION_DOWN.search(text)):
        return False
    return bool(_DIRECTION_PLURAL.search(text))


# The economy as a whole, rather than any one indicator in it. Deliberately
# narrow: it requires the word itself, so "how is the education sector doing"
# and "how are prices doing" keep their own routes.
# The Arabic stem is matched bare, with no word boundary and no definite
# article: \b does not behave on Arabic script, and requiring "ال" missed
# "هل ينمو اقتصاد قطر؟" while matching "الاقتصاد القطري". The stem covers both,
# plus the adjective اقتصادي.
_ECONOMY_WORD = re.compile(r"\becon(omy|omic)\b|اقتصاد", re.IGNORECASE)
_ECONOMY_STATE = re.compile(
    r"\b(doing|going|growing|grow|growth|shrink\w*|contract\w*|expand\w*|"
    r"perform\w*|preform\w*|health\w*|healthy|strong|weak|state|situation|"
    r"outlook|condition|overview|snapshot|shape|fine|ok|okay|well|recover\w*)\b"
    r"|\bhow\s+is\b|\bhow'?s\b"
    r"|ينمو|نمو|كيف|وضع|حال|أداء",
    re.IGNORECASE)


def _asks_about_the_economy(text: str) -> bool:
    """A question about the economy overall, which is the macro snapshot.

    "How is Qatar's economy doing right now?" already worked. "Is Qatar's
    economy growing?" did not — it went to indicator resolution and was told to
    "name the metric more plainly", which asks the user to know the catalogue
    before they are allowed to ask the most natural question there is. Both
    halves are required so that a question naming a real indicator still goes
    to that indicator: "what is economic growth" is one word away, and belongs
    with Real GDP.
    """
    if not text or not _ECONOMY_WORD.search(text):
        return False
    if not _ECONOMY_STATE.search(text):
        return False
    # A question that also names a sector or an indicator type is about that
    # group, not about the economy at large.
    kind, _ = _match_catalog_scope(text)
    return kind is None


def _match_catalog_scope(phrase: str):
    """Maps a question fragment onto a real indicator type or sector name.

    Returns ("type"|"sector", exact_name) or (None, None). Scored on the share
    of the question's own words that the candidate contains, so "education
    sector" prefers the sector "Education Sector" (both words) over the type
    "Sector Indicator" (one word), while a bare "sectors" prefers the type,
    which is the broader grouping.
    """
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", (phrase or "").lower())
             if w not in _CATALOG_STOPWORDS and len(w) > 2}
    if not words:
        return None, None

    def score(candidate: str) -> float:
        cand = {w.rstrip("s") for w in re.findall(r"[a-z]+", candidate.lower())}
        return len(words & cand) / len(words)

    best_type = max(((t, score(t)) for t in retriever.list_indicator_types()),
                    key=lambda x: x[1], default=(None, 0.0))
    best_sector = max(((s, score(s)) for s in retriever.list_sector_names()),
                      key=lambda x: x[1], default=(None, 0.0))
    # Types win ties: a generic word like "sectors" means the grouping, not one
    # particular sector that happens to contain the word.
    if best_type[1] >= best_sector[1] and best_type[1] > 0:
        return "type", best_type[0]
    if best_sector[1] > 0:
        return "sector", best_sector[0]
    return None, None


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
          indicator_name: Optional[str] = None, decimals: Optional[int] = None) -> dict:
    payload = {"ok": result.ok, "facts": {**result.facts, "unit": unit}} if result.ok else \
              {"ok": False, "message": result.message}
    # Name the indicator in every successful payload. Without it the Composer
    # has nothing to name and writes "the specified economic indicator", which
    # hides a mis-resolution: a question about solar energy was answered with a
    # "Share of Qatari Teachers" target and read as plausible, because the
    # answer text never said which indicator it used.
    if payload.get("ok") and indicator_name:
        payload["facts"]["indicator"] = indicator_name.strip()
    # How many decimals SCAI specifies for this indicator, so the frontend and
    # the Composer round the same way instead of each choosing.
    if payload.get("ok") and decimals is not None:
        payload["facts"]["decimal_places"] = decimals
    if extra_note and payload.get("ok"):
        payload["facts"]["note"] = extra_note
    return payload


_SPLIT_METRICS = re.compile(r"\s*(?:,|;|\band\b|\bو\s)\s*", re.IGNORECASE)


def split_indicator_phrases(phrase: str) -> list[str]:
    """Splits "GDP growth, inflation, and government revenues" into its parts.

    One question naming three metrics was sent to the resolver as a single
    string, which then reported — correctly, for what it was given — that
    "GDP growth, inflation, and government revenues" could match more than one
    indicator, and offered a choice of three unrelated ones. Nothing in the
    pipeline had noticed it was three questions.
    """
    parts = [p.strip(" .?!") for p in _SPLIT_METRICS.split(phrase or "")]
    return [p for p in parts if len(p) > 2]


def _asks_for_growth(phrase: str) -> bool:
    return bool(re.search(r"\b(growth|change|rate of change)\b|نمو|تغير", phrase or "",
                          re.IGNORECASE))


def _indicator_snapshot(phrases: list[str], language: str = "en", period=None):
    """One latest reading per named indicator, each with its OWN period.

    Different indicators report on different schedules, so a snapshot that
    forces them onto one period would either drop the fresher ones or imply a
    currency the data does not have. Each line carries the period it came from,
    and nothing historical is added — a comparison the user did not ask for is
    a comparison the user did not ask for.
    """
    facts, citations, missing = [], [], []
    for phrase in phrases:
        res = resolve_indicator(phrase, language=language)
        if res.status not in ("resolved", "inactive") or not res.match:
            missing.append(phrase)
            continue
        match = res.match
        published_id = match.published_detail_id if match.is_published else None
        gran = choose_granularity(None, retriever.get_available_granularities(
            published_id, match.indicator_detail_id))
        if not gran:
            missing.append(phrase)
            continue
        # Honour the period the question asked for. Each indicator is then read
        # at ITS latest reading WITHIN that window, which is why a 2023 question
        # yields 2023-Q4 for a quarterly series and 2023-12 for a monthly one.
        rows = retriever.get_series(match.indicator_detail_id, published_id, gran,
                                     start_date=getattr(period, "start_date", None),
                                     end_date=getattr(period, "end_date", None))
        if period is not None:
            rows = _apply_last_n_years(rows, period)
        latest = compute.latest_value(rows)
        if not latest.ok:
            # Named, resolved, but nothing in the window asked for — distinct
            # from "no such indicator", and reported separately below.
            missing.append(phrase)
            continue
        row = _find_row_by_period(rows, latest.facts.get("period_label"))
        entry = {"indicator": match.name_en.strip(),
                 "unit": display_unit(match.unit_en, match.format),
                 "decimal_places": decimals_from_format(match.format),
                 "granularity": gran, "asked_as": phrase, **latest.facts}
        # SCAI's own vetted period-on-period change, never a fresh derivation
        # (F-022..F-026). Carried on every line so "GDP growth" can be answered
        # as a growth rate rather than a level.
        if row:
            # The reading a year earlier, so the answer can show what the
            # figure moved FROM. A percentage on its own ("Real GDP grew 2.1%")
            # is a claim the reader has to take on trust; the pair of values is
            # the evidence for it, and it is what "show me before and after"
            # asks for.
            prior = _find_row_by_period(rows, year_earlier_label(
                latest.facts.get("period_label")))
            if prior and prior.get("actual") is not None:
                entry["previous_period"] = prior["period_label"]
                entry["previous_value"] = prior["actual"]
                citations += citations_for_rows([prior], match.name_en, match.data_source_en)
            change = compute.preferred_change_field(row, gran)
            if change is not None:
                # SCAI's stored figure carries full float noise
                # (2.0276599261667307). Rounded to the same 4 places every other
                # computed value uses, so the number shown is the number checked.
                entry["change_yoy_percent"] = round(float(change), 4)
            entry["report_as_growth"] = _asks_for_growth(phrase)
            citations += citations_for_rows([row], match.name_en, match.data_source_en)
        facts.append(entry)

    if not facts:
        # A period was asked for and nothing falls inside it — that is a
        # different answer from "no headline indicators exist", which is what
        # the overview message says.
        asked_period = getattr(period, "raw", None)
        if asked_period:
            return {"ok": False, "message": msg("none_in_period", language,
                                                 period=asked_period)}, []
        return {"ok": False, "message": msg("no_overview", language)}, []
    payload = {"ok": True, "facts": {"overview": facts}}
    if missing:
        # Named but not found — said out loud rather than quietly dropped from
        # a list the user can see is short.
        payload["facts"]["not_found"] = missing
    return payload, citations


# The standard macro snapshot. An editorial list — refine it once SCAI says
# which indicators they want surfaced as the headline set.
HEADLINE_NAMES = ["Real GDP", "Inflation", "Trade Balance", "Government Revenues"]


def _macro_overview(language="en"):
    """Fixes F-007/F-018: a deterministic curated overview instead of silently
    falling back to a single indicator.

    Delegates to _indicator_snapshot rather than repeating it. This function
    used to be its own older copy of the same loop, and had drifted: it read
    unit_en raw, so Real GDP printed as "185.17 QAR" instead of "185.17 billion QAR"
    while the prose beside it said "QAR 185.2 billion"; and it carried no
    period-on-period change at all, so "how is the economy doing right now?"
    could only list four levels and call that an answer.
    """
    payload, citations = _indicator_snapshot(HEADLINE_NAMES, language=language)
    if payload.get("ok"):
        # Tells the Composer this is the "how is the economy doing" question
        # rather than an arbitrary set of metrics, so it reports the movement
        # rather than reading out four numbers.
        payload["facts"]["overview_kind"] = "macro"
        # Named indicators that failed to resolve are an internal problem with
        # the editorial list above, not something to report to a user who never
        # named them.
        payload["facts"].pop("not_found", None)
    return payload, citations


# facts keys that represent an actual READING — a value measured at a period.
# A "read this for me" view exists to present readings; offering it on a
# greeting, a refusal, a definition or a catalogue listing gives the user a
# button that reveals nothing they were not already shown.
_READABLE_FACT_KEYS = {"actual", "series", "ranked", "rows", "ranked_periods",
                       "overview", "high_value", "value_a", "value_start",
                       "ranked_indicators", "increasing"}


def is_readable(payload: dict) -> bool:
    if not payload.get("ok"):
        return False
    return bool(_READABLE_FACT_KEYS & set((payload.get("facts") or {}).keys()))


def _finish(payload: dict, language: str, session_state: dict, citations: list[Citation],
            skip_compose: bool = False, canned: Optional[str] = None,
            question: str = "") -> dict:
    # The period this answer actually reported, so "compare it with Q4 2025"
    # has the other half of its comparison. Recorded from the facts rather than
    # from the question, because the question may have named no period at all
    # and still been answered at a specific one.
    facts_out = payload.get("facts") or {}
    for key in ("period_label", "period_b", "period_end", "last_period", "period_used"):
        if facts_out.get(key):
            session_state["last_period_used"] = facts_out[key]
            break

    if skip_compose:
        answer = canned or ""
    else:
        draft = compose_answer(payload, language, question=question)
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
        "readable": is_readable(payload),
        "session_state": session_state,
    }
