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
from typing import TypedDict, Optional, Callable

from app.nlu.intent_agent import extract_intent
from app.core.conversation import carry_forward, remember, forget_stale, turn_index
from app.core.messages import (msg, detect_language, answers_in_language,
                                carries_language_signal, is_arabic,
                                language_request_in_text)
from app.resolvers.indicator_resolver import (resolve_indicator, has_usable_definition,
                                               ResolutionResult,
                                               resembles_catalogue_name, has_identifying_content,
                                               names_nothing_in_catalogue)
from app.resolvers.country_resolver import resolve_countries, display_country
from app.resolvers.period_resolver import (parse_period_expression, parse_explicit_frequency,
                                            choose_granularity, parse_period_pair,
                                            parse_relative_pair, year_earlier_label,
                                            single_period_label, period_kind,
                                            parse_same_period_pair, validate_period_labels,
                                            granularity_from_labels, frequency_in_text)
from app.db import retriever
from app.compute import engine as compute
from app.compute.verifier import (verify_numbers, render_template_fallback,
                                   looks_degenerate, writes_own_sources)
from app.compute.citations import Citation, citations_for_rows, render_sources_footer, citations_to_dicts
from app.compute.chart_builder import build_chart_spec
from app.compute.chart_request import wants_chart
from app.compute.formatting import (display_unit, decimals_from_format, trim_decimal,
                                     round_for_display)
from app.agents.composer_agent import compose_answer, compose_answer_streamed
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


def _stage(session_state: dict, name: str, **detail) -> None:
    """Announces which part of the pipeline is running, if anyone is listening.

    /chat is one blocking POST and the wait is real: an indicator resolution, a
    retrieval and a composition, in series. Nothing came back until all three
    were done, so a slow question was indistinguishable from a hung one.

    Stages, not tokens. The composed draft cannot be streamed as it is written
    because verify_numbers checks it against the payload AFTER it is complete,
    and an answer that fails is replaced by a template — streaming it would show
    the reader figures that are then retracted, which is worse than waiting.
    What can be streamed honestly is where the work has got to.

    The callback is stashed on session_state to avoid threading a parameter
    through the whole call graph, and popped in _finish for the same reason
    _history is: it is not state, and it is not serialisable.
    """
    callback = (session_state or {}).get("_progress")
    if not callback:
        return
    try:
        callback(name, detail)
    except Exception:
        # A listener that has gone away — a closed connection, most often — must
        # not take the answer down with it. The work is worth finishing either
        # way: it is still logged, and still recorded in the conversation.
        pass


class SessionState(TypedDict, total=False):
    """What a follow-up can inherit. Written every turn, read by
    conversation.carry_forward() on the next one."""
    last_indicator_detail_id: str
    last_indicator_name: str
    last_countries: list
    last_country_group: str
    last_period_expression: str
    last_explicit_frequency: str
    # How long the answer should be, and what the last one said. Neither is a
    # slot carry_forward fills — they are presentation, not subject — but they
    # ride in the same dict because it is the one thing already threaded to
    # every _finish call site.
    answer_length: str
    last_exchange: dict


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


def decide_route(user_message: str, intent: dict, session_state: dict) -> tuple[str, bool]:
    """Which computation the question gets, and whether a pattern here decided it.

    Split out of handle_message so it can be measured. The intent agent proposes
    a type and the patterns below override it, which means a routing regression
    is invisible until a user reports one — and two did. Nothing can be said
    about how often the model is right, or whether a prompt change made it
    worse, while the decision is buried in the middle of a 2,300-line function.
    scripts/eval_routing.py calls this directly.

    A pure move: same order, same conditions, same precedence. The rescue block
    comes along because it is routing too — it reads only the message and the
    proposed type, and the returns it used to sit behind (general_chat,
    capabilities) are never out_of_scope, so running it here changes nothing.

    Returns (ctype, scope_pinned). scope_pinned says a pattern chose a group
    route, which the caller trusts more than the model choosing one: a pattern
    fires on wording only a group question has.
    """
    ctype = intent.get("computation_type", "out_of_scope")
    scope_pinned = False

    # The model decides. Every pattern that used to override it here is gone,
    # and the eval is why: across 90 questions the eight of them rescued five
    # cases and broke one, while the intent agent routed 79 unaided. Each has a
    # rule in the prompt now, stated as a distinction rather than as the
    # vocabulary the pattern was matching on.
    #
    # Two things stayed, because neither is a pattern second-guessing a model
    # that answered:
    #
    #   - the limit follow-up, which needs session state. "just give me top 3"
    #     names no group, and which group it means is not in the message at all.
    #   - the out_of_scope rescue below, which fires ONLY when the model has
    #     already given up. It cannot override a routed answer, so it costs
    #     nothing to keep and catches the case where the model refuses a
    #     question one of these types can answer.
    #
    # Greetings are the one deletion with a named risk: the model sent "سلام"
    # and "سلام علبكم" down the data path once, and only "hello" and
    # "سلام عليكم" are in the eval. If greetings regress, is_greeting is the
    # first thing to put back.
    if (_clean_limit(intent.get("limit")) and intent.get("is_followup")
            and session_state.get("last_catalog_scope")
            and not _match_catalog_scope(user_message)[0]):
        # Always the performance ranking: "top 3" asks for an ORDER, and
        # progress against each indicator's own target is the only order this
        # group has. A direction split or a snapshot has no first and last to
        # take three of.
        ctype = "scope_performance"

    # Rescue before refusing. Several deterministic routes are decided further
    # down, once an indicator has resolved — and this return fires first, so
    # anything the intent agent called out_of_scope never reached them.
    # "If non-hydrocarbon exports account for 38.6%, what share still comes
    # from hydrocarbon exports?" reads like a puzzle rather than a data
    # question, the model classified it out_of_scope, and the complement route
    # built for exactly that question was unreachable.
    if ctype == "out_of_scope":
        if _asks_for_complement_share(user_message):
            ctype = "complement_share"
        elif _asks_if_improving(user_message):
            ctype = "direction_check"
        elif _asks_about_diversification(user_message):
            ctype = "diversification_overview"
        elif _proposes_derived_figure(user_message):
            ctype = "denominator_check"

    # The intent agent can now route to the group family itself, which is the
    # only way "which are ahead of target" reaches a ranking — no pattern here
    # contains that wording, and nothing else would have. The cost of letting it
    # route is that it also offers a group type for questions about ONE
    # indicator: "is inflation rising?" is a direction question and reads like a
    # direction split.
    #
    # A group type with no group named in the message and none under discussion,
    # over wording that resolves confidently to a single published indicator, is
    # that mistake. The indicator wins — the same test macro_overview and
    # article_lookup already apply below, for the same reason.
    if (ctype in ("scope_performance", "scope_direction", "scope_snapshot")
            and not scope_pinned
            and not session_state.get("last_catalog_scope")
            and not _match_catalog_scope(user_message)[0]):
        probe = resolve_indicator(user_message, require_data=True,
                                   language=detect_language(user_message, intent.get('language')))
        if (probe.status == "resolved" and probe.match
                and probe.match.confidence >= CONFIDENT_MATCH):
            # Direction asked of one indicator is direction_check, which answers
            # it. Falling back to latest_value there would print a number at
            # someone who asked which way it was going.
            ctype = "latest_value" if ctype == "scope_snapshot" else "direction_check"
            intent["indicator_phrase"] = intent.get("indicator_phrase") or user_message

    return ctype, scope_pinned


ANSWER_LENGTHS = ("brief", "detailed")


def _asks_for_length(intent: dict) -> Optional[str]:
    """The length the model extracted, accepted only if it is one we know.

    No patterns. "Answer in short" and "اختصر الإجابة" and "spare me the
    commentary" all mean the same thing and share no words, which is the reason
    a router exists — the same reason 26b33a4 stopped the patterns overriding
    it. A regex here would have to enumerate the phrasings in two languages and
    would still miss the next one, while reading "short" in "a short-term rate"
    or "the shortest series" as a formatting instruction.

    What is left is not detection but validation: the model is asked for one of
    two words, and anything else — a third word it invented, a stray sentence,
    a null — is treated as "they did not say". A value outside the set would
    silently select no directive at all in the Composer, so it is rejected here
    where that is visible rather than there where it is not.
    """
    value = (intent or {}).get("answer_length")
    value = str(value).strip().lower() if value is not None else ""
    return value if value in ANSWER_LENGTHS else None


def handle_message(user_message: str, conversation_context: str = "",
                    session_state: Optional[SessionState] = None,
                    history: Optional[list] = None,
                    progress: Optional[Callable[[str, dict], None]] = None) -> dict:
    session_state = dict(session_state or {})
    # dict() is shallow, so the stamps would be the SAME object the store holds,
    # and forget_stale below would age slots out of the stored session before
    # this turn had succeeded. A request that then failed would have silently
    # forgotten things on the user's behalf.
    if session_state.get("_slot_turns"):
        session_state["_slot_turns"] = dict(session_state["_slot_turns"])
    if progress:
        session_state["_progress"] = progress
    _stage(session_state, "understanding")
    # Which turn of the conversation this is. Everything that ages — a slot a
    # follow-up may inherit, a length preference — is measured against it, so it
    # has to be bumped before anything reads the state.
    session_state["turn_index"] = turn_index(session_state) + 1
    # Then drop what has aged out, once, here. Several stages below read
    # last_indicator_name and last_catalog_scope directly rather than through
    # carry_forward; expiring at the point of use would have to be repeated at
    # each of them and would be missed at the next one added.
    forget_stale(session_state)
    # Parked on the state so _finish can reach it without a parameter threaded
    # through twenty-five call sites. Popped there before the state is returned,
    # so it is never persisted — see the note at the end of _finish.
    if history:
        session_state["_history"] = history
    intent = extract_intent(user_message, conversation_context)
    # What the message ITSELF said, before anything was inherited into it. The
    # test for "this message carries an instruction and nothing else" has to ask
    # the extraction, not the carried intent — carry_forward fills the very
    # fields that would make it look like the message named something.
    stated = {k: intent.get(k) for k in
              ("indicator_phrase", "countries_mentioned", "country_group_mentioned",
               "period_expression")}
    # The frequency THIS question names, filled in where the router missed it.
    #
    # Before carry_forward, and that ordering is the whole point. It has to run
    # above the block that records what the turn was about, or a frequency read
    # from the text is never remembered: "List Qatar's QUARTERLY GDP values for
    # 2024 and 2025" selected the quarterly series and stored nothing, so the
    # follow-up "بالعربي, لسنه 2021 و 2023" — which names no frequency of its own
    # — had none to inherit and silently answered in yearly figures. A
    # conversation about quarters does not stop being about quarters because the
    # next message only changes the years.
    #
    # And it has to run BEFORE the inheritance, not after. carry_forward fills
    # this slot only when it is empty, so a frequency taken from the message
    # blocks the old one; reading it afterwards instead let "and monthly?"
    # inherit quarterly and ignore the word the user had just typed.
    if not intent.get("explicit_frequency"):
        intent["explicit_frequency"] = frequency_in_text(user_message)
    # And the language, for the same reason and with the same ordering. The
    # router leaves answer_language empty often enough that "بالعربي" on its own
    # was sent to the indicator resolver and refused as the name of an
    # indicator. Only where the message names NOTHING else — see `stated` below,
    # which is what keeps "how many people speak English in Qatar" a question
    # about data.
    if not intent.get("answer_language") and not any(
            intent.get(k) for k in ("indicator_phrase", "countries_mentioned",
                                     "country_group_mentioned", "period_expression")):
        intent["answer_language"] = language_request_in_text(user_message)
    # A follow-up states only what changed. Inherit the rest from the previous
    # turn before anything is resolved, so "and for Saudi Arabia?" keeps the
    # earlier indicator AND period instead of only the indicator.
    intent = carry_forward(intent, session_state, user_message)
    # Set when "which performed best?" is redirected onto a country ranking, so
    # the direction comes from the indicator's polarity rather than from a
    # superlative the user never used.
    performance_followup = False

    ctype, scope_pinned = decide_route(user_message, intent, session_state)
    # An instruction about how to WRITE the answer, mistaken for small talk.
    #
    # "جاوب بالعربي" was answered "على الرحب والسعة — اسألني عن أي شيء آخر": a
    # pleasantry in reply to an instruction, which tells the user their request
    # was not understood and then does not carry it out. The same trap sits under
    # "in short" said on its own.
    #
    # The router is told about these in its prompt and this is not second-
    # guessing it: greeting or not cannot be decided from the message alone. "جاوب
    # بالعربي" with nothing before it IS small talk — there is no answer to
    # restate — and only the session knows whether there is. That is the one
    # thing a rule here has that the router does not.
    #
    # Not gated on general_chat. The router calls this a greeting sometimes and
    # a data question other times — after an article answer, "جاوب بالعربي" was
    # sent to the indicator resolver and came back "no indicator matches 'جاوب
    # بالعربي'". Which wrong route it took does not matter; what matters is that
    # the message carries an instruction and nothing else, which is exactly what
    # the empty fields below say.
    if ((intent.get("answer_language") or intent.get("answer_length"))
            and not any(stated.values())
            and session_state.get("last_ctype")):
        ctype = session_state["last_ctype"]
        intent["is_followup"] = True
        # carry_forward has already run, so the slots it skipped while this
        # looked like a greeting have to be filled now. "Answer the last question
        # in Arabic" is six words and matches none of the short-follow-up shapes,
        # so without this it would inherit the computation and no subject.
        intent = carry_forward(intent, session_state, user_message)
    # A follow-up that only swaps the subject keeps the question being asked —
    # "What does inflation mean?" then "what about GDP" is asking what GDP
    # means, not what it is worth. That decision is the intent agent's, not a
    # pattern's: the transcript it is given annotates each turn with
    # "computation=", so it can see what was being asked, and its prompt now
    # tells it to carry that forward.
    #
    # This was briefly a regex here, matching "and", "what about", "how about".
    # It was the same mistake 26b33a4 removed from decide_route and the same one
    # _asks_for_length is written to argue against: the phrasings are
    # open-ended and bilingual and share no keyword, so a list of them is always
    # one short, and it cannot tell "and GDP?" from "and what about the trend
    # since 2019" without also encoding what counts as too much to inherit over.
    #
    # Recorded twice, and both are needed. Here, because definition,
    # analysis_lookup and every refusal return long before the dispatch; and
    # again just before the dispatch, because the overrides between the two are
    # what turn a router default into the computation that actually ran. It
    # reaches the next turn through the transcript annotation.
    session_state["last_ctype"] = ctype
    _stage(session_state, "routed", computation=ctype)
    # Arabic script is unambiguous; the model's language field is a guess. An
    # Arabic greeting was being answered in English, which is the product simply
    # not working in one of its two languages.
    #
    # The conversation's own language is the last fallback, below the script and
    # below the model. A follow-up often carries no script to read — "2024",
    # "top 3", "GDP" — and detection then fell through to English regardless of
    # what the eight turns above it were in. A thread does not change language
    # because one message in it was a number.
    language = detect_language(user_message, intent.get("language"))
    if session_state.get("last_language") and not carries_language_signal(user_message):
        language = session_state["last_language"]
    # An explicit instruction outranks both. Script says what the QUESTION is in;
    # this says what the ANSWER should be in, and they are different things —
    # "answer in Arabic", typed in English, means the answer is Arabic.
    #
    # It persists, for the same reason answer_length does: someone who asks once
    # to be answered in Arabic means it for the conversation, not for one turn.
    # And it decays with the other slots, so an instruction given about a subject
    # the user has long left does not govern a new one forever.
    asked_language = str(intent.get("answer_language") or "").strip().lower()
    if asked_language in ("en", "ar"):
        remember(session_state, "answer_language", asked_language)
    elif (session_state.get("answer_language")
            and carries_language_signal(user_message)
            and detect_language(user_message) != session_state["answer_language"]):
        # ...until the user writes a question in the other language themselves.
        #
        # "بالعربي" set this and kept it, so "which 3 countries have the lowest
        # inflation" — typed in English, several turns later — came back in
        # Arabic. The preference was a real instruction and outlived its
        # occasion: someone typing a full question in English is telling us
        # which language they want more plainly than a request three turns ago.
        #
        # Only against a message with enough script to be sure of. A bare "2024"
        # or "top 3" says nothing about language, and dropping the preference on
        # those would undo it by accident — which is the whole reason
        # carries_language_signal exists.
        session_state.pop("answer_language", None)
        (session_state.get("_slot_turns") or {}).pop("answer_language", None)
    if session_state.get("answer_language"):
        language = session_state["answer_language"]
    session_state["last_language"] = language

    # Remember what this turn was about, whatever happens below, so the next
    # follow-up has something to refer back to even if this one fails to
    # resolve an indicator.
    # remember(), not a plain assignment: it stamps the turn that wrote the slot,
    # which is what lets forget_stale drop it once the conversation has moved on.
    if intent.get("period_expression"):
        remember(session_state, "last_period_expression", intent["period_expression"])
    if intent.get("explicit_frequency"):
        remember(session_state, "last_explicit_frequency", intent["explicit_frequency"])
    # Which end of a ranking was asked for, so "what about in 2022" keeps it.
    # Without this the follow-up relied on the default happening to match.
    if intent.get("extremum"):
        remember(session_state, "last_extremum", intent["extremum"])
    elif intent.get("is_followup") and session_state.get("last_extremum"):
        intent["extremum"] = session_state["last_extremum"]
    if intent.get("countries_mentioned"):
        remember(session_state, "last_countries", list(intent["countries_mentioned"]))
    if intent.get("country_group_mentioned"):
        remember(session_state, "last_country_group", intent["country_group_mentioned"])

    # How long the answer should be, as the router read it. Without this the
    # request reached the Composer as four words of question text competing
    # with twenty-one numbered rules pulling the other way, and won about half
    # the time: "in a short answer" got one sentence, "respond in short" got
    # four. It is an extraction, not a pattern, because the ways of asking for
    # a shorter answer are open-ended and bilingual and share no keyword.
    #
    # Carried in session_state, which every _finish call already receives, so
    # nothing has to be threaded through twenty-three call sites. It persists
    # across follow-ups — "in short" then "and for 2024?" stays short, which is
    # what asking for it once means — and is dropped on a fresh question, since
    # a new subject is not covered by a preference expressed about the last one.
    #
    # It also expires. Clearing it only on a fresh question left one hole: the
    # router marks long runs of a conversation as follow-ups, so a single "in
    # short" eight turns ago could still be shortening answers to questions that
    # have nothing to do with the one it was said about. remember() stamps it and
    # forget_stale above drops it after STALE_AFTER_TURNS, which is the same
    # window a subject slot gets and for the same reason — a preference stated
    # about one question does not outlive the reader's memory of stating it.
    length = _asks_for_length(intent)
    if length:
        remember(session_state, "answer_length", length)
    elif not intent.get("is_followup"):
        session_state.pop("answer_length", None)

    # --- non-data intents ---
    if ctype == "general_chat":
        payload = {"ok": True, "facts": {"note": "Greeting — no data needed."}}
        # The full greeting introduces the assistant and lists five things it can
        # be asked. That is the right answer to "hello" on turn one and the wrong
        # one on turn nine: someone who has been asking about inflation for eight
        # turns and types "thanks" does not need to be told what the product is,
        # and being told anyway is the clearest sign that nothing was remembered.
        opener = "greeting" if turn_index(session_state) <= 1 else "greeting_again"
        return _finish(payload, language, session_state, [], skip_compose=True,
                        canned=msg(opener, language),
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


    # Parsed HERE, above every route that reads it, rather than part-way down.
    # It already sat too low once: "Show GDP growth, inflation, and government
    # revenues for the year 2023" returned each indicator's latest reading
    # because the period was parsed after the snapshot had answered. The same
    # omission was still live for the scope_* family below, which sits higher
    # still — "the national indicators 3 years ago" was answered with today's
    # twelve readings. A value every branch depends on belongs before the first
    # branch, not before the second.
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

    if ctype in ("scope_performance", "scope_direction", "scope_snapshot"):
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

    if ctype in ("scope_performance", "scope_direction", "scope_snapshot"):
        session_state["last_catalog_scope"] = {"kind": kind, "scope": scope}
        # The period the question asked for, which this family used to drop on
        # the floor. "What about 3 years ago" then re-ran the same query and
        # returned the same twelve current readings, and the answer described
        # them as three years old.
        scope_start = getattr(period, "start_date", None)
        scope_end = getattr(period, "end_date", None)
        entries = retriever.get_scope_performance(kind, scope,
                                                   start_date=scope_start,
                                                   end_date=scope_end)
        # Each row is a different indicator with its own unit and precision, so
        # the Format column has to be applied per row rather than once for the
        # answer. Without this the ratio indicators print as "9.0 NA".
        for e in entries:
            e["unit_en"] = display_unit(e.get("unit_en"), e.get("format"),
                                         language, e.get("unit_ar"))
            e["decimal_places"] = decimals_from_format(e.get("format"))
            # The indicator's own Arabic name, beside its own Arabic unit. Only
            # the unit was being switched here, so an Arabic answer about a
            # sector read "Cost per Student: 45.2 ألف ر.ق" — the number and the
            # unit localised, the subject not.
            if is_arabic(language) and (e.get("indicator_ar") or "").strip():
                e["indicator"] = e["indicator_ar"].strip()
        session_state["last_ctype"] = ctype
        if ctype == "scope_snapshot":
            result = compute.scope_snapshot(entries)
        elif ctype == "scope_direction":
            # Which half was asked for. The model's reading wins where it has
            # one; the regex is the fallback for when it says nothing, the same
            # arrangement scope_performance uses for "order" below.
            asked = intent.get("direction_asked")
            if asked not in ("up", "down"):
                asked = _requested_direction(user_message)
            result = compute.scope_direction(entries, direction_asked=asked)
        else:
            # The order is remembered, because a follow-up restates it even less
            # often than it restates the count: "just give me top 3" following a
            # worst-performing list would otherwise flip the list over under the
            # same heading.
            #
            # The model's reading wins where it has one — it is judging meaning,
            # and "which ones are we failing at" carries no keyword the regex
            # knows. The regex stays as the fallback for when it says nothing.
            order = intent.get("order")
            if order in ("best", "worst"):
                best_first = order == "best"
            else:
                best_first = _requested_order(
                    user_message, session_state.get("last_scope_best_first", True))
            session_state["last_scope_best_first"] = best_first
            result = compute.scope_performance(
                entries, best_first=best_first, limit=_clean_limit(intent.get("limit")))
        payload = {"ok": result.ok,
                   "facts": {**_round_facts(result.facts), "scope": scope, "scope_kind": kind}} \
            if result.ok else {"ok": False, "message": result.message}
        # Which window these readings come from, stated in the payload so the
        # answer can say it and so the Composer has a true period to name
        # instead of the one it read in the question. Where the group is empty
        # BECAUSE of the window, that is a different fact from the group not
        # existing, and the two were coming back as the same message.
        if period is not None and getattr(period, "kind", "unspecified") != "unspecified":
            asked_period = getattr(period, "raw", None) or intent.get("period_expression")
            if payload.get("ok"):
                payload["facts"]["as_of"] = asked_period
            elif not entries:
                payload = {"ok": False, "message": msg("none_in_period", language,
                                                        period=asked_period)}
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
            # Probed against the model's topic AND the message as written, best
            # score wins. The topic alone made this test depend on how the
            # router happened to paraphrase the question, and the router sees the
            # transcript — so the SAME question resolved differently depending on
            # what had been asked before it. "Does Qatar export anything beside
            # oil and gas" came back from the catalogue as Non-Hydrocarbon
            # Exports in one session and from an article in the next, with
            # nothing about the question having changed.
            #
            # The message is often the better probe of the two: "beside oil and
            # gas" sits closer to "Non-Hydrocarbon Exports as Share of Total
            # Exports" than a paraphrase that has dropped the contrast.
            probe, topic = _best_indicator_probe([topic, user_message], language)
            if (probe and probe.status == "resolved" and probe.match
                    and probe.match.confidence >= CONFIDENT_MATCH):
                ctype = "latest_value"
                intent["indicator_phrase"] = topic
        if ctype == "article_lookup":
            return _article_answer(user_message, intent, language, session_state)

    if ctype == "denominator_check":
        warning = _denominator_warning(user_message, language)
        if warning.get("indicators"):
            remember(session_state, "last_offered_indicators", warning["indicators"])
        payload = {"ok": False, "message": warning["message"]}
        return _finish(payload, language, session_state, [], question=user_message)

    # A calculation the user has done, proposed to us with its answer, on a
    # route that was never going to address it.
    #
    # This check existed and was unreachable: it sat inside the out_of_scope
    # rescue, so it only ran once the router had already given up. "There were
    # 126.6 thousand economically active Qataris and 8% of employed Qataris
    # worked in the private sector. Does that mean about 10,100?" was routed to
    # latest_value and answered "8.02 % in 2024" — which reads as agreement. The
    # user multiplied two different populations, got a plausible number, and the
    # system handed back a figure that appeared to confirm it. Silence about the
    # arithmetic IS an answer about the arithmetic when the question was only
    # about the arithmetic.
    #
    # Gated twice so it cannot hijack a question that merely sounds like one:
    # the message must state a figure the user computed, and the catalogue must
    # actually show the two indicators are measured over different populations.
    # Where either is missing, the question continues to the route it was given.
    if _proposes_computed_figure(user_message):
        warning = _different_bases_warning(user_message, language)
        if warning:
            # What the refusal just offered, so "give me on its own" has
            # something to land on. Without it the offer is a dead end.
            remember(session_state, "last_offered_indicators", warning["indicators"])
            return _finish({"ok": False, "message": warning["message"]}, language,
                            session_state, [], question=user_message)

    if ctype == "diversification_overview":
        payload, citations = _diversification_overview(language)
        return _finish(payload, language, session_state, citations,
                        question=user_message)

    # The macro snapshot answers "how is the economy doing". It was also
    # answering "how is Qatar doing on competitiveness?", because the model
    # routes any broad-sounding "how is X doing" to it and this branch returns
    # before an indicator is ever resolved — so the question named a published
    # indicator, and got four unrelated ones.
    #
    # Our own detector is authoritative for whether a question is about the
    # economy at large. Where it disagrees with the model, a confidently
    # resolved indicator wins: the question named something specific.
    if ctype == "macro_overview" and not _asks_about_the_economy(user_message):
        probe = resolve_indicator(user_message, require_data=True, language=language)
        if (probe.status == "resolved" and probe.match
                and probe.match.confidence >= CONFIDENT_MATCH):
            ctype = "latest_value"
            intent["indicator_phrase"] = intent.get("indicator_phrase") or user_message

    if ctype == "macro_overview":
        payload, citations = _macro_overview(language, period=period)
        if payload.get("ok") and wants_chart(user_message, ctype):
            payload["chart"] = build_chart_spec(ctype, payload["facts"], "Economic Overview", None, None)
        return _finish(payload, language, session_state, citations,
                        question=user_message)

    # --- everything below needs an indicator resolved first ---
    indicator_phrase = intent.get("indicator_phrase")
    # A complement question names TWO things: the indicator that exists and the
    # side that does not. Which one the model extracts decides whether the
    # question works at all — "...account for 40%..." gave "non-hydrocarbon
    # exports" and was answered, "...account for 66%..." gave "share of
    # hydrocarbon exports", which is not in the catalogue by design, and was
    # refused. Same question, different wording of a number.
    #
    # The whole message is used instead. It contains the published indicator's
    # own wording, and the complement rule downstream is what decides whether
    # the other side can be derived.
    if _asks_for_complement_share(user_message):
        indicator_phrase = user_message
    # A bare "what about if it was X" continues the previous computation on the
    # previous indicator. Without this it named neither, and a complement
    # question became a latest-value lookup.
    elif (_is_figure_followup(user_message)
            and session_state.get("last_ctype") == "complement_share"
            and session_state.get("last_indicator_name")):
        indicator_phrase = session_state["last_indicator_name"]
        ctype = "complement_share"
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

    # Taking up an offer the previous answer made in prose.
    #
    # The refusals name indicators and then invite the user to ask for them —
    # "I can give you either figure on its own". Saying yes to that is a normal
    # thing to do, and it failed: "give me on its own" was resolved as though it
    # were the name of an indicator and came back "No indicator in SCAI's
    # approved data matches 'give me on its own'". The system offered something,
    # the user accepted, and it denied having anything to give.
    #
    # The chip mechanism above cannot cover this. It matches the typed text
    # against a candidate name exactly, which works for a button and not for a
    # sentence. What is needed is the OFFER itself remembered, so an acceptance
    # in any wording lands on it.
    #
    # Only when the message does not name an indicator of its own: "give me Real
    # GDP on its own" is answered about Real GDP, not about both offered names.
    on_offer = session_state.get("last_offered_indicators") or []
    if on_offer and indicator_phrase:
        probe = resolve_indicator(indicator_phrase, require_data=True, language=language)
        if not (probe.status == "resolved" and probe.match
                and probe.match.confidence >= CONFIDENT_MATCH):
            # Both, because "either" named two and the acceptance chose neither.
            # Answering with one would be picking for them.
            indicator_phrase = ", ".join(on_offer)
        session_state.pop("last_offered_indicators", None)

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
    # because the period was parsed below this point, long after the snapshot
    # had already answered. Asking for any one of the three on its own honoured
    # 2023 correctly, which is what made the omission visible. The parse now
    # sits above the scope_* routes as well, which had the same defect for the
    # same reason; see the comment there.
    # Routes that have already been decided deterministically are not
    # multi-metric questions, whatever punctuation they contain. "If
    # non-hydrocarbon exports account for 38.6% of total exports, what share
    # still comes from hydrocarbon exports?" split on its comma, answered the
    # first half as a metric and reported the second half as an indicator that
    # does not exist — reaching neither the complement route nor a refusal.
    _SINGLE_SUBJECT = ("definition", "complement_share", "direction_check")
    if ctype not in _SINGLE_SUBJECT and not resembles_catalogue_name(indicator_phrase or ""):
        parts = split_indicator_phrases(indicator_phrase or "")
        if len(parts) > 1:
            payload, citations = _indicator_snapshot(parts, language, period)
            # One line is still the answer to a three-metric question when the
            # other two have nothing for the period asked about. This required
            # more than one and therefore THREW AWAY a correct partial answer:
            # "exports, government revenues and the trade balance in Q1 2026"
            # has Q1 2026 for government revenues only, so the snapshot was
            # discarded, the whole sentence was sent to the resolver as if it
            # named one indicator, it matched Trade Balance, and the reply was
            # a flat refusal — hiding the figure the question could be answered
            # with and never mentioning the other two.
            if payload.get("ok") and payload["facts"]["overview"]:
                if wants_chart(user_message, "macro_overview"):
                    payload["chart"] = build_chart_spec("macro_overview", payload["facts"],
                                                         "Economic Snapshot", None, None)
                # Remember WHAT was answered about. The single-indicator path
                # records last_indicator_name and this one recorded nothing, so
                # "what about in 2024" after a three-metric answer had no
                # subject to inherit and was refused for naming no indicator —
                # after the system had just listed three.
                remember(session_state, "last_metrics",
                          [e["indicator"] for e in payload["facts"]["overview"]])
                session_state.pop("last_indicator_name", None)
                return _finish(payload, language, session_state, citations,
                        question=user_message)

    # The question may already have answered the question we are about to ask.
    # "What does the 13.4% share of non-hydrocarbon government revenue mean"
    # offered a choice of three indicators whose latest readings are 13.36, 5.1
    # and 3.94 — the figure the user quoted picks one outright, and asking them
    # to choose again ignores what they already said.
    if resolution.status == "ambiguous" and resolution.candidates:
        picked = (_disambiguate_by_quoted_value(resolution.candidates, user_message)
                  or _disambiguate_by_asked_unit(resolution.candidates, user_message))
        if picked:
            resolution = ResolutionResult(picked, "resolved", [])

    # A stub that was never a name, on a turn that has a subject already.
    #
    # "what does ir mean" — a typo for "it" — was sent to the resolver as an
    # indicator called "ir" and refused: 'No indicator matches "ir"'. The
    # conversation had just been about Real GDP, and the user was plainly asking
    # about that. Quoting two characters back at someone as though they had
    # misnamed a metric is the system telling them their question was
    # unrecognisable when it was not.
    #
    # The same failure at full length: "what is the latest recorded value?",
    # asked straight after a definition of Inflation, was refused as the name of
    # an indicator. That is the canonical follow-up this system was built for —
    # conversation.py quotes it as QC finding F-030 — and it was failing because
    # the test here was the phrase's LENGTH, which "latest recorded value" is
    # not short enough to pass.
    #
    # Length was always the wrong question. The right one is whether the phrase
    # attempts a name this catalogue could hold, and the catalogue can answer
    # that: not one of "latest", "recorded" or "value" appears in any of the 583
    # indicator names, so the phrase names nothing and is scaffolding around a
    # question about whatever is already under discussion.
    #
    # Still narrow, because the alternative failure is worse — silently
    # answering about the previous indicator when the user really did name a new
    # one this system does not carry. "Solar energy share" is not a catalogue
    # name either, but "energy" and "share" ARE catalogue words, so it reads as
    # a genuine attempt and still earns the refusal that says so.
    # Not gated on is_followup. That is the router's field, it is what failed
    # on this very question, and it adds nothing here: a phrase that names
    # nothing the catalogue holds, asked while an indicator is under discussion,
    # IS a follow-up whatever the router called it. The remembered indicator is
    # the real gate — it exists only within the decay window, so there is no
    # subject to fall back to once the conversation has moved on.
    if (resolution.status not in ("resolved", "inactive")
            and session_state.get("last_indicator_name")
            and names_nothing_in_catalogue(indicator_phrase or "")):
        indicator_phrase = session_state["last_indicator_name"]
        resolution = resolve_indicator(indicator_phrase,
                                        require_data=(ctype != "definition"),
                                        language=language)

    if resolution.status != "resolved" and resolution.status != "inactive":
        # A question about THE ECONOMY, which is not an indicator and never
        # will be. "Compare Qatar's economy between 2023 and 2025" was told to
        # "specify particular metrics such as inflation, Real GDP, trade
        # balance" — the exact four the macro overview would have given
        # unprompted, so the system knew the answer and asked the user to ask
        # again in its own vocabulary.
        #
        # _asks_about_the_economy is stricter than this on purpose: it decides
        # whether to PREFER the overview over a possible indicator, and needs
        # both the subject and a word about its state so that "what is economic
        # growth", one word away, still reaches Real GDP. Here nothing resolved,
        # so there is no indicator to prefer and no such care is needed — the
        # word "economy" and a failed lookup are between them the whole case.
        if _ECONOMY_WORD.search(user_message or ""):
            payload, citations = _macro_overview(language, period=period)
            if payload.get("ok"):
                if wants_chart(user_message, "macro_overview"):
                    payload["chart"] = build_chart_spec("macro_overview", payload["facts"],
                                                         "Economic Overview", None, None)
                return _finish(payload, language, session_state, citations,
                                question=user_message)
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
    # The slowest step the reader can be told anything useful about: it names the
    # indicator the rest of the answer will be about, so a wrong match is visible
    # before the retrieval and the composition have been paid for.
    _stage(session_state, "resolved", indicator=match.name_en)
    remember(session_state, "last_indicator_detail_id", match.indicator_detail_id)
    remember(session_state, "last_indicator_name", match.name_en)
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
        # Honour a period the question named. This route always took the six
        # most recent readings and returned the newest commentary among them,
        # so "what did the Council say about inflation in 2023" answered with
        # this quarter's analysis — commentary written about a different
        # reading, presented as though it were about the one asked for.
        #
        # Narrowed rather than refused when the window holds nothing: unlike a
        # figure, analysis is published irregularly, and the nearest commentary
        # is often what the reader wants. Which reading each entry belongs to is
        # stated in the answer either way, so a period that could not be met is
        # visible rather than silently substituted.
        if period.start_date or period.end_date or period.kind == "last_n_years":
            in_window = _apply_last_n_years(
                [r for r in with_actuals
                 if (not period.start_date or (r.get("period_date") and r["period_date"] >= period.start_date))
                 and (not period.end_date or (r.get("period_date") and r["period_date"] <= period.end_date))],
                period)
            if in_window:
                with_actuals = in_window
        recent = list(reversed(with_actuals[-6:]))
        found = retriever.get_analysis_for_data_points([r["record_id"] for r in recent])
        entries = []
        for row in recent:
            analysis = found.get(row["record_id"])
            if not analysis:
                continue
            entry = {"period_label": row["period_label"], "value": row.get("actual")}
            entry.update(retriever.analysis_text(analysis, language))
            if len(entry) > 2:
                entries.append(entry)
            if entries:
                break   # the LATEST analysis, not a history of them

        if not entries:
            payload = {"ok": False, "message": msg("no_analysis", language,
                                                    indicator=match.display_name(language))}
            return _finish(payload, language, session_state, [],
                        question=user_message)

        citations = [Citation(indicator=match.name_en.strip(), data_source=match.data_source_en,
                               table="indicator_analysis", record_id=None,
                               period_label=entries[0]["period_label"], country="Qatar")]
        payload = {"ok": True, "facts": {"indicator": match.display_name(language),
                                          "unit": match.unit_en, "analysis": entries}}
        # Verbatim, and skip the Composer entirely: this is the Council's own
        # wording, and a model asked to "phrase" it would paraphrase it.
        body = "\n\n".join(
            f"{match.display_name(language)} — {e['period_label']}"
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
                remember(session_state, "last_indicator_detail_id", match.indicator_detail_id)
                remember(session_state, "last_indicator_name", match.name_en)

        definition = match.definition_ar if language == "ar" and match.definition_ar else match.definition_en
        definition = strip_html(definition or "").strip()
        if not definition or definition == "-":
            payload = {"ok": False, "message": msg("no_definition", language,
                                                     indicator=match.display_name(language))}
            return _finish(payload, language, session_state, [],
                        question=user_message)
        payload = {"ok": True, "facts": {
            "indicator": match.display_name(language),
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
    # Already filled from the question in handle_message where the router left
    # it empty, so this is one source again and a follow-up can inherit it.
    explicit_gran = parse_explicit_frequency(intent.get("explicit_frequency"))
    # A named period implies its own granularity, and it is stronger evidence
    # than the default. Without this the finest series always won: "GDP for
    # 2023" returned 170.55, which is Real GDP in 2023-Q4, while the published
    # yearly row for 2023 reads 696.697 — a quarter reported as a year, wrong
    # by a factor of four and entirely plausible-looking. "Compare 2023 to
    # 2024" simply failed, because a quarterly series has no row called "2023".
    #
    # An explicitly stated frequency still wins over both: someone asking for
    # "quarterly GDP in 2023" means quarters.
    implied_gran = granularity_from_labels(period.raw or user_message)
    if not explicit_gran and implied_gran in available_gran:
        granularity = implied_gran
        # ...unless nothing is published at that granularity INSIDE the window.
        #
        # A year implies the yearly series, which is right whenever the yearly
        # row exists. For the CURRENT year it does not: the year is not over.
        # "قارن نسبة التضخم بين قطر وسنغافورة لسنة 2026" was refused — no
        # approved data for either country in 2026 — while the same question
        # without the year answered from April 2026 for both. The data was
        # there the whole time, one granularity down.
        #
        # Falling back rather than refusing matches what the multi-indicator
        # snapshot already does with a window: read each series at its latest
        # reading WITHIN it, so a 2023 question gives 2023-Q4 for a quarterly
        # series and 2023-12 for a monthly one. Refusing instead tells the
        # reader SCAI published nothing that year, which is a claim about the
        # data rather than about the shape of the query.
        if (period.start_date or period.end_date) and not retriever.get_series(
                match.indicator_detail_id, published_detail_id, granularity,
                start_date=period.start_date, end_date=period.end_date):
            finer = choose_granularity(None, {g for g in available_gran if g != granularity})
            if finer:
                granularity = finer
    else:
        granularity = choose_granularity(explicit_gran, available_gran)

    if granularity is None:
        # `wanted` is None when the user named no frequency, which rendered as
        # the literal 'has no None data'. Distinguish the two real cases: the
        # indicator has data but not at the frequency asked for, versus it has
        # no data at all.
        wanted = intent.get("explicit_frequency")
        if available_gran:
            message = msg("no_data_at_frequency", language, indicator=match.display_name(language),
                           wanted=wanted, available=", ".join(sorted(available_gran)))
        else:
            message = msg("no_data_at_all", language, indicator=match.display_name(language))
        payload = {"ok": False, "message": message}
        return _finish(payload, language, session_state, [],
                        question=user_message)

    # Decided here rather than by the model, and only once an indicator has
    # resolved: "is X improving or deteriorating?" is about direction of
    # travel, which is neither a level nor a trend shape. It was being routed
    # to whichever of those the model picked, and answered with a number that
    # did not address the question asked.
    if _asks_if_improving(user_message) and ctype not in ("definition", "count_list"):
        ctype = "direction_check"
    # ...but a direction question that names a SPAN is a question about that
    # span, and direction_check cannot answer it. It compares the latest reading
    # with the one a year earlier, full stop — the period the user asked for
    # never reaches it.
    #
    # "How was inflation over the last 3 years, did it increase or not?" came
    # back as April 2026 against April 2025: a one-year answer to a three-year
    # question. Worse, the Composer could see the mismatch and tried to explain
    # it, writing that the data "does not provide monthly readings for the
    # entire three-year period" — an absence rule 3 forbids it from asserting,
    # and one that is not true. The payload held one reading because this route
    # only ever retrieves one, not because the series is short.
    #
    # A trend answers both halves of the question: the shape across the span,
    # and the change from one end of it to the other.
    if ctype == "direction_check" and period.kind in ("last_n_years", "range"):
        ctype = "trend"
    # Checked after the indicator resolves, because whether this is answerable
    # depends on WHICH indicator it is, not on the wording. The wording only
    # gets it as far as being considered.
    if _asks_for_complement_share(user_message) and ctype not in ("definition", "count_list"):
        ctype = "complement_share"

    # The computation this turn actually ran, after every override above. Set
    # here rather than at the top because the overrides are the point: a
    # question the router called latest_value and the pipeline turned into a
    # direction_check is a direction_check, and that is what a follow-up should
    # inherit. Two branches used to record it for their own purposes; every
    # route needs it now that a subject swap keeps the question.
    session_state["last_ctype"] = ctype

    countries_res = resolve_countries(intent.get("countries_mentioned", []), intent.get("country_group_mentioned"))

    # The database reads and the computation. Fast next to the two model calls
    # either side of it, but without this the listener went straight from
    # "resolved" to "composing" and sat there for the whole of the slowest step —
    # a progress display that is silent through the longest stretch tells the
    # reader less than one that admits what it is doing.
    _stage(session_state, "retrieving", indicator=match.name_en)

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
    unit = display_unit(match.unit_en, match.format, language, match.unit_ar)
    decimals = decimals_from_format(match.format)
    # SCAI's own Arabic name when the answer is Arabic, its English one
    # otherwise. Every figure in an Arabic answer carried an English indicator
    # name — "سجّلت Bahrain أدنى Inflation" — because this took name_en
    # unconditionally, while unit_ar and definition_ar were already being used a
    # few lines away. The catalogue holds the translation; nothing has to invent
    # one, which is the whole of composer rule 4's concern.
    indicator_name = match.display_name(language)
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
        #
        # "Named none" is not the same as "resolved to an empty list". Qatar is
        # the domestic sentinel — it carries no country_en string, so naming it
        # leaves resolved_countries empty — and this read that as naming nobody.
        # "وماذا عن قطر", asked straight after a ranking, therefore re-ran the
        # benchmark ranking and returned the identical three countries with
        # Qatar nowhere in it: the one country the question was about was the
        # one the answer left out.
        named_nobody = (not countries and not countries_res.includes_qatar
                        and not countries_res.unresolved_names)
        if ctype == "country_ranking" and named_nobody:
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
        # A limit cuts the BOTTOM off a ranking, which is the wrong end when the
        # user has named who they are asking about. "وماذا عن قطر" following
        # "which 3 countries have the lowest inflation" kept the three and cut
        # Qatar, whose inflation is higher — so the one country the question was
        # about was the one the answer removed.
        #
        # "Top 3" means "the top 3 of the set"; naming countries means "these
        # ones". They do not combine, and where both appear the named countries
        # win, because they are the subject and the limit is only a length.
        ranking_limit = None if not named_nobody else _clean_limit(intent.get("limit"))
        result = (compute.country_ranking(series_by_country, period_label, ascending=ascending,
                                           limit=ranking_limit)
                  if ctype == "country_ranking"
                  else compute.country_comparison(series_by_country, period_label))

        citations = []
        if result.ok:
            entries = result.facts.get("ranked") or result.facts.get("rows") or []
            # Country names in the language the answer is written in. Done here,
            # after the ranking and before the payload, because everything above
            # keys on the data's own English strings and everything below only
            # displays them. The citations built further down keep the English
            # name, since a citation identifies a row rather than reads as prose.
            if is_arabic(language):
                for entry in entries:
                    entry["country"] = display_country(entry.get("country"), language)
                for key in ("leader",):
                    if isinstance(result.facts.get(key), dict):
                        result.facts[key]["country"] = display_country(
                            result.facts[key].get("country"), language)
                if result.facts.get("only_country_with_data"):
                    result.facts["only_country_with_data"] = display_country(
                        result.facts["only_country_with_data"], language)
                result.facts["countries_with_no_data"] = [
                    display_country(c, language)
                    for c in result.facts.get("countries_with_no_data") or []]

            periods = {e.get("period_label") for e in entries if e.get("period_label")}
            if len(periods) > 1:
                # Only reachable when the countries share no period at all, so
                # alignment was impossible and each contributed its own latest.
                mixed = msg("no_common_period", language,
                             first=min(periods), last=max(periods))
                note = f"{note} {mixed}" if note else mixed
            elif period_label and len(entries) > 1:
                # Only when there is actually something to align. "Compare Real
                # GDP across Qatar and Saudi Arabia" has data for Qatar alone,
                # and telling the reader it was "compared at 2025-Q4, the most
                # recent period all of these countries report" described a
                # comparison that did not happen.
                aligned = msg("compared_at", language, period=period_label)
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
    # The periods the question actually LISTED, where it listed more than one.
    rows = _keep_named_periods(rows, intent.get("period_labels"))

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

    if ctype == "complement_share":
        rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                     granularity, country_en=single_country)
        latest = compute.latest_value(rows)
        # A figure quoted in the question may not be a wrong premise at all —
        # it may BE one of this indicator's readings, and so name the period
        # the question is about. "...account for 33.41532% of total exports..."
        # is Non-Hydrocarbon Exports in 2023-Q2 to six decimals; answering it
        # about Q4 2025 and calling 33.41532 a mistake gets both halves wrong.
        #
        # Only an exact match at the quoted precision, and only when exactly
        # one reading matches — otherwise the figure identifies nothing and the
        # latest reading stands, with the disagreement reported as before.
        anchored = _row_matching_quoted_value(rows, user_message)             if not (period.start_date or period.end_date) else None
        if anchored:
            latest = compute.ComputeResult(True, facts={
                "period_label": anchored["period_label"],
                "actual": anchored["actual"],
                "target": anchored.get("target"),
            })
        elif period.start_date or period.end_date or period.kind == "last_n_years":
            # A period was named, and until now nothing used it: the series was
            # fetched unfiltered and the LATEST reading answered, whatever year
            # the question was about. "If non-hydrocarbon exports were 38.6% in
            # 2023, what is the rest?" derived its remainder from the Q4 2025
            # share — the right arithmetic on the wrong reading, and nothing
            # downstream could catch it because the figure traces to the data
            # and passes the verifier.
            #
            # The guard above already knew a period could be present; it only
            # used that to switch OFF the quoted-value anchor, and then fell
            # through to the latest reading anyway.
            windowed = _apply_last_n_years(
                retriever.get_series(match.indicator_detail_id, published_detail_id,
                                      granularity, start_date=period.start_date,
                                      end_date=period.end_date,
                                      country_en=single_country), period)
            scoped = compute.latest_value(windowed)
            if not scoped.ok:
                # Named a period this indicator has no reading in. Say so with
                # the range it does cover, rather than deriving from a period
                # the user did not ask about.
                actuals = [r for r in rows if r.get("actual") is not None]
                return {"ok": False, "message": msg(
                    "no_data_for_period", language, indicator=indicator_name.strip(),
                    granularity=granularity, period=period.raw or "that period",
                    first=actuals[0]["period_label"] if actuals else "—",
                    last=actuals[-1]["period_label"] if actuals else "—")}, []
            latest, rows = scoped, windowed
        # complement_of_share requires the question to NAME the other side, and
        # a follow-up does not repeat it — "what about if it was 29.44474" says
        # only the figure. The question that established the complement is kept
        # and used, so the follow-up continues the same computation instead of
        # being told no complementary share was asked for.
        asked_as = user_message
        if _is_figure_followup(user_message) and session_state.get("last_complement_question"):
            asked_as = session_state["last_complement_question"]
        result = compute.complement_of_share(
            latest.facts.get("actual") if latest.ok else None,
            indicator_name, asked_as, stated_source=user_message)
        if not result.ok:
            # Not a two-way share, or the question did not ask for the other
            # side after all. Fall back to simply reporting the indicator,
            # which is a better answer than a refusal about a derivation the
            # user may not have meant to ask for.
            used = [_find_row_by_period(rows, latest.facts.get("period_label"))] if latest.ok else []
            return (_wrap(latest, unit, indicator_name=indicator_name, decimals=decimals),
                    citations_for_rows([r for r in used if r], indicator_name, data_source_en))
        result.facts["period_label"] = latest.facts.get("period_label")
        used = [_find_row_by_period(rows, latest.facts.get("period_label"))]
        # So "what about if it was 29.44474" continues this computation rather
        # than starting a different one.
        session_state["last_ctype"] = "complement_share"
        session_state["last_complement_question"] = asked_as
        return ({"ok": True, "facts": {**result.facts, "indicator": indicator_name.strip()}},
                citations_for_rows([r for r in used if r], indicator_name, data_source_en))

    if ctype == "direction_check":
        # Whichever granularity actually PUBLISHES a change, finest first.
        # The default is the finest available, and for Public Debt as a
        # Percentage of GDP that is quarterly — which carries no year-on-year
        # figure at all, while the yearly series carries yearly_yoy_pp. Taking
        # the default would have answered "its direction cannot be stated" on
        # an indicator whose direction is recorded, one granularity over.
        available = retriever.get_available_granularities(published_detail_id,
                                                           match.indicator_detail_id)
        # An explicitly requested frequency is tried FIRST, not just included in
        # the sweep. The order below is finest-first, which is the right default
        # and the wrong answer to "is YEARLY inflation improving?" — that was
        # answered from the monthly series whenever monthly publishes a change,
        # silently, because this loop takes the first granularity that works and
        # never saw what the user asked for. `granularity` above already honours
        # explicit_frequency; this branch computed its own and discarded it.
        order = [g for g in ("monthly", "quarterly", "yearly") if g in available]
        asked_gran = parse_explicit_frequency(intent.get("explicit_frequency"))
        if asked_gran in order:
            order = [asked_gran] + [g for g in order if g != asked_gran]
        result = None
        for gran in order:
            candidate_rows = retriever.get_series(match.indicator_detail_id, published_detail_id,
                                                   gran, country_en=single_country)
            candidate = compute.direction_assessment(
                candidate_rows, gran, match.polarity_en, unit=unit,
                period_label=single_period_label(period.raw or user_message))
            result = result or candidate
            if candidate.ok:
                result, rows, granularity = candidate, candidate_rows, gran
                break
        used = [_find_row_by_period(rows, result.facts.get("period_label"))] if result.ok else []
        return (_wrap(result, unit, indicator_name=indicator_name, decimals=decimals),
                citations_for_rows([r for r in used if r], indicator_name, data_source_en))

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
            # A position on its own says very little. "Qatar ranked 11th" is a
            # fact; "11th, from 9th the year before" is the answer to how it is
            # doing, and the reading before is published, not derived. Only for
            # rankings, where a bare ordinal is least informative and no
            # period-on-period figure is ever published.
            if compute._is_rank(unit):
                actuals = [r for r in rows if r.get("actual") is not None]
                if row in actuals and actuals.index(row) > 0:
                    previous = actuals[actuals.index(row) - 1]
                    result.facts["previous_value"] = previous["actual"]
                    result.facts["previous_period"] = previous["period_label"]
                    used.append(previous)
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(used, indicator_name, data_source_en)

    if ctype == "period_ranking":
        # extremum "min" means lowest-first; anything else (including the usual
        # "highest to lowest" phrasing, and the null default) means highest-first.
        descending = intent.get("extremum") != "min"
        result = compute.period_ranking(rows, descending=descending,
                                         limit=_clean_limit(intent.get("limit")))
        # Cite every row that appears in the ranking, not just the winner —
        # each listed figure is a claim of its own.
        return _wrap(result, unit, indicator_name=indicator_name, decimals=decimals), citations_for_rows(rows, indicator_name, data_source_en)

    if ctype == "trend":
        # The unit decides whether the move over the span is a percentage, a
        # percentage-point difference or a number of places. Without it the
        # series summary divided a rate by a rate and called Inflation's
        # -1.84% -> 2.62% "a decrease of 241.57%".
        result = compute.trend(rows, unit=unit)
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
            state = session_state or {}
            # Prefer an endpoint of the previous comparison that is NOT the one
            # just named; fall back to the single period the last answer
            # reported. "compare it with Q4 2025" after "2024-Q4 vs 2025-Q4"
            # means the other end, not a period the user has to restate.
            candidates = [p for p in (state.get("last_periods_used") or []) if p != named]
            prior = candidates[-1] if candidates else state.get("last_period_used")
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


def _keep_named_periods(rows: list[dict], period_labels) -> list[dict]:
    """Narrows a window to the periods the question actually listed.

    "لسنه 2021 و 2023" was answered with 2021, 2022 AND 2023. The period parser
    reads two years joined by "and" as a SPAN, which is right for "2024 and
    2025" — adjacent, so span and list agree — and wrong the moment they are not
    adjacent. The user named two years and was shown three.

    Nothing here re-reads the question. The intent agent already reports the
    periods it found, canonically, in period_labels, and that field was consumed
    in exactly one branch — period_comparison — while every other route fell
    back to the date window and ignored it. This is the field being used where
    it was already being filled.

    A label covers the periods INSIDE it, so "2021" keeps 2021-Q1 through
    2021-Q4: the user names the year, the series answers in quarters, and the
    granularity was decided elsewhere.

    Applied only when it leaves something behind. A label finer than the rows —
    "2026-Q1" against a yearly series — matches nothing, and an empty window
    here would report an absence that is really a mismatch of grain.
    """
    labels = [str(lab).strip().upper() for lab in (period_labels or []) if str(lab).strip()]
    if len(labels) < 2 or not rows:
        return rows
    kept = [r for r in rows
            if any((r.get("period_label") or "").upper() == lab
                   or (r.get("period_label") or "").upper().startswith(f"{lab}-")
                   for lab in labels)]
    return kept or rows


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


def _best_indicator_probe(phrases: list, language: str = "en"):
    """The strongest resolution among several wordings of the same question.

    Returns (result, phrase) for whichever scored highest, or (None, first).

    Exists because a routing decision that probes ONE phrasing inherits every
    wobble in how that phrasing was produced. The intent agent paraphrases the
    question, and it sees the transcript while doing so, which makes its
    paraphrase depend on what was asked earlier — so a test built on it gives
    the same question different answers in different sessions.

    Duplicates and blanks are skipped, so the common case where the model simply
    echoed the message costs one lookup, not two.
    """
    best, best_phrase = None, (phrases[0] if phrases else "")
    seen = set()
    for phrase in phrases:
        text = (phrase or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result = resolve_indicator(text, require_data=True, language=language)
        if result.status != "resolved" or not result.match:
            continue
        if best is None or result.match.confidence > best.match.confidence:
            best, best_phrase = result, text
    return best, best_phrase


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
_WORST_FIRST = re.compile(
    r"\b(worst|weakest|lowest|least|bottom|behind|lagging|furthest|"
    r"underperform\w*)\b|أسوأ|أدنى|متأخر",
    re.IGNORECASE)

_BEST_FIRST = re.compile(
    r"\b(best|top|strongest|highest|leading|closest|most)\b|أفضل|أعلى",
    re.IGNORECASE)
def _asks_for_worst(text: str) -> bool:
    return bool(text and _WORST_FIRST.search(text))


_RISING_ONLY = re.compile(
    r"\b(ris\w*|rose|increas\w*|grow\w*|up|upward|improv\w*|climb\w*|gain\w*)\b"
    r"|ترتفع|يرتفع|ارتفع|تزايد|تحسن",
    re.IGNORECASE)

_FALLING_ONLY = re.compile(
    r"\b(fall\w*|fell|declin\w*|decreas\w*|drop\w*|down|downward|"
    r"worsen\w*|backslid\w*|shrink\w*|deteriorat\w*)\b|\bwrong way\b"
    r"|تنخفض|ينخفض|انخفض|تراجع|تدهور",
    re.IGNORECASE)

# A question that asks for both halves, in a phrasing where only one of them is
# spelled out as a direction word. "What is improving and what is not" names
# only "improving", and reading it as a one-sided question would answer half of
# a question that plainly asked for both.
_BOTH_SIDES = re.compile(
    r"\b(vs\.?|versus)\b|\band\s+(what|which|who|the ones)\b"
    r"|\bor\s+(fall\w*|declin\w*|decreas\w*|down)\b"
    r"|مقابل|وأيها|وما\s+لا",
    re.IGNORECASE)


def _requested_direction(text: str) -> Optional[str]:
    """Which half of a direction split the question asked for, or None.

    None when BOTH are named — "which got better and which got worse" is a
    two-sided question and answering it one-sided would be the same fault in
    reverse — and None when neither is, which leaves the two-sided answer as
    the default it has always been.
    """
    if not text or _BOTH_SIDES.search(text):
        return None
    up, down = bool(_RISING_ONLY.search(text)), bool(_FALLING_ONLY.search(text))
    if up == down:
        return None
    return "up" if up else "down"


def _requested_order(text: str, default: bool = True) -> bool:
    """Which end of a ranking to put first, as a boolean best_first.

    The default matters: a bare "just give me top 3" following a worst-first
    ranking would otherwise silently flip the list over, and the user would be
    reading the opposite of what they just asked for under the same heading.
    """
    if text and _WORST_FIRST.search(text):
        return False
    if text and _BEST_FIRST.search(text):
        return True
    return default


def _clean_limit(value) -> Optional[int]:
    """How many entries to show, as stated by the intent agent.

    Reading "just give me top 3" is a language question — the ways to ask for a
    shorter list are open-ended ("kindly shorten that", "top 3 pls", "cut it
    down to three") and a word list that decides whether the question gets
    answered at all will always be missing one of them. The model reads it; this
    only checks the answer is a number in a sane range, because everything
    crossing that boundary is checked.

    A limit is safe to take from the model in a way an indicator name is not:
    getting it wrong shows five rows instead of three, and the answer says it
    was shortened. Getting an indicator wrong silently answers a different
    question.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 50 else None


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


def _round_facts(value, decimals=None):
    """round_for_display over a whole facts tree.

    Applied in the same place as _strip_padding, for the same reason: the
    frontend renders payload values into its own tiles, so rounding only the
    sentence would leave "166.107182" on the card beside it.

    A dict carrying its own decimal_places — one line of a multi-indicator
    answer — is rounded to ITS precision rather than the answer's, and
    decimal_places itself is never rounded, since it is a count of digits and
    not a figure.
    """
    if isinstance(value, dict):
        own = value.get("decimal_places", decimals)
        return {k: (v if k == "decimal_places" else _round_facts(v, own))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_round_facts(v, decimals) for v in value]
    return round_for_display(value, decimals)


def _strip_padding(value):
    """trim_decimal applied to a whole facts tree.

    Done in one place rather than at each computation so that no call site can
    forget it. The frontend renders payload values into its own tiles, so a
    Decimal still carrying its stored scale showed up as "+3.680000 billion
    QAR" beside prose that correctly said 3.68 — the text had been fixed and
    the payload had not.
    """
    if isinstance(value, dict):
        return {k: _strip_padding(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_padding(v) for v in value]
    return trim_decimal(value)


def _wrap(result: compute.ComputeResult, unit: str, extra_note: Optional[str] = None,
          indicator_name: Optional[str] = None, decimals: Optional[int] = None) -> dict:
    payload = {"ok": result.ok,
               "facts": {**_round_facts(_strip_padding(result.facts), decimals), "unit": unit}} \
        if result.ok else {"ok": False, "message": result.message}
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


# Commas and "and" were the only separators, so "compare the diversification of
# exports WITH the diversification of government revenues" was never seen as two
# questions and went to the resolver as one long phrase.
#
# Safe despite "Domestic vs External Debt" and "Revenue vs Expenditure" being
# real catalogue names: the split only runs when the phrase does NOT resemble a
# catalogue name, and both of those do.
_SPLIT_METRICS = re.compile(
    r"\s*(?:,|;|\band\b|\bcompared\s+(?:with|to)\b|\bwith\b|\bversus\b|\bvs\.?\b"
    r"|\bagainst\b|\bو\s|\bمقابل\b|\bمقارنة\s*ب)\s*",
    re.IGNORECASE)


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


_IMPROVING = re.compile(
    r"\b(improv\w*|deteriorat\w*|worsen\w*|getting\s+(better|worse)|"
    r"better\s+or\s+worse|on\s+the\s+right\s+track|healthier)\b"
    r"|يتحسن|تتحسن|يتدهور|تتدهور|يسوء|نحو\s+الأفضل",
    re.IGNORECASE)


# "what about if it was 29.44474" — a follow-up that supplies only a new
# figure. It names no indicator, asks for no share, and matches none of the
# complement wording, so it fell through to a plain latest-value lookup and
# answered about Q4 2025: the same question asked twice, answered two ways
# depending on whether it opened a session or continued one.
_FIGURE_FOLLOWUP = re.compile(
    r"^\W*(what|how)\s+about\b|^\W*and\s+(if|for)\b|\bif\s+it\s+(was|were)\b"
    r"|^\W*what\s+if\b|ماذا\s+(لو|عن)|وماذا\s+لو",
    re.IGNORECASE)


def _is_figure_followup(text: str) -> bool:
    """A follow-up whose only new content is a number.

    Deliberately narrow: it must open like a follow-up AND carry a figure AND
    name nothing else identifiable, so "what about inflation in 2024" stays a
    question about inflation rather than inheriting the previous computation.
    """
    if not text or not _FIGURE_FOLLOWUP.search(text):
        return False
    if not _QUOTED_ANY_FIGURE.search(text):
        return False
    return not has_identifying_content(text)


# Any number, with or without a percent sign. _QUOTED_FIGURE requires the "%"
# because it is reading a share out of a sentence; a bare follow-up often
# drops it.
_QUOTED_ANY_FIGURE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


_COMPLEMENT_ASK = re.compile(
    r"\bwhat\s+share\s+(still\s+)?(comes|come|is)\b"
    r"|\b(the\s+)?(rest|remainder|remaining|balance)\b"
    r"|\bthe\s+other\s+(share|portion|part)\b"
    r"|\bmakes?\s+up\s+the\s+rest\b"
    r"|\bhow\s+much\s+(is\s+)?left\b"
    r"|\bالباقي|\bالمتبقي|النسبة\s*المتبقية",
    re.IGNORECASE)


def _asks_for_complement_share(text: str) -> bool:
    """"...what share still comes from hydrocarbon exports?"

    A derived figure, which this system does not normally produce. It is
    allowed here because it is one subtraction over a share whose whole is
    split in two, checked in compute.complement_of_share — which refuses for
    every indicator that is not that shape, and only one in the catalogue is.
    """
    return bool(text and _COMPLEMENT_ASK.search(text))


def _asks_if_improving(text: str) -> bool:
    """"Is X improving or deteriorating?" — a question about direction of travel.

    Distinct from a trend, which describes a shape over a span, and from a
    latest value, which describes a level. This one asks whether the movement
    is the WELCOME one, which only polarity_en can answer.
    """
    return bool(text and _IMPROVING.search(text))


# A figure the user quoted back at us, e.g. "the 13.4% share of ...".
_QUOTED_FIGURE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")


# "how much ... was generated" asks for an AMOUNT; "what share/percentage"
# asks for a proportion. The catalogue holds both for the same subject — three
# non-hydrocarbon government revenue indicators, one in riyals and two as
# shares — so the wording separates them without troubling the user.
_ASKS_AMOUNT = re.compile(
    r"\bhow much\b|\bhow many\b|\bwas generated\b|\bwas raised\b|\bwas collected\b"
    r"|\bvalue of\b|\bamount of\b|\btotal\b|كم\s|قيمة|مبلغ",
    re.IGNORECASE)
_ASKS_SHARE = re.compile(
    r"\bshare\b|\bpercent\w*\b|\bproportion\b|\bas a %\b|%|نسبة|حصة"
    # "How much OF Qatar's exports are non-oil" is partitive: it asks for a
    # part of a whole, which is a share, even though it opens with "how much".
    # Without this the amount rule fired and would have picked total exports
    # in riyals to answer a question about a percentage.
    r"|\bhow much of\b|\bwhat (?:share|part|portion) of\b|\bكم من\b",
    re.IGNORECASE)


def _disambiguate_by_asked_unit(candidates, user_message: str):
    """Picks between candidates that differ by what they MEASURE.

    "How much non-hydrocarbon government revenue was generated in Q4 2025"
    offered three indicators: one in riyals and two expressed as shares. The
    question asks how much, so only the amount can answer it — and asking the
    user to choose between a figure and two percentages, when they have already
    said which they want, is the system not reading the question.

    Decides only when exactly one candidate is left.
    """
    text = user_message or ""
    wants_amount = bool(_ASKS_AMOUNT.search(text)) and not _ASKS_SHARE.search(text)
    wants_share = bool(_ASKS_SHARE.search(text)) and not _ASKS_AMOUNT.search(text)
    if not (wants_amount or wants_share):
        return None
    shares = [c for c in candidates if (c.unit_en or "").strip() == "%"]
    amounts = [c for c in candidates if (c.unit_en or "").strip() != "%"]
    wanted = amounts if wants_amount else shares
    return wanted[0] if len(wanted) == 1 else None


def _row_matching_quoted_value(rows, user_message: str):
    """The reading a quoted figure names, when it names exactly one.

    The counterpart of _disambiguate_by_quoted_value, which uses a quoted
    figure to pick an INDICATOR. This uses one to pick a PERIOD: a reader
    quoting 33.41532% is quoting a specific row, not proposing a hypothesis.

    Matched at the precision written, so "38.6%" matches 38.589 and "33%"
    matches nothing in particular — it would match several rows and is
    therefore ignored.
    """
    # Percent-signed figures first, since they are the unambiguous ones; then
    # bare numbers, because a follow-up says "what about if it was 29.44474"
    # without repeating the sign.
    quoted = _QUOTED_FIGURE.findall(user_message or "")
    quoted += [m for m in _QUOTED_ANY_FIGURE.findall(user_message or "")
               if m not in quoted]
    actuals = [r for r in rows if r.get("actual") is not None]
    for text in quoted:
        value = float(text)
        places = len(text.split(".")[1]) if "." in text else 0
        hits = [r for r in actuals if round(float(r["actual"]), places) == value]
        if len(hits) == 1:
            return hits[0]
    return None


def _disambiguate_by_quoted_value(candidates, user_message: str):
    """Picks the candidate whose latest reading is the figure the user quoted.

    Only decides when exactly ONE candidate matches. Two candidates that both
    round to the quoted figure is a real ambiguity and still gets the question;
    no match at all means the figure came from somewhere else and proves
    nothing.

    Matched at the precision the user wrote, so "13.4%" selects a reading of
    13.36014 and would not select 13.9. That is the whole signal: they are
    quoting a number the system published.
    """
    quoted = [float(m) for m in _QUOTED_FIGURE.findall(user_message or "")]
    if not quoted or not candidates:
        return None

    matches = []
    for candidate in candidates:
        published_id = candidate.published_detail_id if candidate.is_published else None
        gran = choose_granularity(None, retriever.get_available_granularities(
            published_id, candidate.indicator_detail_id))
        if not gran:
            continue
        rows = retriever.get_series(candidate.indicator_detail_id, published_id, gran)
        latest = compute.latest_value(rows)
        if not latest.ok or latest.facts.get("actual") is None:
            continue
        actual = float(latest.facts["actual"])
        for value in quoted:
            places = len(str(value).split(".")[1]) if "." in str(value) else 0
            if round(actual, places) == value:
                matches.append(candidate)
                break
    return matches[0] if len(matches) == 1 else None


def _asks_for_growth(phrase: str) -> bool:
    return bool(re.search(r"\b(growth|change|rate of change)\b|نمو|تغير", phrase or "",
                          re.IGNORECASE))


def _movement_verdict(change, polarity) -> Optional[str]:
    """Whether a move is the welcome one, decided from SCAI's own polarity.

    "improved" / "worsened" rather than "good" / "bad": the Council records a
    desired direction per indicator, and this reports that record against the
    sign of the change. It asserts nothing beyond the two — a rise in Inflation
    has worsened because polarity_en says Decrease, not because rising prices
    sound bad.

    Plain words on purpose. These are the labels the reader ends up seeing,
    because the Composer echoes them, and "an adverse movement" appended to
    four consecutive lines is the register of a compliance notice rather than
    an answer. "Improved" and "worsened" say the same thing in words a
    policymaker reads without stopping, which is rule 9's whole point.

    Returns None where there is nothing to say: no change figure, or no
    polarity recorded. A missing verdict has to stay missing. Defaulting to
    "improved" for an unpolarised indicator would be the same invented
    judgement this exists to remove, just with the blame moved into code.
    """
    if change is None or not polarity:
        return None
    try:
        change = float(change)
    except (TypeError, ValueError):
        return None
    if change == 0:
        return "unchanged"
    wants_lower = str(polarity).strip().lower().startswith("decrease")
    return "improved" if (change < 0) == wants_lower else "worsened"


def _indicator_snapshot(phrases: list[str], language: str = "en", period=None):
    """One latest reading per named indicator, each with its OWN period.

    Different indicators report on different schedules, so a snapshot that
    forces them onto one period would either drop the fresher ones or imply a
    currency the data does not have. Each line carries the period it came from,
    and nothing historical is added — a comparison the user did not ask for is
    a comparison the user did not ask for.
    """
    facts, citations, missing, out_of_range = [], [], [], []
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
            # Named and resolved, but nothing in the window asked for. The
            # comment here used to say this was "reported separately below" and
            # it was not — it went into the same list as a phrase that matched
            # no indicator at all. Those are different facts and the difference
            # matters: "Trade Balance has no Q1 2026 reading, its series ends at
            # 2025-Q4" is useful, "Trade Balance was not found" is wrong.
            covers = retriever.get_series(match.indicator_detail_id, published_id, gran)
            out_of_range.append({
                "indicator": match.display_name(language),
                "asked_as": phrase,
                "covers_from": covers[0]["period_label"] if covers else None,
                "covers_to": covers[-1]["period_label"] if covers else None,
            })
            continue
        row = _find_row_by_period(rows, latest.facts.get("period_label"))
        entry = {"indicator": match.display_name(language),
                 "unit": display_unit(match.unit_en, match.format, language, match.unit_ar),
                 "decimal_places": decimals_from_format(match.format),
                 # Which way is WELCOME for this indicator, as SCAI records it.
                 # Every other group shape carries this — scope_direction and
                 # the target ranking both put polarity on each row, and the
                 # Composer's rules 16 and 18 are written around it — but the
                 # snapshot did not, so the macro overview was the one answer
                 # with no way to tell a welcome move from an unwelcome one.
                 # Asked "is Qatar's economy performing well?", it had four
                 # changes, no idea what any of them meant, and answered "Yes,
                 # performing well" over a rising inflation rate and two falling
                 # balances.
                 "polarity": (match.polarity_en or "").strip() or None,
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
                entry["change_kind"] = "percent"
            else:
                # Rate indicators leave the percent column empty and publish the
                # move in percentage POINTS instead — a percent-of-a-percent
                # would have called Inflation's 0.63% -> 2.62% a rise of 316%.
                # Every other group shape already falls back here
                # (scope_snapshot, scope_direction); this one did not, so the
                # macro overview showed Inflation with no year-on-year figure at
                # all while the figure sat in the next column of the same row.
                change = compute.preferred_change_field(row, gran, as_points=True)
                if change is not None:
                    entry["change_yoy_pp"] = round(float(change), 4)
                    entry["change_kind"] = "percentage_points"
            entry["report_as_growth"] = _asks_for_growth(phrase)
            # Polarity and a direction combined ONCE, here, instead of at the
            # far end of a prompt. The Composer is good at restating a word and
            # unreliable at deriving one, and this derivation has a trap in it:
            # for a Decrease indicator the welcome move is the fall, so the
            # naive reading of a minus sign is backwards exactly where it
            # matters most. Naming the answer is the same technique rule 20's
            # "assessment" already uses, for the same reason.
            entry["direction_assessment"] = _movement_verdict(
                entry.get("change_yoy_percent", entry.get("change_yoy_pp")),
                entry.get("polarity"))
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
    # Each line carries its own decimal_places, so _round_facts rounds each to
    # its own precision rather than to one shared figure.
    payload = {"ok": True, "facts": {"overview": _round_facts(facts)}}
    # The window the question named, stated in the payload.
    #
    # Each line already carries its own period, and that was assumed to be
    # enough. It is not: asked "what about in 2023" after a macro overview, the
    # Composer had four lines whose periods were not 2023, no statement of what
    # had been asked for, and no way to say the two did not match — so it wrote
    # "Real GDP: 181.49 Bn QAR in 2024-Q4, which is the closest available
    # figure" over the year-earlier column, and "No data available for 2023"
    # against a line whose series covers 2023 perfectly well. Both invented.
    #
    # Composer rule 19 is written around this key and fires once it is present:
    # say which window was used, because a figure for the wrong period reads
    # exactly like a figure for the right one. The scope routes have set it
    # since that rule was added; the snapshot every macro and multi-indicator
    # answer is built from never did.
    asked_period = getattr(period, "raw", None) if period is not None else None
    if asked_period:
        payload["facts"]["as_of"] = asked_period
    # Ranking by movement, so "which experienced the largest decline?" can be
    # answered without the Composer ordering figures itself — the exact class
    # of work the QC report found it getting wrong (F-009). Only over the lines
    # that actually have a change; the ones that do not are reported, never
    # ranked as though they were flat.
    # Comparing the LEVELS, when they are in the same unit and so comparable at
    # all. change_ranking orders by movement; this answers "which is more X",
    # which is a different question and had no answer: "compare the
    # diversification of exports with that of government revenues" needs
    # 38.589% against 13.36014%, not their growth rates.
    comparable = [e for e in facts
                  if e.get("actual") is not None and (e.get("unit") or "").strip()]
    units = {(e.get("unit") or "").strip() for e in comparable}
    if len(comparable) > 1 and len(units) == 1:
        ordered = sorted(comparable, key=lambda e: float(e["actual"]), reverse=True)
        unit = units.pop()
        gap = round(float(ordered[0]["actual"]) - float(ordered[-1]["actual"]), 4)
        periods = [e.get("period_label") for e in ordered]
        payload_levels = {
            "ranked_by_level": [
                {"indicator": e["indicator"], "actual": e["actual"],
                 "period_label": e.get("period_label"), "unit": unit}
                for e in ordered
            ],
            "difference": gap,
            # A gap between two percentages is percentage POINTS, not a percent.
            "difference_kind": "percentage_points" if unit == "%" else "absolute",
            "unit": unit,
            # QC's point, and a fair one: Q4 2025 against Q1 2026 is not a
            # matched-period comparison, and an answer that does not say so
            # implies a precision it does not have.
            "periods_differ": len(set(periods)) > 1,
            "periods_compared": periods,
        }
        facts_out = payload["facts"]
        facts_out.update(payload_levels)

    changed = [e for e in facts if e.get("change_yoy_percent") is not None]
    if len(changed) > 1:
        payload["facts"]["change_ranking"] = [
            {"indicator": e["indicator"], "change_yoy_percent": e["change_yoy_percent"],
             "period_label": e.get("period_label")}
            for e in sorted(changed, key=lambda e: e["change_yoy_percent"])
        ]
    if out_of_range:
        payload["facts"]["no_data_in_period"] = out_of_range
    if missing:
        # Named but not found — said out loud rather than quietly dropped from
        # a list the user can see is short.
        payload["facts"]["not_found"] = missing
    return payload, citations


# The standard macro snapshot. An editorial list — refine it once SCAI says
# which indicators they want surfaced as the headline set.
HEADLINE_NAMES = ["Real GDP", "Inflation", "Trade Balance", "Government Revenues"]


# "Is Qatar becoming less dependent on oil and gas?" is a real question about
# this dataset and was refused, because no single indicator is named and the
# macro snapshot answers a different one — GDP, inflation, trade balance and
# government revenues say nothing about hydrocarbon dependence.
#
# The three that do are curated, the same way HEADLINE_NAMES is. They are the
# level of non-hydrocarbon output, and the non-hydrocarbon share of each of the
# two things Qatar earns from: exports and government revenue.
DIVERSIFICATION_NAMES = [
    "Non-Hydrocarbon Real GDP",
    "Non-Hydrocarbon Exports (share of total exports)",
    "Non-Hydrocarbon Government Revenues as Share of Government Revenues",
]

_DIVERSIFICATION = re.compile(
    r"\bdiversif\w*\b"
    r"|\b(less|reduc\w*|reducing|decreasing)\s+(its\s+)?depend\w*\b"
    r"|\bdepend\w*\s+(on|upon)\s+(oil|gas|hydrocarbon\w*|energy)\b"
    r"|\baway\s+from\s+(oil|gas|hydrocarbon\w*)\b"
    r"|\bnon[-\s]?(oil|hydrocarbon)\s+econom\w*\b"
    # Bare stems, no definite article: Arabic attaches prefixes, and requiring
    # "الاعتماد" missed "هل يقل اعتماد قطر على النفط".
    # Both verbal nouns: تنويع (diversifying something) and تنوع (being
    # diversified). A question uses whichever fits its grammar.
    r"|تنويع|تنوع|اعتماد\s+\w*\s*على\s+(النفط|الغاز)",
    re.IGNORECASE)

# An explicit comparison of two named things is answered as a comparison, not
# as a curated snapshot. "Compare the diversification of exports with the
# diversification of government revenues" names both sides and asks which is
# higher; replacing that with a three-indicator overview answers a question
# nobody asked.
_COMPARES_TWO = re.compile(r"\bcompare[sd]?\b|\bwhich\s+is\s+more\b|مقارنة\b|\bقارن\b",
                            re.IGNORECASE)


# A calculation the user has already done and is asking us to confirm. The QC
# workbook calls this the critical guardrail case, and it is: "126.6 thousand
# economically active Qataris, 8% of employed Qataris work in the private
# sector — does that mean about 10,100?" The arithmetic is fine and the answer
# is still no, because the 8% is measured over EMPLOYED Qataris while the
# 126.6k counts ECONOMICALLY ACTIVE ones, a population that also includes
# people who are not employed.
#
# Refusing it as out-of-scope was safe but unhelpful: it neither confirmed nor
# explained, so a reader could reasonably conclude the sum was right and the
# system merely unwilling to say so.
_PROPOSES_CALC = re.compile(
    r"\bdoes\s+that\s+mean\b|\bso\s+that\s+means\b|\bthat\s+would\s+(mean|be)\b"
    r"|\bwhich\s+implies\b|\bimplying\b|\btherefore\s+about\b|\bso\s+about\b"
    r"|\bcan\s+i\s+(just\s+)?multiply\b|\bworks?\s+out\s+(to|at)\b"
    r"|هل\s+يعني\s+ذلك|أي\s+أن\s+ذلك\s+يعني",
    re.IGNORECASE)

# The population or total a share is measured over, read from its own name.
_MEASURED_OVER = re.compile(
    r"(?:as\s+(?:a\s+)?(?:share|percentage|proportion)\s+of|out\s+of(?:\s+the)?"
    r"|%\s*of|\bshare\s+of)\s+(?P<base>.+?)\s*\.?$",
    re.IGNORECASE)


def _measured_over(name: str) -> str:
    """What this indicator is a share OF, or the indicator itself when it is a
    level rather than a share."""
    match = _MEASURED_OVER.search((name or "").strip())
    return match.group("base").strip() if match else (name or "").strip()


def _proposes_derived_figure(text: str) -> bool:
    return bool(text and _PROPOSES_CALC.search(text))


def _proposes_computed_figure(text: str) -> bool:
    """A calculation proposal that STATES the figure it arrived at.

    Stricter than _proposes_derived_figure, because this one is allowed to
    intercept a question the router has already routed somewhere. The wording
    alone is not enough: "does that mean inflation is above target?" matches the
    trigger and proposes no arithmetic at all — it asks for a comparison, and
    answering it with a lecture about populations would be its own failure.

    What separates the two is a number the USER worked out. "Does that mean
    about 10,100 Qataris worked in the private sector?" carries one, and it is
    the thing being asked about.
    """
    if not text:
        return False
    match = _PROPOSES_CALC.search(text)
    if not match:
        return False
    return bool(re.search(r"(?<![\w.])\d[\d,]*\.?\d*", text[match.end():]))


def _resolved_indicator_names(user_message: str, language: str = "en") -> list[str]:
    """Up to two catalogue names the message actually names.

    Shared by both halves of the denominator refusal, which each need the same
    walk for a different reason: one to compare what the two are measured over,
    the other to remember what it just offered.
    """
    names = []
    for part in split_indicator_phrases(user_message or ""):
        if not has_identifying_content(part):
            continue
        res = resolve_indicator(part, language=language)
        if res.status in ("resolved", "inactive") and res.match:
            name = res.match.name_en.strip()
            if name not in names:
                names.append(name)
        if len(names) == 2:
            break
    return names


def _different_bases_warning(user_message: str, language: str = "en"):
    """The refusal, but ONLY where the catalogue positively supports it.

    Returns None unless two indicators resolve AND are measured over different
    populations. That reticence is what makes it safe to run on a question the
    router has already assigned somewhere else: where the evidence is not there,
    the question carries on to the route it was given, and nothing is refused on
    a hunch.
    """
    names = _resolved_indicator_names(user_message, language)
    if len(names) < 2:
        return None
    (name_a, base_a), (name_b, base_b) = [(n, _measured_over(n)) for n in names]
    if base_a.lower() == base_b.lower():
        return None
    # The names come back with the message, because the message OFFERS them —
    # "I can give you either figure on its own" — and the caller has to remember
    # what it just offered for that sentence to mean anything on the next turn.
    return {"message": msg("derived_different_bases", language,
                            a=name_a, base_a=base_a, b=name_b, base_b=base_b),
            "indicators": [name_a, name_b]}


def _denominator_warning(user_message: str, language: str = "en"):
    """Explains WHY two published figures cannot simply be multiplied.

    Only says it when the catalogue supports it: both indicators have to
    resolve, and the things they are measured over have to be different. That
    difference is in their own names — "as Share of Total Qataris Employed"
    against "Qatari Nationals (Economically Active)" — so this reports the
    catalogue rather than reasoning about economics.
    """
    found = _different_bases_warning(user_message, language)
    if found:
        return found
    # The generic refusal still ENDS with an offer — "Ask me for either
    # indicator on its own and I'll give you the published value" — so whatever
    # did resolve is returned with it, even when there was only one or the two
    # shared a base. An offer whose subject the caller cannot record is the same
    # dead end "give me on its own" already fell into once.
    return {"message": msg("derived_not_supported", language),
            "indicators": _resolved_indicator_names(user_message, language)}


def _asks_about_diversification(text: str) -> bool:
    """Diversification away from hydrocarbons, as a subject in its own right.

    Kept separate from the group snapshot, which needs a scope that matches a
    catalogue name: "diversifying away from hydrocarbons" matches none, because
    the type is called "Economic Diversification Targets" and the word in the
    question is "diversifying".
    """
    if not text or not _DIVERSIFICATION.search(text):
        return False
    return not _COMPARES_TWO.search(text)


def _diversification_overview(language="en"):
    payload, citations = _indicator_snapshot(DIVERSIFICATION_NAMES, language=language)
    if payload.get("ok"):
        payload["facts"]["overview_kind"] = "diversification"
        # Names that fail to resolve are a problem with the editorial list
        # above, not something to report to a user who never named them.
        payload["facts"].pop("not_found", None)
    return payload, citations


def _macro_overview(language="en", period=None):
    """Fixes F-007/F-018: a deterministic curated overview instead of silently
    falling back to a single indicator.

    Delegates to _indicator_snapshot rather than repeating it. This function
    used to be its own older copy of the same loop, and had drifted: it read
    unit_en raw, so Real GDP printed as "185.17 QAR" instead of "185.17 Bn QAR"
    while the prose beside it said "QAR 185.2 billion"; and it carried no
    period-on-period change at all, so "how is the economy doing right now?"
    could only list four levels and call that an answer.
    """
    # The period the question named, where it named one. "Compare Qatar's
    # economy between 2023 and 2025" is a macro question with a window, and this
    # used to take none — so the only macro answer available was "right now",
    # whatever span was asked about. _indicator_snapshot already reads each
    # series at its latest reading INSIDE a window; it was simply never given
    # one from here.
    payload, citations = _indicator_snapshot(HEADLINE_NAMES, language=language, period=period)
    if payload.get("ok"):
        # Tells the Composer this is the "how is the economy doing" question
        # rather than an arbitrary set of metrics, so it reports the movement
        # rather than reading out four numbers.
        payload["facts"]["overview_kind"] = "macro"
        # Counted here so the opening sentence is a fact about the payload
        # rather than a verdict about the economy. "Is Qatar's economy
        # performing well?" is a yes/no question with no yes/no in the data:
        # four indicators can and do move in different directions at once, and
        # the honest opening is that the picture is mixed, with the split to
        # show it. Left to itself the Composer picked a side.
        verdicts = [e.get("direction_assessment")
                    for e in payload["facts"].get("overview") or []]
        n_better = verdicts.count("improved")
        n_worse = verdicts.count("worsened")
        payload["facts"]["n_improved"] = n_better
        payload["facts"]["n_worsened"] = n_worse
        payload["facts"]["mixed_signals"] = bool(n_better and n_worse)
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
            question: str = "", retries_left: int = 1) -> dict:
    # The period this answer actually reported, so "compare it with Q4 2025"
    # has the other half of its comparison. Recorded from the facts rather than
    # from the question, because the question may have named no period at all
    # and still been answered at a specific one.
    facts_out = payload.get("facts") or {}
    for key in ("period_label", "period_b", "period_end", "last_period", "period_used"):
        if facts_out.get(key):
            session_state["last_period_used"] = facts_out[key]
            break
    # BOTH endpoints when the answer was a comparison. Keeping only one made
    # "compare it with Q4 2025" fail after an answer about 2024-Q4 vs 2025-Q4:
    # the period named was the one already recorded, the "must be a different
    # period" guard rejected it, and the user was asked to supply two periods
    # they had just been shown.
    endpoints = [facts_out.get(k) for k in ("period_a", "period_b", "period_start", "period_end")]
    endpoints = [p for p in endpoints if p]
    if endpoints:
        session_state["last_periods_used"] = endpoints

    if skip_compose:
        answer = canned or ""
    else:
        _stage(session_state, "composing")
        compose_args = dict(question=question,
                                length=session_state.get("answer_length"),
                                # The last few exchanges, not just the previous
                                # one. See _render_history in composer_agent —
                                # a reference can reach two turns back, and a
                                # Composer shown only one restates what the
                                # reader has already had twice.
                                history=session_state.get("_history"),
                                # The exchange before this one, so a follow-up
                                # does not re-state what the reader has just
                                # read. Asked to shorten an answer about the
                                # inflation trend, the Composer gave the April
                                # 2023 peak and the January 2025 trough a second
                                # time — it could not know they had already been
                                # said, because it had never seen its own last
                                # answer. One turn, not the transcript: the
                                # prompt budget is shared with the facts, and
                                # only the immediately preceding turn is what
                                # "that" and "shorter" refer to.
                                previous_turn=session_state.get("last_exchange"))
        progress = session_state.get("_progress")
        if progress:
            # Streamed: the same prompt, the same model, the same temperature —
            # the only difference is that the text is released as it is written.
            # The gate holds each line until every number in it traces back to
            # the payload, so a listener never sees a figure that is then
            # withdrawn. See app/compute/streaming.py.
            draft, gate_rejected, gate_kind = compose_answer_streamed(
                payload, language, on_chunk=lambda text: _stage(
                    session_state, "delta", text=text), **compose_args)
        else:
            draft = compose_answer(payload, language, **compose_args)
            gate_rejected, gate_kind = None, None

        _stage(session_state, "verifying")
        clean, offending = verify_numbers(draft, payload)
        # Kept apart from the checks below, because the two failures are not
        # alike and must not be answered alike — see the retry.
        offending_numbers = list(offending)
        # The gate found an invented figure and stopped the stream before it was
        # shown. The full check below would reach the same verdict on the text
        # that was written, but not on the text that was RELEASED — the offending
        # line never made it into `draft`. Recorded explicitly so the outcome does
        # not depend on that coincidence.
        if gate_rejected:
            clean = False
            offending = list(offending) + [gate_rejected]
            # An invented figure counts as a numeric failure even though
            # verify_numbers above could not see it: the gate stopped the line
            # carrying it before it was released, so the draft the final check
            # looked at is clean. Without this the retry below read a rejection
            # for a bad number as a malformed draft and asked the model again,
            # which is exactly what it must not do.
            if gate_kind == "number":
                offending_numbers.append(gate_rejected)
        # An answer in the wrong language is not an answer. The model is asked
        # for Arabic and mostly complies, but "mostly" is not a guarantee and
        # the failure is total from the reader's side. Checked here rather than
        # hoped for, and the template it falls back to is bilingual, so the
        # reply is in the right language whichever path produces it.
        if clean and not answers_in_language(draft, language):
            clean = False
            payload["_wrong_language_draft"] = True
        # Whether the text is SANE, which nothing above asks. An answer went out
        # that restated itself eight times, drifted into Chinese and narrated its
        # own attempts to fix the language — and passed every check, because its
        # numbers were in the payload and it did contain Arabic.
        if clean:
            broken = looks_degenerate(draft)
            if broken:
                clean = False
                offending = list(offending) + [broken]
        # A sources footer the model wrote itself. Rule 10 forbids it and the
        # real one is appended below, so the reader would get two — and the
        # model's is the one that can be wrong.
        if clean and writes_own_sources(draft):
            clean = False
            offending = list(offending) + ["answer wrote its own sources footer"]
        # One retry, but only for a draft that came out MALFORMED — the wrong
        # language, a loop, a footer of its own. Those are the model losing its
        # footing on a prompt it can handle, and asking again generally gets a
        # clean answer; falling straight to the template turns a recoverable
        # stumble into a raw dump of payload fields at the reader.
        #
        # Never for a bad NUMBER. A figure that is not in the payload is the one
        # failure this system exists to catch, and a second attempt is just as
        # likely to invent a second figure. That falls to the template as before.
        if not clean and not offending_numbers and retries_left > 0:
            payload.pop("_wrong_language_draft", None)
            return _finish(payload, language, session_state, citations,
                            skip_compose=skip_compose, canned=canned,
                            question=question, retries_left=retries_left - 1)
        answer = draft if clean else render_template_fallback(payload, language)
        if not clean:
            payload["_verifier_rejected_numbers"] = offending  # for logging/debugging only
            if progress:
                # A listener has text on screen that is about to be superseded.
                # Told explicitly rather than left to infer it from the final
                # answer differing from the deltas: the frontend has to clear
                # what it has drawn, and working that out by comparing strings
                # is the kind of thing that is right in testing and wrong on the
                # one answer where it matters.
                _stage(session_state, "replace", reason="unverified")

    # Sources footer: always appended deterministically, never left to the
    # LLM's discretion. Only skipped when there's genuinely nothing to cite
    # (greetings, or a refusal where no data was retrieved at all).
    #
    # A weak match used to add a trailer here — "I matched your question to
    # "..." (approximate match)". Removed by request: it fired on every match
    # under CONFIDENT_MATCH, including the many that were simply long indicator
    # names matched correctly, and hedging a right answer teaches the reader to
    # ignore the hedge. The named cost of dropping it, recorded so the decision
    # can be revisited rather than rediscovered: "What percentage of Qatar's
    # energy is generated by solar?" once resolved to "Share of Qatari
    # Teachers" — enough lexical overlap to clear MIN_CONFIDENCE — and was
    # answered with that indicator's 2030 target of 50%. Nothing now signals
    # that. The answer for a mismatch like that one is a better resolver or a
    # higher bar to answer at all, not a caveat on the way out.
    #
    # _low_confidence_match is still SET, and deliberately left in the payload
    # rather than popped: CONFIDENT_MATCH is documented as provisional and
    # needing calibration against real bge-m3 scores, and the logged flag is
    # where those scores come from. _public() strips it before the Composer
    # sees it, so it is never narrated.
    # Kept for the NEXT turn's Composer, so a follow-up can be told what has
    # already been said. Stored before the footer is attached — the sources
    # block is rendered deterministically every time and repeating it into the
    # prompt would spend budget on text the Composer never writes. Capped for
    # the same reason MAX_STORED_TEXT caps the transcript: this exists to stop
    # repetition, not to be re-read in full.
    session_state["last_exchange"] = {"user": (question or "")[:300],
                                       "assistant": (answer or "")[:600]}

    # Language, so the footer does not end an Arabic answer in English — the
    # one path to the reader that never passes through the Composer, and so the
    # one the language check could not catch.
    # The answer WITHOUT the sources block, for the conversation store.
    #
    # last_exchange has always been stored pre-footer, for the reason written
    # above it. recent_turns then reintroduced the problem through a different
    # door: it serves what the store holds, the store held the full answer, and
    # the Composer was handed its own previous footer as an example of how an
    # answer looks. It copied it — an Arabic answer opened "المصدر:" and
    # reproduced the whole block, beside the real one, in breach of rule 10.
    #
    # Nothing is lost by storing the body alone. The footer is rendered
    # deterministically from the citations every turn, so it is never something a
    # later turn needs to recall, and as history it is pure noise: the same
    # boilerplate on every line, spending prompt budget to teach the model a
    # habit the prompt forbids.
    answer_body = answer

    footer = render_sources_footer(citations, language)
    if footer:
        answer = f"{answer}\n\n{footer}"

    payload["citations"] = citations_to_dicts(citations)

    # The transcript's own working memory, which handle_message put here to
    # avoid threading a parameter through twenty-five call sites. It is not part
    # of the session's state and must not be persisted into it: stored, it would
    # be fed back as history of a history on every later turn.
    session_state.pop("_history", None)
    # Same reasoning, and more pressing: a callback is not serialisable, and this
    # dict is copied into the session store and read back on every later turn.
    session_state.pop("_progress", None)

    return {
        "answer": answer,
        # What the conversation store should keep. The caller returns "answer"
        # to the user and records this — see the note beside answer_body.
        "answer_for_history": answer_body,
        "facts_payload": payload,
        "readable": is_readable(payload),
        "session_state": session_state,
        # What this turn RESOLVED to, for the transcript the next turn is shown.
        # Prose alone made the router re-derive from an English sentence what the
        # pipeline had already established — see ConversationStore._render_slots.
        "turn_slots": _turn_slots(payload, session_state),
        "language": language,
    }


# Which facts key names the subject, in the order they should be preferred.
def _turn_slots(payload: dict, session_state: dict) -> dict:
    """The resolved subject of this turn, as a small flat dict.

    Read from the PAYLOAD first and the session state second. The state holds
    what the question asked for; the payload holds what was actually answered,
    and where they differ the answer is the thing a follow-up refers back to —
    "what about the year before that" means the year before the one on screen,
    not the year before the one that was requested and had no reading.
    """
    facts = payload.get("facts") or {}
    period = next((facts.get(k) for k in
                   ("period_label", "period_b", "period_end", "last_period", "period_used")
                   if facts.get(k)), None)
    indicator = (facts.get("indicator") or session_state.get("last_indicator_name")
                 or (", ".join(session_state["last_metrics"])
                     if session_state.get("last_metrics") else None))
    scope = session_state.get("last_catalog_scope") or {}
    return {k: v for k, v in {
        "indicator": indicator,
        "period": period,
        "countries": session_state.get("last_countries"),
        "group": scope.get("scope") or session_state.get("last_country_group"),
        "computation": session_state.get("last_ctype"),
    }.items() if v}
