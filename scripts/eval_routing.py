"""
Measures which computation type each question actually gets, against the real
router model.

Why this exists: two bugs reached a user in the same session — "which one
preformed the best" and "just give me top 3" — and both were routing failures,
not data failures. Neither was detectable before a user hit it, because nothing
in this repo has ever measured routing. The intent agent proposes a type, the
patterns in decide_route override it, and until now the only way to learn that a
question routes wrongly was for someone to ask it.

That matters more since the intent prompt grew from 18 computation types to 25.
The prompt has lost a type before: count_list worked, then broke when two newer
types were added, because it was the only one with no explicit rule. There is no
reason to believe that cannot happen again, and no way to see it if it does.

Run:
    docker compose exec api python scripts/eval_routing.py

    # compare models (the router and the chat model are the same served name
    # today, so this is how you would test a smaller/faster router)
    docker compose exec api python scripts/eval_routing.py --model qwen72b

    # no database handy: measure only what the model proposes
    docker compose exec api python scripts/eval_routing.py --intent-only

Read the output like this:

  MODEL   is what the intent agent proposed, on its own.
  ROUTED  is what the application uses, after decide_route's patterns.

  The gap between those two columns is the whole argument about how much of the
  routing the model should own:

  - MODEL wrong, ROUTED right  -> a pattern is carrying that route. Removing the
    pattern would break it. This is the count_list situation.
  - MODEL right, ROUTED wrong  -> a pattern is OVERRIDING the model incorrectly.
    That pattern is now doing harm and should be narrowed or dropped.
  - both wrong                 -> the prompt needs a rule, or a Contrast line
    against whichever type it lost to. The MODEL column tells you which.
  - both right                 -> the pattern may be redundant. Worth deleting
    only if the model gets it right consistently, so re-run before believing it.

A case marked (guarded) is one where handle_message applies a further check
after routing — a group type over a message that resolves to a single indicator
falls back to that indicator. Those are expected to be corrected downstream, so
a ROUTED value of scope_* there is not necessarily a bug.

LIMIT reports whether the model extracted a count from the question. The top-N
follow-ups depend on it entirely; if that column is empty for them, the model is
not filling the field and those questions will refuse.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.nlu.intent_agent import extract_intent, COMPUTATION_TYPES

# (question, expected ctype or tuple of acceptable ones, context, state, note)
#
# The state dicts are what the previous turn would have left behind. A follow-up
# that carries none of it is a different question, so the ones that need it say
# so rather than being tested in a state they never occur in.
EDU = {"last_catalog_scope": {"kind": "sector", "scope": "Education Sector"}}
EDU_RANKED = {**EDU, "last_ctype": "scope_performance"}
LISTED = "User: give me all the indicator names for the education sector\nAssistant: There are 13 published indicators in the Education Sector."
RANKED = "User: which one performed the best\nAssistant: Education Sector, ranked by progress against each indicator's own target."

CASES = [
    # --- the two reported bugs -------------------------------------------------
    ("which one preformed the best", "scope_performance", LISTED, EDU, "reported bug 1"),
    ("which one performed the best", "scope_performance", LISTED, EDU, "reported bug 1"),
    ("just give me top 3", "scope_performance", RANKED, EDU_RANKED, "reported bug 2"),

    # --- top-N follow-ups: the phrasings the old word list missed --------------
    ("kindly give me top 3", "scope_performance", RANKED, EDU_RANKED, "needs limit"),
    ("top 3 pls", "scope_performance", RANKED, EDU_RANKED, "needs limit"),
    ("cut it down to the top 3", "scope_performance", RANKED, EDU_RANKED, "needs limit"),
    ("top 3 from the above", "scope_performance", RANKED, EDU_RANKED, "needs limit"),
    ("show me the worst 5 instead", "scope_performance", RANKED, EDU_RANKED, "needs limit+order"),

    # --- group routes the patterns could not reach -----------------------------
    ("which are ahead of target", "scope_performance", LISTED, EDU, ""),
    ("which ones are we falling short on", "scope_performance", LISTED, EDU, ""),
    ("rank the education sector indicators", "scope_performance", "", {}, ""),
    ("where are we succeeding in the education sector", "scope_performance", "", {}, ""),
    ("which education indicator is closest to what we promised", "scope_performance", "", {}, ""),
    ("which education indicators got better and which got worse", "scope_direction", "", {}, ""),
    ("what is improving and what is not in the education sector", "scope_direction", "", {}, ""),
    ("which education indicators are backsliding", "scope_direction", "", {}, ""),
    ("which ones are moving the wrong way", "scope_direction", LISTED, EDU, ""),
    ("give me the numbers for the education sector", "scope_snapshot", "", {}, ""),
    ("show me where the education sector stands", "scope_snapshot", "", {}, ""),
    ("what do the education indicators say", "scope_snapshot", "", {}, ""),

    # --- the other newly reachable types ---------------------------------------
    ("is inflation improving?", "direction_check", "", {}, ""),
    ("is the trade balance on the right track?", "direction_check", "", {}, ""),
    ("is the economy diversifying?", "diversification_overview", "", {}, ""),
    ("how dependent are we still on oil and gas?", "diversification_overview", "", {}, ""),
    ("if non-hydrocarbon exports account for 38.6%, what share still comes from hydrocarbons?",
     "complement_share", "", {}, ""),
    ("so that means about 40% of the workforce?", "denominator_check",
     "User: what is the share of Qataris employed\nAssistant: 12.3%.", {}, ""),

    # --- single indicator: must NOT be swallowed by a group route --------------
    ("is inflation rising?", "direction_check", "", {}, "guarded"),
    ("what is the latest value of Real GDP", "latest_value", "", {}, ""),
    ("what was inflation in May 2025?", "latest_value", "", {}, ""),
    ("what is inflation?", "definition", "", {}, ""),
    ("show me the trend of Real GDP", "trend", "", {}, ""),
    ("compare Real GDP in Q1 2025 and Q4 2025", "period_comparison", "", {}, ""),
    ("what is the growth rate of GDP between 2020 and 2024", "growth_rate", "", {}, ""),
    ("what was the highest quarterly GDP value recorded, and in which quarter?", "min_max", "", {}, ""),
    ("list the quarterly GDP values for 2024 and 2025 ranked highest to lowest",
     "period_ranking", "", {}, ""),
    ("what is the difference between the highest and lowest quarterly GDP", "difference", "", {}, ""),
    ("which country has the lowest inflation", "country_ranking", "", {}, ""),
    ("compare inflation between Qatar and Singapore", "country_comparison", "", {}, ""),
    ("show GDP growth, inflation, and government revenues", "multi_indicator", "", {}, ""),

    # --- catalogue vs group: the pair that broke once before -------------------
    ("give me all the indicator names for the education sector", "count_list", "", {}, ""),
    ("how many indicators are in the education sector", "count_list", "", {}, ""),
    ("list the indicators in Sectors", "count_list", "", {}, ""),

    # --- commentary and meta ---------------------------------------------------
    ("what has SCAI written about the trade war", "article_lookup", "", {}, ""),
    ("what is the latest analysis of inflation", "analysis_lookup", "", {}, ""),
    ("how is Qatar's economy doing right now?", "macro_overview", "", {}, ""),
    ("what can you do", "capabilities", "", {}, ""),
    ("hello", "general_chat", "", {}, ""),
    ("what is the weather in Doha", "out_of_scope", "", {}, ""),

    # --- Arabic ----------------------------------------------------------------
    ("سلام عليكم", "general_chat", "", {}, ""),
    ("ما هو التضخم في 2025", "latest_value", "", {}, ""),
    ("هل يتحسن التضخم؟", "direction_check", "", {}, ""),
    ("كم عدد المؤشرات في قطاع التعليم", "count_list", "", {}, ""),
    ("أيها الأفضل أداءً", "scope_performance", LISTED, EDU, ""),
]

# The questions that actually broke this system.
#
# Everything above is a phrasing someone thought of while fixing a bug, which is
# the wrong test for whether the patterns in decide_route can be removed: those
# patterns exist because real users typed things nobody anticipated. Each one
# carries its motivating question in a comment beside it, and these are those
# questions, harvested from the source.
#
# This is the set that decides whether a pattern is load-bearing. A pattern that
# catches nothing here is not insurance against the unknown — it is insurance
# against something that has already been shown not to need it.
TOURISTS = "User: How many tourists arrived in May 2025?\nAssistant: 142,000 in May 2025."
INFL = "User: What was inflation in May 2025?\nAssistant: 0.2% in May 2025."
SHARE = ("User: If non-hydrocarbon exports account for 38.6%, what share still comes from hydrocarbons?\n"
         "Assistant: 61.4%.")

HISTORICAL = [
    # --- greeting / economy patterns -------------------------------------------
    ("Is Qatar's economy growing?", "macro_overview", "", {}, "hist"),
    ("how is Qatar's economy doing", "macro_overview", "", {}, "hist"),
    ("how is the economy doing", "macro_overview", "", {}, "hist"),
    ("هل ينمو اقتصاد قطر؟", "macro_overview", "", {}, "hist"),
    # named a real indicator and was wrongly swallowed by the macro snapshot
    ("how is Qatar doing on competitiveness?", "latest_value", "", {}, "hist"),
    ("how is the education sector doing", ("scope_snapshot", "scope_performance"), "", {}, "hist"),
    ("how are prices doing", ("latest_value", "trend", "direction_check"), "", {}, "hist"),

    # --- diversification -------------------------------------------------------
    ("is the economy diversifying?", "diversification_overview", "", {}, "hist"),
    ("Is Qatar becoming less dependent on oil and gas?", "diversification_overview", "", {}, "hist"),
    ("هل يقل اعتماد قطر على النفط", "diversification_overview", "", {}, "hist"),

    # --- catalogue vs ranking --------------------------------------------------
    ("List the indicators in Sectors", "count_list", "", {}, "hist"),
    ("which are the best performing ones", "scope_performance", LISTED, EDU, "hist"),
    ("what are the top performing indicators", "scope_performance", LISTED, EDU, "hist"),
    ("what are the most performing ones", "scope_performance", LISTED, EDU, "hist"),
    ("which performed best?", "scope_performance", LISTED, EDU, "hist"),
    ("Which is the best performing one?", "scope_performance", LISTED, EDU, "hist"),
    ("which one is the best", "scope_performance", LISTED, EDU, "hist"),
    ("which indicator is the worst", "scope_performance", LISTED, EDU, "hist"),
    ("which ones are we failing at", "scope_performance", LISTED, EDU, "hist"),

    # --- direction split vs one indicator --------------------------------------
    ("which national indicators are rising", "scope_direction", "", {}, "hist"),
    ("which indicators are increasing?", "scope_direction", "", {}, "hist"),
    ("is inflation increasing?", "direction_check", "", {}, "hist"),
    ("Give me the latest snapshot of Qatar's economic diversification indicators.",
     "scope_snapshot", "", {}, "hist"),

    # --- follow-ups that lost the thread (F-030) -------------------------------
    ("and for Saudi Arabia?", ("latest_value", "country_comparison"), TOURISTS, {}, "hist"),
    ("what about in 2022", "latest_value", INFL, {}, "hist"),
    ("what about in 2024", "latest_value", INFL, {}, "hist"),
    ("what about if it was 29.44474", "complement_share", SHARE,
     {"last_ctype": "complement_share", "last_indicator_name": "Non-Hydrocarbon Exports"}, "hist"),

    # --- forecasts are not published, and must not be invented -----------------
    ("What is the GDP forecast 2026", "out_of_scope", "", {}, "hist"),
    ("What is the Real GDP forecast for 2026", "out_of_scope", "", {}, "hist"),

    # --- the rest --------------------------------------------------------------
    ("what has SCAI written about penguins", "article_lookup", "", {}, "hist"),
    ("What has SCAI written about inflation?", "article_lookup", "", {}, "hist"),
    ("ما هو أخر تحليل للتضخم", "analysis_lookup", "", {}, "hist"),
    ("Show GDP growth, inflation, and government revenues for the year 2023",
     "multi_indicator", "", {}, "hist"),
    ("How many GCC tourists arrived into Qatar in 2025?", "latest_value", "", {}, "hist"),
    ("How much OF Qatar's exports are non-oil", "latest_value", "", {}, "hist"),
    ("What does the 13.4% share of non-hydrocarbon government revenue mean",
     ("latest_value", "definition"), "", {}, "hist"),
    ("الربع الافتتاحي من 2026 مقابل ما يقابله في 2025", "period_comparison", "", {}, "hist"),
]

CASES = CASES + HISTORICAL


def _ok(actual, expected):
    return actual in (expected if isinstance(expected, tuple) else (expected,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.ROUTER_MODEL_NAME,
                    help="served model name to route with (default: ROUTER_MODEL_NAME)")
    ap.add_argument("--intent-only", action="store_true",
                    help="skip decide_route, which needs the database")
    args = ap.parse_args()

    # Swapping the model is the cheapest experiment available: it says whether a
    # routing miss is the prompt's fault or the model's.
    settings.ROUTER_MODEL_NAME = args.model
    decide_route = None
    if not args.intent_only:
        from app.core.graph_v2 import decide_route

    print(f"router model: {args.model}   cases: {len(CASES)}   types: {len(COMPUTATION_TYPES)}\n")
    header = f"{'':2} {'MODEL':<24} {'ROUTED':<24} {'EXPECTED':<24} {'LIM':<4} QUESTION"
    print(header)
    print("-" * len(header))

    misroutes, rescued, broken, stats = [], [], [], Counter()
    model_says = []
    for question, expected, context, state, note in CASES:
        try:
            intent = extract_intent(question, context)
        except Exception as exc:                      # a 404 on the model name, usually
            print(f"\nFAILED to reach the model: {type(exc).__name__}: {exc}")
            return 2
        model_ctype = intent.get("computation_type")
        model_says.append(model_ctype)
        limit = intent.get("limit")
        if decide_route:
            routed, _pinned = decide_route(question, dict(intent), dict(state))
        else:
            routed = model_ctype

        m_ok, r_ok = _ok(model_ctype, expected), _ok(routed, expected)
        stats["model_ok"] += m_ok
        stats["routed_ok"] += r_ok
        # The two sets answer different questions and are scored apart: the
        # written set says whether today's routing works, the harvested set says
        # whether the patterns are still needed.
        bucket = "hist" if note == "hist" else "written"
        stats[f"{bucket}_n"] += 1
        stats[f"{bucket}_model"] += m_ok
        stats[f"{bucket}_routed"] += r_ok
        if m_ok and not r_ok:
            broken.append((question, model_ctype, routed, expected))
        elif r_ok and not m_ok:
            rescued.append((question, model_ctype, routed, expected))
        elif not r_ok:
            misroutes.append((question, model_ctype, routed, expected, note))

        mark = "ok" if r_ok else ("~~" if note == "guarded" else "XX")
        exp = expected if isinstance(expected, str) else "|".join(expected)
        print(f"{mark:2} {str(model_ctype):<24} {str(routed):<24} {exp:<24} "
              f"{str(limit or ''):<4} {question[:44]}")

    n = len(CASES)
    print(f"\n{'='*70}")
    print(f"{'':<22}{'MODEL ALONE':>14}{'AFTER ROUTING':>16}")
    for label, key in (("written cases", "written"), ("questions that broke it", "hist")):
        k = stats[f"{key}_n"] or 1
        print(f"{label:<22}{stats[f'{key}_model']:>8}/{k:<5}{stats[f'{key}_routed']:>10}/{k}")
    print(f"{'TOTAL':<22}{stats['model_ok']:>8}/{n:<5}{stats['routed_ok']:>10}/{n}")
    if decide_route:
        print(f"\npatterns RESCUED {len(rescued)} case(s) the model got wrong:")
        for q, m, r, e in rescued:
            print(f"   {q[:50]:<52} model said {m}")
        print(f"\npatterns BROKE {len(broken)} case(s) the model got right:")
        for q, m, r, e in broken:
            print(f"   {q[:50]:<52} pattern forced {r}, wanted {e}")
    # --- which patterns are still carrying anything --------------------------
    #
    # decide_route's chain has precedence, so "this pattern matched" is not the
    # same as "this pattern decided". What can be said exactly, and is what the
    # deletion question turns on, is: of the questions this pattern matches, how
    # many did the model already route correctly on its own? A pattern that
    # matches nothing, or only ever agrees with a model that was already right,
    # is not protecting anything.
    if decide_route:
        from app.core import graph_v2 as g
        PATTERNS = [
            ("is_greeting", g.is_greeting, "general_chat"),
            ("_proposes_derived_figure", g._proposes_derived_figure, "denominator_check"),
            ("_asks_about_diversification", g._asks_about_diversification, "diversification_overview"),
            ("_asks_about_the_economy", g._asks_about_the_economy, "macro_overview"),
            # _asks_for_direction_split was deleted after this audit reported it
            # matching none of the 90 questions.
            ("_asks_what_to_watch", g._asks_what_to_watch, "scope_direction"),
            ("_asks_for_group_snapshot", g._asks_for_group_snapshot, "scope_snapshot"),
            ("_asks_for_performance_ranking", g._asks_for_performance_ranking, "scope_performance"),
            ("_looks_like_catalog_request", g._looks_like_catalog_request, "count_list"),
        ]
        print(f"\n{'='*70}\npattern audit — what each one is still doing\n")
        print(f"{'PATTERN':<32}{'MATCHES':>8}{'AGREES':>8}{'RESCUES':>8}{'HARMS':>7}  VERDICT")
        for name, fn, forces in PATTERNS:
            matched = agrees = rescues = harms = 0
            for (question, expected, _c, _s, _n), model_ctype in zip(CASES, model_says):
                try:
                    hit = bool(fn(question))
                except Exception:
                    hit = False
                if not hit:
                    continue
                matched += 1
                if model_ctype == forces:
                    agrees += 1
                elif _ok(forces, expected):
                    rescues += 1
                elif _ok(model_ctype, expected):
                    harms += 1
            verdict = ("no match — delete" if matched == 0
                       else "HARMS — fix or drop" if harms
                       else "load-bearing" if rescues
                       else "redundant here")
            print(f"{name:<32}{matched:>8}{agrees:>8}{rescues:>8}{harms:>7}  {verdict}")
        print("\n  MATCHES  questions this pattern fires on")
        print("  AGREES   ...where the model said the same thing anyway")
        print("  RESCUES  ...where the model was wrong and this pattern is right")
        print("  HARMS    ...where the model was RIGHT and this pattern overrides it")

    print(f"\nstill wrong after routing: {len(misroutes)}")
    for q, m, r, e, note in misroutes:
        tag = f" ({note})" if note else ""
        print(f"   {q[:50]:<52} got {r}, wanted {e}{tag}")

    # The top-N follow-ups cannot work without it, so it is called out separately
    # rather than left to be inferred from a routing number.
    print("\nlimit extraction on the top-N follow-ups: see the LIM column above; "
          "an empty LIM on those rows means they will refuse.")
    return 0 if not misroutes else 1


if __name__ == "__main__":
    sys.exit(main())
