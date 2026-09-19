"""
Intent Agent — the ONLY place an LLM call is allowed to touch a user's
numeric question before real data is retrieved. Its job is strictly
extraction: turn free text into structured fields. It never computes,
never estimates, never fills in a plausible-looking number.

This directly targets QC findings F-026 ("compare" mis-routed to a
country-ranking function) and F-005/006/022-025 (frequency and computation
type guessed wrong) — by making the LLM's output a narrow, validated schema
instead of free-form SQL or free-form reasoning.
"""
import json
from app.core.llm_client import chat, router_client
from app.core.config import settings

COMPUTATION_TYPES = [
    "latest_value",       # "what is the latest GDP?"
    "trend",              # "show me the trend of X" / "how has X changed over time"
    "period_comparison",  # "compare X in period A and period B" (same entity, two specific periods)
    "country_comparison", # "compare X between Qatar and Singapore" (named countries)
    "country_ranking",    # "which country has the lowest X" (no explicit period comparison)
    "period_ranking",     # "list X for 2024 and 2025 ranked highest to lowest"
    "min_max",            # "what was the highest/lowest value of X"
    "difference",         # "what is the difference between the highest and lowest X"
    "growth_rate",        # "what is the growth rate of X between A and B" (CAGR or simple)
    "definition",         # "what is inflation?" / "what does non-hydrocarbon GDP mean?"
    "analysis_lookup",    # "what is the latest analysis of inflation" / "ما هو أخر تحليل للتضخم"
    "article_lookup",     # "what has SCAI written about tariffs?" / "what is SCAI's view on X"
    "count_list",         # "how many indicators are in sector Y" / "list indicators in Z"
    "macro_overview",     # "what's the latest in Qatar's economy" / "how is the economy doing"
    "capabilities",       # "what can you do"
    "general_chat",       # greetings, small talk
    "out_of_scope",
]

SYSTEM_PROMPT = f"""You extract structured intent from a question about Qatari economic
data (SCAI). You NEVER answer the question, calculate anything, or state a figure.
Output ONLY JSON with this exact shape:

{{
  "computation_type": one of {COMPUTATION_TYPES},
  "indicator_phrase": "<the user's own words for the indicator/metric they mean, or null>",
  "countries_mentioned": ["<country name as the user wrote it>", ...] or [],
  "country_group_mentioned": "<e.g. 'GCC' if a group was named, else null>",
  "period_expression": "<the user's own words for any period/range/'lately'/'last N years', or null>",
  "explicit_frequency": "monthly" | "quarterly" | "yearly" | null,
  "growth_method_hint": "CAGR" | "simple" | null,
  "extremum": "max" | "min" | null,
  "language": "en" | "ar",
  "is_followup": true|false,
  "followup_reference": "<what the follow-up refers back to, e.g. 'same indicator, different period', or null>"
}}

Rules:
- Use "count_list" when the user asks HOW MANY indicators there are, or to LIST or
  NAME them, for a group rather than for one metric: "list the indicators in
  Sectors", "how many indicators in diversification target", "give me all the
  indicator names for the education sector", "what are the national indicators".
  Put the group the user named — the sector or indicator type — in
  indicator_phrase. These questions are about the CATALOGUE, not about any single
  indicator's value, so never route them to latest_value or definition.
- Use "analysis_lookup" when the user asks for SCAI's ANALYSIS or COMMENTARY on a
  specific INDICATOR — "what is the latest analysis of inflation", "what did the
  Council say about the trade balance this quarter", "ما هو أخر تحليل للتضخم",
  "summary analysis for GDP". Put the indicator in indicator_phrase.
  Contrast: analysis_lookup is analyst commentary attached to one indicator's data
  point; article_lookup is a published article about a theme; latest_value is the
  number itself.
- Use "article_lookup" when the user asks about SCAI's published WRITING, ANALYSIS,
  VIEWS or COMMENTARY on a topic, rather than for a number: "what has SCAI written
  about the trade war", "what is the Council's view on knowledge transfer", "tell me
  about the In-Country Value programme", "ماذا كتب المجلس عن الرسوم الجمركية".
  Signals: the question asks about a THEME, POLICY or PROGRAMME rather than a measurable
  indicator; or it asks for an opinion, argument, explanation or discussion; or it uses
  words like article, paper, publication, wrote, view, opinion, analysis, discuss.
  Put the topic the user is asking about in indicator_phrase.
  Contrast: "what is inflation" wants a definition, "what is the inflation rate" wants a
  number, and "what does SCAI say about inflation's causes" wants an article.
- Use "definition" when the user asks what an indicator MEANS rather than what it
  currently measures: "what is inflation?", "what does non-hydrocarbon GDP mean?",
  "define trade balance", "ما معنى التضخم". The distinguishing test is that no value,
  period, country or ranking is being requested.
  Do NOT use "definition" when a value is clearly wanted — "what is the latest value
  of Real GDP", "what is Inflation in 2025", "what is the GDP figure" are all
  latest_value. A bare "what is X?" naming an indicator, with no period and no word
  like value/rate/figure/level, is a definition question.
- If the user asks for per-period detail — "detail what happened in every year",
  "break it down by quarter", "show each year" — use "trend", NOT "growth_rate".
  growth_rate collapses the whole span into one start-to-end figure and cannot
  answer a request for what happened in between.
- "min_max" returns ONE value: the single highest or lowest, and when it occurred.
  "What was the highest quarterly GDP value recorded, and in which quarter?" is
  min_max with extremum="max" — it asks for one figure, not a table. Use
  "period_ranking" ONLY when the user explicitly wants several periods listed or
  ranked ("list them", "rank them", "top 5", "from highest to lowest").
- A "why" question — "why did inflation fall to 0.2%", "what caused the drop" — is
  NEVER a definition. Use "analysis_lookup": the user wants the explanation SCAI
  wrote, not the meaning of the word. Do not restate the premise as fact; the
  figure quoted in the question may be wrong.
- Use "period_ranking" when the user wants SEVERAL periods of ONE indicator listed in value
  order: "list the quarterly GDP values for 2024 and 2025 and rank them highest to lowest",
  "order the monthly inflation figures from lowest to highest". Set extremum="max" for
  highest-first (the default) or "min" for lowest-first.
  Contrast: "trend"/"show me how X changed" is chronological, "min_max" returns only the
  single highest or lowest value, and "country_ranking" ranks COUNTRIES, not periods.
- "compare" only means country_comparison or period_comparison if the user named specific
  countries or specific periods. If they say "compare" with no explicit countries/periods,
  and instead ask something like "which is lowest/highest", use country_ranking or min_max.
- extremum should be "max" for "highest"/"largest"/"most" style questions, "min" for
  "lowest"/"smallest"/"least", and null when computation_type isn't min_max/country_ranking.
- Do not invent countries, periods, or indicators the user did not mention.
- If the question is a follow-up ("what about last year", "and Q2?", "and for Saudi Arabia?",
  "what is the latest value?"), set is_followup=true and fill ONLY the fields the user
  actually restated. Leave every other field null — do not copy the indicator, country or
  period forward from the prior conversation yourself. Slots left null on a follow-up are
  filled deterministically from the previous turn's resolved values; a value you guess
  here overrides that and cannot be checked.
  Judge this from the "Prior conversation" block above: a question that only makes sense
  as a continuation of it ("and the year before that?") is a follow-up, while a question
  that names its own indicator is not, even when a conversation exists.
- indicator_phrase should be the user's own wording, not a guess at the catalog's exact name.
"""


def extract_intent(user_message: str, conversation_context: str = "") -> dict:
    user_prompt = f"Prior conversation (for follow-up detection only):\n{conversation_context}\n\nQuestion: {user_message}"
    raw = chat(
        client=router_client,
        model=settings.ROUTER_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=400,
    )
    try:
        parsed = json.loads(raw.strip().strip("```json").strip("```"))
        return parsed
    except json.JSONDecodeError:
        return {
            "computation_type": "out_of_scope",
            "indicator_phrase": None,
            "countries_mentioned": [],
            "country_group_mentioned": None,
            "period_expression": None,
            "explicit_frequency": None,
            "growth_method_hint": None,
            "extremum": None,
            "language": "en",
            "is_followup": False,
            "followup_reference": None,
        }
