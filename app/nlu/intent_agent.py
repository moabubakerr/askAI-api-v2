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
    "min_max",            # "what was the highest/lowest value of X"
    "difference",         # "what is the difference between the highest and lowest X"
    "growth_rate",        # "what is the growth rate of X between A and B" (CAGR or simple)
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
- "compare" only means country_comparison or period_comparison if the user named specific
  countries or specific periods. If they say "compare" with no explicit countries/periods,
  and instead ask something like "which is lowest/highest", use country_ranking or min_max.
- extremum should be "max" for "highest"/"largest"/"most" style questions, "min" for
  "lowest"/"smallest"/"least", and null when computation_type isn't min_max/country_ranking.
- Do not invent countries, periods, or indicators the user did not mention.
- If the question is a follow-up ("what about last year", "and Q2?"), set is_followup=true
  and describe only what's being changed relative to the prior turn — do not guess the
  actual prior indicator/country yourself, that's resolved separately with conversation state.
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
