"""
Chart-request detection — a plain keyword check, not an LLM judgment call.

Why not let the Intent Agent decide this: it's a binary signal with a small,
stable vocabulary ("chart", "graph", "plot", "visualize", "trend line") —
exactly the kind of thing that doesn't need a model call, and every model
call is one more place a wrong decision could slip in. Trend questions also
imply a chart even without the word "chart" being used, so that's included
too — deterministically, via computation_type, not by asking a model.
"""
import re

_CHART_KEYWORDS = re.compile(
    r"\b(chart|graph|plot|visuali[sz]e|diagram|trend\s*line)\b", re.IGNORECASE
)


def wants_chart(user_message: str, computation_type: str) -> bool:
    if _CHART_KEYWORDS.search(user_message):
        return True
    # A "trend" question is implicitly a chart request even if the word
    # "chart" never appears — "show me how GDP has been trending" (test sheet).
    return computation_type == "trend"
