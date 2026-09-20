"""
"Read this for me" — the same retrieved readings as /chat, returned in the
shape a reader needs rather than as one block of prose.

The organising principle is that the response keeps three kinds of text
strictly apart, because they carry different authority:

  council_analysis   written by SCAI analysts. Returned VERBATIM from
                     indicator_analysis. Never paraphrased, never merged into
                     the narration, always labelled with the period it was
                     written about.
  evidence           the readings themselves — value, period, unit, and the
                     record id each came from. Machine-checkable.
  narration          generated prose. Numeric-verified against the same facts
                     payload the readings came from, and carries a disclaimer
                     saying it is generated, not Council analysis.

Merging the first and the third would be the most damaging thing this endpoint
could do: it would let generated sentences inherit the Council's authority.
That is why they are separate fields and not a single rendered string — the
frontend cannot accidentally blur them, and a reader can always see which is
which.
"""
import re
from typing import Optional

from app.core.graph_v2 import handle_message
from app.agents.reader_agent import read_plainly
from app.compute.verifier import verify_numbers
from app.db import retriever


_MONTHS = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}


def humanize_period(period_label: Optional[str]) -> Optional[str]:
    """'2025-12' -> 'December 2025', '2025-Q4' -> 'Q4 2025', '2025' -> '2025'."""
    if not period_label:
        return None
    label = str(period_label).strip()
    if "-Q" in label:
        year, q = label.split("-Q", 1)
        return f"Q{q} {year}"
    if "-" in label:
        year, month = label.split("-", 1)
        if month in _MONTHS:
            return f"{_MONTHS[month]} {year}"
    return label


def _format_value(value, unit: Optional[str]) -> str:
    if value is None:
        return "—"
    text = str(value)
    return f"{text} {unit}".strip() if unit else text


def _headline_from_facts(facts: dict) -> Optional[dict]:
    """Only single-reading answers get a headline card. A trend or a ranking has
    no single number to put in large type, and inventing one (the latest? the
    highest?) would be the system choosing an emphasis the user didn't ask for."""
    if "actual" not in facts:
        return None
    return {
        "value": facts.get("actual"),
        "unit": facts.get("unit"),
        "indicator": facts.get("indicator"),
        "period_label": facts.get("period_label"),
        "period_human": humanize_period(facts.get("period_label")),
    }


def _evidence_rows(facts: dict, citations: list[dict]) -> list[dict]:
    """The readings behind the answer, paired with the record each came from."""
    by_period = {c.get("period_label"): c for c in citations if c.get("period_label")}

    def row(period_label, value, target=None):
        cite = by_period.get(period_label, {})
        return {
            "period_label": period_label,
            "period_human": humanize_period(period_label),
            "value": value,
            "target": target,
            "record_id": cite.get("record_id"),
            "table": cite.get("table"),
            "country": cite.get("country"),
        }

    if "series" in facts:
        return [row(p.get("period_label"), p.get("actual")) for p in facts["series"]]
    if "ranked_periods" in facts:
        return [row(p.get("period_label"), p.get("actual")) for p in facts["ranked_periods"]]
    for key in ("ranked", "rows"):
        if key in facts:
            return [{**row(e.get("period_label"), e.get("actual")), "country": e.get("country")}
                    for e in facts[key]]
    if "actual" in facts:
        return [row(facts.get("period_label"), facts.get("actual"), facts.get("target"))]

    # Two-reading answers — growth rate, period comparison, min/max, difference.
    # These were missing entirely, so /read showed an empty "Data evidence"
    # section for exactly the answers where seeing both endpoints matters most:
    # a CAGR is only checkable if you can see the two values it was computed
    # from.
    pairs = [("period_start", "value_start"), ("period_end", "value_end"),
             ("period_a", "value_a"), ("period_b", "value_b"),
             ("high_period", "high_value"), ("low_period", "low_value")]
    endpoints = [row(facts[p], facts[v]) for p, v in pairs if p in facts and v in facts]
    return endpoints


def _period_mismatch(entry: dict) -> bool:
    """True when the commentary text talks about a different year than the data
    point it is filed against.

    This is a source-data problem, not ours: the analysis row attached to Real
    GDP's 2019-Q1 point reads "Real GDP grew by 6.1% YoY in Q4 2024". Labelling
    it "According to SCAI, Q1 2019" is accurate about WHERE it is filed and
    misleading about WHAT it says, so the disagreement is surfaced rather than
    smoothed over. The frontend can then attribute it to the data point instead
    of to the period, and SCAI can fix the filing.
    """
    label = str(entry.get("period_label") or "")
    year = re.match(r"(\d{4})", label)
    if not year:
        return False
    text = " ".join(str(entry.get(k) or "") for k in
                    ("summary", "detailed", "npc_analysis", "benchmark"))
    years_in_text = set(re.findall(r"\b(20\d{2})\b", text))
    # No year mentioned at all means nothing to disagree with.
    return bool(years_in_text) and year.group(1) not in years_in_text


def _council_analysis(citations: list[dict]) -> list[dict]:
    """SCAI's written commentary for the exact readings used, verbatim."""
    record_ids = [c["record_id"] for c in citations if c.get("record_id")]
    found = retriever.get_analysis_for_data_points(record_ids)
    if not found:
        return []

    by_record = {c.get("record_id"): c for c in citations}
    out = []
    for record_id, analysis in found.items():
        cite = by_record.get(record_id, {})
        entry = {
            "period_label": cite.get("period_label"),
            "period_human": humanize_period(cite.get("period_label")),
            "indicator": cite.get("indicator"),
            "record_id": record_id,
            "is_published_analysis": analysis.get("source") == "published",
        }
        for field, key in (("summary_en", "summary"),
                            ("detailed_analysis_en", "detailed"),
                            ("npc_analysis_en", "npc_analysis"),
                            ("benchmark_en", "benchmark")):
            value = (analysis.get(field) or "").strip()
            if value and value != "-":
                entry[key] = value
        # Nothing but metadata means the row exists but is empty — drop it
        # rather than render an "analysis" heading over blank space.
        if any(k in entry for k in ("summary", "detailed", "npc_analysis", "benchmark")):
            entry["period_mismatch"] = _period_mismatch(entry)
            out.append(entry)

    # Most recent first, and at most two. A growth-rate answer cites both
    # endpoints, so without this a CAGR from 2019 to 2025 rendered the 2019
    # commentary alongside the 2025 one as though both described the result.
    out.sort(key=lambda e: e.get("period_label") or "", reverse=True)
    return out[:2]


def _looks_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text or ""))


def read_message(user_message: str, conversation_context: str = "",
                  session_state: Optional[dict] = None) -> dict:
    """Runs the normal pipeline, then re-presents it as a reading.

    Deliberately built on handle_message rather than beside it: the readings,
    the citations and the numeric verification must be identical to what /chat
    returns for the same question. A second retrieval path would be a second
    chance to disagree with itself.
    """
    result = handle_message(user_message, conversation_context, session_state)
    payload = result["facts_payload"]
    citations = payload.get("citations", [])
    facts = payload.get("facts", {}) or {}

    if not payload.get("ok"):
        return {
            "ok": False,
            "readable": False,
            "message": payload.get("message", "No approved data is available for this request."),
            "headline": None,
            "one_liner": None,
            "council_analysis": [],
            "evidence": [],
            "narration": None,
            "disclaimer": None,
            "facts_payload": payload,
            "session_state": result["session_state"],
        }

    language = "ar" if _looks_arabic(user_message) else "en"
    headline = _headline_from_facts(facts)
    indicator = facts.get("indicator")
    one_liner = None
    if headline and indicator:
        one_liner = (f"{indicator} — {_format_value(headline['value'], headline['unit'])}"
                     f"{', ' + headline['period_label'] if headline['period_label'] else ''}.")

    analysis = _council_analysis(citations)

    # A plain-language retelling of the SAME facts, rather than the /chat answer
    # repeated. Reusing that answer made "Read this for me" change nothing: the
    # Composer writes for a policymaker — full precision, "Q4 2025", "YoY" —
    # which is exactly the register a reader who pressed that button is asking
    # for help with.
    #
    # Verified like any other generated text. The reader agent may round in its
    # prose, so the check allows the rounding the numeric verifier already
    # tolerates; if it slips a figure that is not in the facts, the /chat
    # answer stands in rather than shipping an unverified retelling.
    # Strip the Sources footer from the fallback: /chat appends it because that
    # response has nowhere else to put provenance, while /read returns
    # citations as their own field and the frontend renders them separately.
    narration = re.split(r"\n\s*Sources:\s*\n", result["answer"])[0].strip()
    try:
        plain = read_plainly(payload, language, question=user_message)
        clean, offending = verify_numbers(plain, payload)
        if clean and plain.strip():
            narration = plain.strip()
        else:
            payload["_plain_reading_rejected"] = offending
    except Exception:
        # The reading view must still render if this second model call fails.
        payload["_plain_reading_failed"] = True

    return {
        "ok": True,
        "readable": result.get("readable", False),
        "headline": headline,
        "one_liner": one_liner,
        "council_analysis": analysis,
        "evidence": _evidence_rows(facts, citations),
        "narration": narration,
        "disclaimer": ("Generated from the readings above — not Council analysis."
                        if analysis else
                        "Generated from the readings above."),
        "verified": "_verifier_rejected_numbers" not in payload,
        "facts_payload": payload,
        "chart": payload.get("chart"),
        "session_state": result["session_state"],
    }
