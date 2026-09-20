"""
Numeric Verifier — a regex-based, non-LLM check that every number appearing
in the Composer's output text actually came from the facts payload. This is
the hard enforcement mechanism behind "the solution should not generate any
number" — the Composer prompt asks nicely, this makes sure.

If a stray number is found, the caller should NOT ship the LLM's text —
fall back to a template-rendered answer built directly from the facts
payload (see render_template_fallback), which has no LLM in the loop at all.
"""
import re
from decimal import Decimal

from app.compute.formatting import trim_zeros


# The lookbehind stops a hyphen being read as a minus sign when it follows a
# word character, which made period labels parse asymmetrically: the payload's
# "2025-Q4" yielded ["2025", "4"], but the same quarter written the way SCAI and
# the Composer write it, "Q4-2025", yielded ["4", "-2025"]. "-2025" was in
# neither the allowed set nor the 1900-2100 year exemption, so any answer naming
# a quarter that way was rejected as containing an invented number.
# Genuine negatives ("fell by -3.2%") still parse, since the hyphen there
# follows a space.
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def _extract_numbers_from_payload(payload) -> set[str]:
    """Walks the facts payload and collects every number as a normalized
    string (no commas, trimmed trailing zeros) so it can be compared against
    numbers found in the LLM's text."""
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return  # bool is a subclass of int; "True" is not a data point
        # Decimal matters here: `actual`/`target` are NUMERIC columns, so
        # psycopg2 hands back decimal.Decimal, not float. Decimal matches none
        # of the branches below, so every retrieved value used to be skipped —
        # which meant the verifier saw an EMPTY set of allowed numbers and
        # rejected the Composer's correct figure as invented. The visible
        # symptom was every numeric answer arriving as the raw template dump
        # ("actual: 185.17") with verified=false.
        if isinstance(value, (int, float, Decimal)):
            found.add(_normalize(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            for m in NUMBER_PATTERN.findall(value):
                found.add(_normalize(m))

    walk(payload)
    return found


def _normalize(value) -> str:
    try:
        f = float(str(value).replace(",", ""))
    except ValueError:
        return str(value)
    # Normalize both "1935" and "1935.0" to the same key; keep reasonable precision
    return f"{f:.4f}".rstrip("0").rstrip(".")


def verify_numbers(answer_text: str, facts_payload: dict, rounding_tolerance_decimals: int = 1) -> tuple[bool, list[str]]:
    """Returns (is_clean, offending_numbers). Allows for the headline
    rounding convention (payload may carry full precision, e.g. 177.429,
    while the answer rounds to 177.4) by checking the LLM's number against
    the payload's number rounded to `rounding_tolerance_decimals` as well."""
    payload_numbers = _extract_numbers_from_payload(facts_payload)
    payload_rounded = set()
    for n in payload_numbers:
        try:
            payload_rounded.add(_normalize(round(float(n), rounding_tolerance_decimals)))
        except ValueError:
            pass

    # A negative value is routinely stated as a magnitude plus a direction
    # word: the payload holds -23.4946 and the answer says "revenues fell
    # 23.5%". That is correct, and it was being rejected, which discarded a
    # good answer in favour of a template. The magnitude does trace to the
    # data, so it is allowed.
    #
    # The limit this accepts: the verifier cannot tell "fell 23.5%" from "rose
    # 23.5%". It never could — direction lives in the prose, not the number,
    # and no version of this check has ever verified a direction word. What it
    # still guarantees is that no figure appears that is not in the data.
    payload_absolute = set()
    for n in payload_numbers | payload_rounded:
        try:
            payload_absolute.add(_normalize(abs(float(n))))
        except ValueError:
            pass

    allowed = payload_numbers | payload_rounded | payload_absolute
    text_numbers = {_normalize(m) for m in NUMBER_PATTERN.findall(answer_text)}

    # ignore tiny integers that are almost certainly not data (years handled
    # separately since 2019-2030 range appears constantly and legitimately)
    offending = []
    for n in text_numbers:
        try:
            f = float(n)
        except ValueError:
            continue
        if 1900 <= f <= 2100 and f == int(f):
            continue  # looks like a year — years are allowed to appear as period labels
        if n not in allowed:
            offending.append(n)

    return (len(offending) == 0, offending)


# Every phrase the template fallback needs, in both languages.
#
# This existed only in English. render_template_fallback ACCEPTED a `language`
# argument and ignored it, so any Arabic question whose composed answer failed
# verification was answered in English — the product silently switching
# language at the moment it was least confident in itself. The Composer was
# never the problem here; the safety net was.
_T = {
    "no_data": {
        "en": "No approved data is available for this request.",
        "ar": "لا تتوفر بيانات معتمدة لهذا الطلب.",
    },
    "this_indicator": {"en": "This indicator", "ar": "هذا المؤشر"},
    "was_in": {"en": "{ind} was {val} in {per}.", "ar": "بلغ {ind} {val} في {per}."},
    "target_was": {"en": " The target for that period was {val}.",
                    "ar": " وكان المستهدف لتلك الفترة {val}."},
    "complement": {
        "en": ("{other} account for approximately {pct}% of the total. {ind} was "
                "{share}% in {per}, so the remainder is {derivation}."),
        "ar": ("يمثل {other} نحو {pct}% من الإجمالي. وقد بلغ {ind} {share}% في {per}، "
                "أي أن الباقي هو {derivation}."),
    },
    "remainder_unnamed": {
        "en": ("The remaining share is approximately {pct}% of the total. {ind} was "
                "{share}% in {per}, so the remainder is {derivation}. The data does not "
                "state what that remainder consists of."),
        "ar": ("النسبة المتبقية نحو {pct}% من الإجمالي. وقد بلغ {ind} {share}% في {per}، "
                "أي أن الباقي هو {derivation}. ولا تحدد البيانات مكوّنات هذه النسبة المتبقية."),
    },
    "assessment": {
        "en": "{ind} was {val} in {per}, {dir} {amount} {kind} year on year — {verdict}. {basis}",
        "ar": "بلغ {ind} {val} في {per}، {dir} {amount} {kind} على أساس سنوي — {verdict}. {basis}",
    },
    "improving": {"en": "an improvement", "ar": "تحسّن"},
    "deteriorating": {"en": "a deterioration", "ar": "تدهور"},
    "unchanged_word": {"en": "no change", "ar": "دون تغيير"},
    "up": {"en": "up", "ar": "بارتفاع"},
    "down": {"en": "down", "ar": "بانخفاض"},
    "flat": {"en": "unchanged", "ar": "دون تغيير"},
    "pp": {"en": "percentage points", "ar": "نقطة مئوية"},
    "pct_sign": {"en": "%", "ar": "%"},
    "growth": {
        "en": "{ind} moved from {v0} in {p0} to {v1} in {p1} — a {rate}% {method} growth rate.",
        "ar": "تحرك {ind} من {v0} في {p0} إلى {v1} في {p1} — بمعدل نمو {rate}% {method}.",
    },
    "difference": {
        "en": ("The highest {ind} was {hi} in {hip} and the lowest {lo} in {lop} — "
                "a difference of {diff}."),
        "ar": "أعلى قيمة لـ {ind} كانت {hi} في {hip} وأدناها {lo} في {lop} — بفارق {diff}.",
    },
    "change": {
        "en": "{ind} was {a} in {pa} and {b} in {pb} — a change of {abschg} ({pct}%).",
        "ar": "بلغ {ind} {a} في {pa} و{b} في {pb} — بتغير قدره {abschg} ({pct}%).",
    },
    "no_reading": {"en": "no reading", "ar": "لا توجد قراءة"},
    "from_in": {"en": ", from {val} in {per}", "ar": "، مقارنة بـ {val} في {per}"},
    "yoy": {"en": " ({pct}% YoY)", "ar": " ({pct}% على أساس سنوي)"},
    "largest_decline": {"en": "Largest decline: ", "ar": "أكبر تراجع: "},
    "then": {"en": ", then ", "ar": "، ثم "},
    "no_reading_in_period": {
        "en": "{ind} has no reading in the period asked about.",
        "ar": "لا توجد قراءة لـ {ind} في الفترة المطلوبة.",
    },
    "covers": {
        "en": " Its published series runs from {a} to {b}.",
        "ar": " وتمتد سلسلته المنشورة من {a} إلى {b}.",
    },
    "not_found": {
        "en": "Not found in the approved data: {names}.",
        "ar": "لم يتم العثور عليها في البيانات المعتمدة: {names}.",
    },
    "these_indicators": {"en": "These indicators", "ar": "هذه المؤشرات"},
    "vs_year_earlier": {
        "en": "{scope}, compared with a year earlier:",
        "ar": "{scope}، مقارنة بالعام السابق:",
    },
    "increasing": {"en": "Increasing", "ar": "مرتفعة"},
    "declining": {"en": "Declining", "ar": "منخفضة"},
    "unchanged_group": {"en": "Unchanged", "ar": "دون تغيير"},
    "no_yoy_for": {
        "en": "No year-on-year comparison available for {n} of {total}: ",
        "ar": "لا تتوفر مقارنة سنوية لـ {n} من {total}: ",
    },
    "this_group": {"en": "this group", "ar": "هذه المجموعة"},
    "ranked_by_target": {
        "en": "{scope}, ranked by progress against each indicator's own target ({order}):",
        "ar": "{scope}، مرتبة حسب التقدم نحو مستهدف كل مؤشر ({order}):",
    },
    "best_first": {"en": "closest to target first", "ar": "الأقرب إلى المستهدف أولاً"},
    "worst_first": {"en": "furthest from target first", "ar": "الأبعد عن المستهدف أولاً"},
    "of_target": {
        "en": "{i}. {ind} — {pct}% of target ({val} in {per}, target {tgt})",
        "ar": "{i}. {ind} — {pct}% من المستهدف ({val} في {per}، المستهدف {tgt})",
    },
    "indicators_word": {"en": "indicators", "ar": "مؤشراً"},
    "in_the_sector": {"en": "published indicators in the {scope}",
                       "ar": "مؤشراً منشوراً في {scope}"},
    "published_type": {"en": "published {scope}s", "ar": "من {scope} المنشورة"},
    "there_are": {"en": "There are {n} {what} in the approved data.",
                   "ar": "يوجد {n} {what} في البيانات المعتمدة."},
    "for_example": {"en": " For example: {names}.", "ar": " على سبيل المثال: {names}."},
    "all_listed_below": {"en": " All {n} are listed below.",
                          "ar": " جميعها البالغ عددها {n} مدرجة أدناه."},
    "n_readings": {"en": "{ind}: {n} readings", "ar": "{ind}: {n} قراءة"},
    "series_span": {"en": ", from {v0} in {p0} to {v1} in {p1}",
                     "ar": "، من {v0} في {p0} إلى {v1} في {p1}"},
    "series_change": {"en": " That is a change of {pct}% over the period.",
                       "ar": " أي بتغير قدره {pct}% خلال الفترة."},
    "series_extremes": {
        "en": " The highest reading was {hi} in {hip}, the lowest {lo} in {lop}.",
        "ar": " وكانت أعلى قراءة {hi} في {hip}، وأدناها {lo} في {lop}.",
    },
    "reason_no_reading": {"en": "no reading yet", "ar": "لا توجد قراءة بعد"},
    "reason_no_target": {"en": "no target set", "ar": "لم يُحدَّد مستهدف"},
    "reason_no_yoy_published": {"en": "no year-on-year figure published",
                                 "ar": "لم تُنشر مقارنة سنوية"},
    "desired_decrease": {
        "en": "SCAI records the desired direction for this indicator as a decrease.",
        "ar": "يسجّل المجلس الاتجاه المرغوب لهذا المؤشر على أنه انخفاض.",
    },
    "desired_increase": {
        "en": "SCAI records the desired direction for this indicator as an increase.",
        "ar": "يسجّل المجلس الاتجاه المرغوب لهذا المؤشر على أنه ارتفاع.",
    },
    "places": {"en": "places", "ar": "مراكز"},
    "rank_assessment": {
        "en": ("{ind} was {val} in {per}, {dir} from {prev} in {prevper} — a move of "
                "{amount} {kind}, {verdict}. {basis}"),
        "ar": ("بلغ {ind} {val} في {per}، {dir} من {prev} في {prevper} — بتغير قدره "
                "{amount} {kind}، {verdict}. {basis}"),
    },
    "rank_leader": {
        "en": "{country} had the {extremum} {ind} in {per}, at {val}.",
        "ar": "سجّلت {country} {extremum} {ind} في {per}، بقيمة {val}.",
    },
    "lowest": {"en": "lowest", "ar": "أدنى"},
    "highest": {"en": "highest", "ar": "أعلى"},
    "no_data_for": {"en": "No approved data for: {names}.",
                     "ar": "لا تتوفر بيانات معتمدة لـ: {names}."},
    "level_lead": {
        "en": "{top} is the higher at {topval} ({topper}); {bottom} is {botval} ({botper}).",
        "ar": "{top} هو الأعلى بـ {topval} ({topper})؛ و{bottom} بـ {botval} ({botper}).",
    },
    "level_gap": {"en": " The gap is {gap} {kind}.", "ar": " والفارق {gap} {kind}."},
    "periods_differ": {
        "en": (" The readings are from different periods ({periods}), so treat this "
                "as a directional comparison rather than a matched-period one."),
        "ar": (" القراءتان من فترتين مختلفتين ({periods})، لذا تُعامل هذه المقارنة "
                "كمؤشر اتجاه لا كمقارنة متطابقة الفترة."),
    },
    "component_note": {
        "en": " — this is the \"{part}\" component, one of {n} published under this indicator; no combined total is published.",
        "ar": " — هذا هو مكوّن \"{part}\"، أحد {n} مكوّنات منشورة تحت هذا المؤشر؛ ولا يُنشر إجمالي مجمّع.",
    },
    "not_ranked": {
        "en": "{n} of {total} could not be ranked: ",
        "ar": "تعذّر ترتيب {n} من {total}: ",
    },
}


def _t(key: str, language: str = "en", **kwargs) -> str:
    entry = _T.get(key, {})
    text = entry.get(language) or entry.get("en") or ""
    return text.format(**kwargs) if kwargs else text


def _reason(entry: dict, language: str) -> str:
    """The reason an indicator could not be ranked or compared, in words.

    The engine stores a code because it has no language. Falls back to any
    legacy prose so a payload built before this change still renders.
    """
    code = entry.get("reason_code")
    if code:
        return _t(f"reason_{code}", language)
    return entry.get("reason") or ""


def render_template_fallback(facts_payload: dict, language: str = "en") -> str:
    """A zero-LLM, template-only rendering used when the Composer's text
    fails verification. Deliberately plain — correctness over eloquence.

    Bilingual, because this is what a reader actually sees whenever the
    generated wording is rejected. An English-only safety net means the product
    changes language precisely when it is least sure of itself, which is how an
    Arabic question about the economy came back as an English list.
    """
    language = "ar" if str(language).lower().startswith("ar") else "en"
    if facts_payload.get("ok") is False:
        return facts_payload.get("message") or _t("no_data", language)

    facts = facts_payload.get("facts", {})
    unit = facts.get("unit") or ""
    indicator = facts.get("indicator") or _t("this_indicator", language)

    def fmt(value):
        # trim_zeros, not rounding: Decimal('3.680000') printed "a change of
        # 3.680000 QAR" — six decimals of apparent precision on a figure that
        # has two. Padding only; the digits themselves are untouched.
        return f"{trim_zeros(value)} {unit}".strip() if value is not None else "—"

    if "complement_share" in facts:
        key = "complement" if facts.get("complement_of") else "remainder_unnamed"
        return _t(key, language, other=facts.get("complement_of") or "",
                   pct=facts["complement_share"], ind=facts["reported_share_of"],
                   share=facts["reported_share"], per=facts.get("period_label"),
                   derivation=facts["derivation"])
    if "assessment" in facts:
        verdict = _t({"improving": "improving", "deteriorating": "deteriorating",
                       "unchanged": "unchanged_word"}[facts["assessment"]], language)
        kind = _t({"percentage_points": "pp", "places": "places"}.get(
            facts.get("change_kind"), "pct_sign"), language)
        direction = _t({"up": "up", "down": "down"}.get(facts.get("direction"), "flat"), language)
        # A rank move is stated as a move BETWEEN POSITIONS, never as a
        # percentage. "9th to 11th" is two places; "22.2% worse" is a
        # percentage of an ordinal, which means nothing.
        if facts.get("change_kind") == "places" and facts.get("previous_value") is not None:
            return _t("rank_assessment", language, ind=indicator,
                       val=fmt(facts.get("actual")), per=facts.get("period_label"),
                       dir=direction, prev=fmt(facts.get("previous_value")),
                       prevper=facts.get("previous_period"),
                       amount=abs(facts.get("change", 0)), kind=_t("places", language),
                       verdict=verdict,
                       basis=_t(f"desired_{facts.get('desired_direction', 'increase')}",
                                 language) if facts.get("desired_direction")
                       else facts.get("assessment_basis", "")).strip()
        return _t("assessment", language, ind=indicator, val=fmt(facts.get("actual")),
                   per=facts.get("period_label"), dir=direction,
                   amount=abs(facts.get("change", 0)), kind=kind, verdict=verdict,
                   basis=_t(f"desired_{facts.get('desired_direction', 'increase')}",
                             language) if facts.get("desired_direction")
                   else facts.get("assessment_basis", "")).strip()
    if "actual" in facts and "period_label" in facts:
        line = _t("was_in", language, ind=indicator, val=fmt(facts["actual"]),
                   per=facts["period_label"])
        if facts.get("target") is not None:
            line += _t("target_was", language, val=fmt(facts["target"]))
        return line
    if "growth_rate_percent" in facts:
        return _t("growth", language, ind=indicator, v0=fmt(facts.get("value_start")),
                   p0=facts.get("period_start"), v1=fmt(facts.get("value_end")),
                   p1=facts.get("period_end"), rate=facts["growth_rate_percent"],
                   method=facts.get("method", "")).replace("  ", " ")
    if "absolute_difference" in facts:
        return _t("difference", language, ind=indicator, hi=fmt(facts.get("high_value")),
                   hip=facts.get("high_period"), lo=fmt(facts.get("low_value")),
                   lop=facts.get("low_period"), diff=fmt(facts["absolute_difference"]))
    if "percent_change" in facts:
        return _t("change", language, ind=indicator, a=fmt(facts.get("value_a")),
                   pa=facts.get("period_a"), b=fmt(facts.get("value_b")),
                   pb=facts.get("period_b"), abschg=fmt(facts.get("absolute_change")),
                   pct=facts["percent_change"])
    if "definition" in facts:
        return facts["definition"]
    if "overview" in facts:
        lines = []
        for e in facts["overview"]:
            value = e.get("actual")
            row_unit = e.get("unit") or ""
            if e.get("report_as_growth") and e.get("change_yoy_percent") is not None:
                lines.append(f"{e.get('indicator')}:"
                             + _t("yoy", language, pct=e["change_yoy_percent"])
                             + f" ({e.get('period_label')})")
                continue
            shown = (f"{trim_zeros(value)} {row_unit}".strip() if value is not None
                     else _t("no_reading", language))
            line = f"{e.get('indicator')}: {shown} ({e.get('period_label')})"
            # The movement, not just the level. "How is the economy doing"
            # answered with four current values is a list of readings, not an
            # answer — the direction is the question.
            if e.get("previous_value") is not None:
                line += _t("from_in", language,
                            val=f"{trim_zeros(e['previous_value'])} {row_unit}".strip(),
                            per=e.get("previous_period"))
            if e.get("change_yoy_percent") is not None:
                line += _t("yoy", language, pct=e["change_yoy_percent"])
            if e.get("component_name"):
                line += _t("component_note", language, part=e["component_name"],
                            n=e.get("component_of_total"))
            lines.append(line)
        if facts.get("ranked_by_level"):
            ordered = facts["ranked_by_level"]
            top, bottom = ordered[0], ordered[-1]
            kind = _t("pp", language) if facts.get("difference_kind") == "percentage_points"                 else (facts.get("unit") or "")
            line = _t("level_lead", language, top=top["indicator"],
                       topval=f"{trim_zeros(top['actual'])} {top.get('unit') or ''}".strip(),
                       topper=top.get("period_label"), bottom=bottom["indicator"],
                       botval=f"{trim_zeros(bottom['actual'])} {bottom.get('unit') or ''}".strip(),
                       botper=bottom.get("period_label"))
            if facts.get("difference") is not None:
                line += _t("level_gap", language, gap=facts["difference"], kind=kind)
            if facts.get("periods_differ"):
                line += _t("periods_differ", language,
                            periods=", ".join(str(p) for p in facts.get("periods_compared") or []))
            lines.append(line)
        if facts.get("change_ranking"):
            lines.append(_t("largest_decline", language) + _t("then", language).join(
                f"{e['indicator']} ({e['change_yoy_percent']}%)"
                for e in facts["change_ranking"]) + ".")
        for e in facts.get("no_data_in_period") or []:
            covers = (_t("covers", language, a=e["covers_from"], b=e["covers_to"])
                      if e.get("covers_from") else "")
            lines.append(_t("no_reading_in_period", language, ind=e["indicator"]) + covers)
        if facts.get("not_found"):
            lines.append(_t("not_found", language, names=", ".join(facts["not_found"])))
        return "\n".join(lines)
    if "count" in facts and "names" in facts:
        scope = facts.get("scope")
        # A sector is a container, a type is a label: "101 published Sector
        # Indicators" is right, "13 published Education Sectors" is not.
        if not scope:
            what = _t("indicators_word", language)
        elif facts.get("scope_kind") == "sector":
            what = _t("in_the_sector", language, scope=scope)
        else:
            what = _t("published_type", language, scope=scope)
        line = _t("there_are", language, n=facts["count"], what=what)
        # The names are rendered in full beside this text, so enumerating them
        # here prints them twice. But answering "give me all the names" with
        # "for example: ..." reads as a refusal of the question that was asked,
        # so say plainly that all of them are there. Examples only earn their
        # space on a list too long to take in at a glance.
        if facts["count"] > 15:
            sample = facts.get("names_sample") or facts["names"][:8]
            if sample:
                line += _t("for_example", language,
                            names="; ".join(str(n) for n in sample))
        return line + _t("all_listed_below", language, n=facts["count"])
    if "series" in facts:
        # Summarise; do NOT enumerate. The series is already rendered as a chart
        # and a table beside this text, so listing all 28 points printed the
        # same data three times on one screen.
        line = _t("n_readings", language, ind=indicator,
                   n=facts.get("n_points", len(facts["series"])))
        if facts.get("first_period"):
            line += _t("series_span", language, v0=fmt(facts.get("first_value")),
                        p0=facts["first_period"], v1=fmt(facts.get("last_value")),
                        p1=facts.get("last_period"))
        line += "."
        if facts.get("change_percent") is not None:
            line += _t("series_change", language, pct=facts["change_percent"])
        if facts.get("highest_period"):
            line += _t("series_extremes", language, hi=fmt(facts.get("highest_value")),
                        hip=facts["highest_period"], lo=fmt(facts.get("lowest_value")),
                        lop=facts.get("lowest_period"))
        return line
    if "ranked" in facts and facts.get("leader"):
        leader = facts["leader"]
        lines = [_t("rank_leader", language, country=leader.get("country"),
                     extremum=_t(facts.get("extremum", "lowest"), language),
                     ind=indicator, per=leader.get("period_label"),
                     val=f"{trim_zeros(leader.get('actual'))} {unit}".strip())]
        for i, row in enumerate(facts["ranked"], 1):
            lines.append(f"{i}. {row.get('country')}: "
                         f"{trim_zeros(row.get('actual'))} {unit}".strip()
                         + f" ({row.get('period_label')})")
        if facts.get("countries_with_no_data"):
            lines.append(_t("no_data_for", language,
                             names=", ".join(facts["countries_with_no_data"])))
        return "\n".join(lines)
    if "increasing" in facts and "declining" in facts:
        scope = facts.get("scope") or _t("these_indicators", language)
        lines = [_t("vs_year_earlier", language, scope=scope)]
        for key in ("increasing", "declining", "unchanged"):
            group = facts.get(key) or []
            if not group:
                continue
            label = _t({"increasing": "increasing", "declining": "declining",
                         "unchanged": "unchanged_group"}[key], language)
            lines.append(f"\n{label} ({len(group)}):")
            for e in group:
                value = f"{trim_zeros(e.get('actual'))} {e.get('unit') or ''}".strip()
                # A percentage-point move is not a percentage change. Printing
                # "+0.1%" for a ratio that moved 40.5 -> 40.6 misstates it.
                moved = (f"{e['change_yoy_percent']}%" if e.get("change_yoy_percent") is not None
                         else f"{e.get('change_yoy_pp')} {_t('pp', language)}")
                lines.append(f"- {e.get('indicator')}: {moved} "
                             f"({value} — {e.get('period_label')})")
        skipped = facts.get("no_comparison") or []
        if skipped:
            lines.append("\n" + _t("no_yoy_for", language, n=len(skipped),
                                    total=facts.get("n_total"))
                         + "; ".join(f"{e.get('indicator')} ({_reason(e, language)})"
                                      for e in skipped) + ".")
        return "\n".join(lines)
    if "ranked_indicators" in facts:
        scope = facts.get("scope") or _t("this_group", language)
        order = _t("best_first" if facts.get("order") == "best_first" else "worst_first",
                    language)
        lines = [_t("ranked_by_target", language, scope=scope, order=order)]
        for i, e in enumerate(facts["ranked_indicators"], 1):
            value = f"{trim_zeros(e.get('actual'))} {e.get('unit') or ''}".strip()
            lines.append(_t("of_target", language, i=i, ind=e.get("indicator"),
                             pct=e.get("attainment_percent"), val=value,
                             per=e.get("period_label"), tgt=e.get("target")))
        skipped = facts.get("not_assessable") or []
        if skipped:
            lines.append(_t("not_ranked", language, n=len(skipped), total=facts.get("n_total"))
                         + "; ".join(f"{e.get('indicator')} ({_reason(e, language)})"
                                      for e in skipped) + ".")
        return "\n".join(lines)

    lines = []
    for key, value in facts.items():
        if key in ("unit", "indicator", "n_points"):
            continue
        if key == "series":
            for point in value:
                lines.append(f"{point.get('period_label')}: {point.get('actual')}")
        elif key == "rows":
            for row in value:
                lines.append(f"{row.get('country')} ({row.get('period_label')}): {row.get('actual')}")
        elif key == "ranked":
            for i, row in enumerate(value, 1):
                lines.append(f"{i}. {row.get('country')}: {row.get('actual')} ({row.get('period_label')})")
        elif key == "countries_with_no_data" and value:
            lines.append("No approved data for: " + ", ".join(value))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "No approved data is available for this request."
