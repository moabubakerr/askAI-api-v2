"""
User-facing strings that are NOT produced by the Composer, in both languages.

Most replies are phrased by the Composer, which is told the language and so
answers in Arabic when asked to. Two categories never reach it:

  - canned replies that deliberately skip the LLM (greetings), and
  - refusals, where sending text to a model to be "rephrased" risks it
    answering around the refusal instead of stating it.

Those were English-only, so an Arabic "اهلا" was answered in English. Asking in
Arabic and being answered in English is not a cosmetic defect for a Qatari
government assistant — it is the product not working in one of its two
languages.

Language detection here is deliberately NOT left to the model. The intent agent
returns a language field, but it is one more thing an LLM can get wrong, and
Arabic script is unambiguous: if the question contains Arabic characters, the
question is in Arabic. The model's answer is used only when the script is
inconclusive.
"""
import re
from typing import Optional

# Arabic block plus the Arabic Supplement/Extended-A ranges.
_ARABIC_CHARS = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")


def detect_language(user_message: str, model_language: Optional[str] = None) -> str:
    """Arabic script wins over the model's guess; otherwise defer to the model."""
    if user_message and _ARABIC_CHARS.search(user_message):
        return "ar"
    if model_language in ("ar", "en"):
        return model_language
    return "en"


def is_arabic(language: Optional[str]) -> bool:
    return (language or "en").lower().startswith("ar")


_MESSAGES = {
    # Greetings answer in kind and say what the assistant is actually for —
    # a bare "Hello!" leaves the user guessing what they may ask.
    "greeting": {
        "en": ("Hello. I answer questions about Qatar's published SCAI economic indicators.\n\n"
                "You can ask me for:\n"
                "• a latest value — \"what is the latest Real GDP?\"\n"
                "• a trend or chart — \"show the monthly trend of inflation\"\n"
                "• a comparison — \"compare inflation in Qatar and Singapore\"\n"
                "• a definition — \"what does non-hydrocarbon GDP mean?\"\n"
                "• what I cover — \"what can you do?\""),
        "ar": ("أهلاً بك. أجيب عن الأسئلة المتعلقة بالمؤشرات الاقتصادية المنشورة "
                "للمجلس الأعلى للشؤون الاقتصادية والاستثمار.\n\n"
                "يمكنك أن تسألني عن:\n"
                "• أحدث قيمة — \"ما هو أحدث ناتج محلي إجمالي حقيقي؟\"\n"
                "• اتجاه أو رسم بياني — \"اعرض الاتجاه الشهري للتضخم\"\n"
                "• مقارنة — \"قارن التضخم بين قطر وسنغافورة\"\n"
                "• تعريف — \"ما معنى الناتج المحلي غير الهيدروكربوني؟\"\n"
                "• ما الذي أغطيه — \"ماذا يمكنك أن تفعل؟\""),
    },
    "out_of_scope": {
        "en": ("That's outside what I can help with — I only cover Qatar's published SCAI "
                "economic indicators, and I report figures from that data rather than "
                "forecasting or commenting beyond it. Ask me for an indicator's value, "
                "trend, comparison or definition."),
        "ar": ("هذا خارج نطاق ما يمكنني المساعدة فيه — أغطي فقط المؤشرات الاقتصادية المنشورة "
                "للمجلس الأعلى للشؤون الاقتصادية والاستثمار، وأعرض الأرقام الواردة في تلك "
                "البيانات دون تنبؤ أو تحليل يتجاوزها. يمكنك أن تسألني عن قيمة مؤشر أو اتجاهه "
                "أو مقارنته أو تعريفه."),
    },
    # Refusals name the gap AND the next move. "No data available" alone leaves
    # the user unable to tell whether they misnamed the indicator, asked for a
    # period that isn't covered, or hit a genuine gap.
    "indicator_not_found": {
        "en": ("I couldn't find an indicator in the approved SCAI data matching \"{phrase}\". "
                "I won't guess at a similar-sounding one, because answering with the wrong "
                "indicator is worse than not answering. Try the metric's name as SCAI "
                "publishes it, or ask \"what can you do?\" to see what's covered."),
        "ar": ("لم أجد مؤشراً في بيانات المجلس المعتمدة يطابق \"{phrase}\". "
                "لن أخمّن مؤشراً مشابهاً في الاسم، لأن الإجابة بمؤشر خاطئ أسوأ من عدم الإجابة. "
                "جرّب اسم المؤشر كما ينشره المجلس، أو اسأل \"ماذا يمكنك أن تفعل؟\" "
                "لمعرفة ما هو مشمول."),
    },
    "indicator_ambiguous": {
        "en": ("\"{phrase}\" could match more than one indicator: {names}. "
                "Which one did you mean?"),
        "ar": ("\"{phrase}\" قد يطابق أكثر من مؤشر: {names}. أيّها تقصد؟"),
    },
    "indicator_contradiction": {
        "en": ("You asked about \"{phrase}\", but the approved data only has \"{matched}\" — "
                "that's a different measure, so I won't substitute it. There's no matching "
                "indicator for what you asked."),
        "ar": ("سألت عن \"{phrase}\"، لكن البيانات المعتمدة تحتوي فقط على \"{matched}\" — "
                "وهو مقياس مختلف، لذا لن أستبدله. لا يوجد مؤشر مطابق لما سألت عنه."),
    },
    "indicator_inactive": {
        "en": ("\"{matched}\" is marked inactive in the approved data. "
                "I can still report what was recorded, but it is not being maintained."),
        "ar": ("\"{matched}\" مُعلَّم كغير نشط في البيانات المعتمدة. "
                "يمكنني عرض ما تم تسجيله، لكنه لم يعد يُحدَّث."),
    },
    "no_data_for_period": {
        "en": ("\"{indicator}\" has no {granularity} data for {period}. "
                "Approved data runs from {first} to {last}."),
        "ar": ("\"{indicator}\" لا يحتوي على بيانات {granularity_ar} للفترة {period}. "
                "البيانات المعتمدة تمتد من {first} إلى {last}."),
    },
    "no_data_at_all": {
        "en": "\"{indicator}\" has no data points in the approved dataset.",
        "ar": "\"{indicator}\" لا يحتوي على أي نقاط بيانات في مجموعة البيانات المعتمدة.",
    },
    "no_data_at_frequency": {
        "en": ("\"{indicator}\" has no {wanted} data in the approved dataset "
                "(available: {available})."),
        "ar": ("\"{indicator}\" لا يحتوي على بيانات {wanted_ar} في البيانات المعتمدة "
                "(المتاح: {available})."),
    },
    "no_definition": {
        "en": ("\"{indicator}\" is in the approved data, but SCAI's catalog carries no "
                "definition for it. I won't supply one of my own. I can still report its "
                "values if that helps."),
        "ar": ("\"{indicator}\" موجود في البيانات المعتمدة، لكن كتالوج المجلس لا يتضمن "
                "تعريفاً له. لن أضع تعريفاً من عندي. يمكنني عرض قيمه إذا كان ذلك مفيداً."),
    },
    "no_benchmarks": {
        "en": ("\"{indicator}\" has no benchmark countries designated in the approved data, "
                "so there is nothing to rank it against. Name the countries you want "
                "compared and I'll use those."),
        "ar": ("\"{indicator}\" ليس له دول مرجعية محددة في البيانات المعتمدة، "
                "لذا لا يوجد ما يُقارن به. حدّد الدول التي تريد مقارنتها وسأستخدمها."),
    },
    "no_indicators_matching": {
        "en": ("No indicators found matching \"{query}\". Try a sector name such as Tourism, "
                "Manufacturing or Education, or ask \"what can you do?\" for the full list."),
        "ar": ("لم أجد مؤشرات تطابق \"{query}\". جرّب اسم قطاع مثل السياحة أو الصناعة "
                "التحويلية أو التعليم، أو اسأل \"ماذا يمكنك أن تفعل؟\" للقائمة الكاملة."),
    },
    "no_overview": {
        "en": "No headline indicators were available to build an overview.",
        "ar": "لا توجد مؤشرات رئيسية متاحة لإعداد نظرة عامة.",
    },
    "low_confidence_match": {
        "en": ("I matched your question to \"{indicator}\" (approximate match). "
                "If that isn't the indicator you meant, please name it exactly."),
        "ar": ("طابقت سؤالك مع \"{indicator}\" (مطابقة تقريبية). "
                "إذا لم يكن هذا هو المؤشر المقصود، يرجى تحديد اسمه بدقة."),
    },
    "no_actuals_only_targets": {
        "en": ("No actual readings have been recorded for this indicator in the approved "
                "data — only targets (most recently {target} for {period})."),
        "ar": ("لم تُسجَّل أي قراءات فعلية لهذا المؤشر في البيانات المعتمدة — "
                "توجد أهداف فقط (آخرها {target} للفترة {period})."),
    },
}

_GRANULARITY_AR = {"monthly": "شهرية", "quarterly": "ربع سنوية", "yearly": "سنوية"}


def msg(key: str, language: str = "en", **kwargs) -> str:
    """Renders a message in the requested language, falling back to English."""
    entry = _MESSAGES.get(key)
    if not entry:
        return key
    lang = "ar" if is_arabic(language) else "en"
    template = entry.get(lang) or entry["en"]
    if lang == "ar":
        for field in ("granularity", "wanted"):
            if field in kwargs:
                kwargs[f"{field}_ar"] = _GRANULARITY_AR.get(str(kwargs[field]), kwargs[field])
    try:
        return template.format(**kwargs)
    except KeyError:
        return entry["en"].format(**kwargs)
