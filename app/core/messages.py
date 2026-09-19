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
import unicodedata
from difflib import SequenceMatcher
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


# --- greeting detection -----------------------------------------------------
#
# Classifying greetings is left to the intent agent, and it is inconsistent at
# it: "اهلا" and "كيف حالك" were routed to general_chat, while "سلام" and
# "سلام علبكم" were routed to a data lookup, fell through to indicator
# resolution and came back "No indicator was mentioned." Greeting someone and
# being told you failed to name an indicator is a bad first impression, and it
# is avoidable — this is a closed set of short phrases, not a judgement call,
# so it is decided deterministically before the model is consulted.

_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")

_GREETING_PHRASES = {
    # Arabic (normalized forms)
    "السلام عليكم", "السلام عليكم ورحمه الله", "السلام عليكم ورحمه الله وبركاته",
    "وعليكم السلام", "كيف حالك", "كيف الحال", "شلونك", "صباح الخير", "مساء الخير",
    "اهلا وسهلا", "حياك الله", "تحياتي", "شكرا", "شكرا لك", "مع السلامه",
    # English
    "hi", "hello", "hey", "hiya", "yo", "greetings", "good morning",
    "good afternoon", "good evening", "how are you", "how are you doing",
    "whats up", "thanks", "thank you", "thank you very much", "bye", "goodbye",
    "salam", "salaam", "assalamu alaikum", "as salamu alaykum",
}

# A message may START with one of these and still be only a greeting.
_GREETING_STARTERS = {
    "سلام", "السلام", "اهلا", "هلا", "مرحبا", "مرحبتين", "صباح", "مساء",
    "تحيه", "شكرا", "وعليكم",
    "hi", "hello", "hey", "salam", "salaam", "greetings", "thanks", "thank",
}

# Tokens that may legitimately follow a greeting without making it a question.
_GREETING_CONTINUATIONS = {
    "عليكم", "ورحمه", "الله", "وبركاته", "وسهلا", "الخير", "بك", "بكم", "لك",
    "you", "there", "everyone", "all", "morning", "afternoon", "evening", "again",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _ARABIC_DIACRITICS.sub("", text)
    # Unify the alef/ya/ta-marbuta variants that Arabic typing varies freely.
    for src, dst in (("أإآٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        for ch in src:
            text = text.replace(ch, dst)
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _close(a: str, b: str, cutoff: float = 0.8) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= cutoff


def is_greeting(text: str) -> bool:
    """True when the message is ONLY a greeting.

    Fuzzy on each token, because "سلام علبكم" is a transposition of
    "سلام عليكم" and a user who mistypes a greeting should still be greeted.

    Deliberately conservative about length: every token after the opening
    greeting must itself be a greeting word, so "مرحبا ما هو التضخم" is a
    question that opens politely, not a greeting, and is answered as a
    question.
    """
    normalized = _normalize(text)
    if not normalized or len(normalized) > 60:
        return False
    if normalized in _GREETING_PHRASES:
        return True
    if any(_close(normalized, phrase, 0.9) for phrase in _GREETING_PHRASES):
        return True

    tokens = normalized.split()
    if not tokens or len(tokens) > 6:
        return False
    if not any(_close(tokens[0], starter) for starter in _GREETING_STARTERS):
        return False
    allowed = _GREETING_STARTERS | _GREETING_CONTINUATIONS
    return all(any(_close(token, word) for word in allowed) for token in tokens[1:])


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
    # Replaces a bare "No indicator was mentioned." — technically true, useless
    # in practice, and what an unrecognised greeting used to be answered with.
    "no_indicator_in_question": {
        "en": ("I couldn't tell which indicator you're asking about. Name the metric — "
                "for example \"Real GDP\", \"Inflation\" or \"Number of International "
                "Visitors\" — or ask \"what can you do?\" to see what's covered."),
        "ar": ("لم أتبيّن المؤشر الذي تسأل عنه. اذكر اسم المؤشر — مثل \"الناتج المحلي "
                "الإجمالي الحقيقي\" أو \"التضخم\" أو \"عدد الزوار الدوليين\" — أو اسأل "
                "\"ماذا يمكنك أن تفعل؟\" لمعرفة ما هو مشمول."),
    },
    "empty_catalog": {
        "en": ("The indicator catalog is empty — the database has no indicator data loaded. "
                "This is a setup problem, not a limit of what you asked."),
        "ar": ("كتالوج المؤشرات فارغ — لا توجد بيانات مؤشرات محمّلة في قاعدة البيانات. "
                "هذه مشكلة في الإعداد وليست حدوداً لما سألت عنه."),
    },
    "no_articles_found": {
        "en": ("SCAI's published articles don't appear to cover \"{topic}\". I searched the "
                "article text and found nothing close enough to answer from — rather than "
                "stretch a loosely-related piece into an answer. Try a broader topic, or ask "
                "me for an indicator instead."),
        "ar": ("لا يبدو أن مقالات المجلس المنشورة تتناول \"{topic}\". بحثت في نصوص المقالات "
                "ولم أجد ما يكفي قربه للإجابة منه، بدلاً من تحميل مقال غير وثيق الصلة ما لا "
                "يحتمل. جرّب موضوعاً أوسع، أو اسألني عن مؤشر."),
    },
    "no_article_topic": {
        "en": ("I couldn't tell which topic you'd like me to look up in SCAI's articles. "
                "Name the subject — for example \"the trade war\", \"In-Country Value\" or "
                "\"knowledge transfer\"."),
        "ar": ("لم أتبيّن الموضوع الذي تريد البحث عنه في مقالات المجلس. حدّد الموضوع — "
                "مثل \"الحرب التجارية\" أو \"القيمة المحلية\" أو \"نقل المعرفة\"."),
    },
    "articles_not_indexed": {
        "en": ("The article text hasn't been indexed yet, so I can't search it. This is a "
                "setup step (index-articles), not a gap in what SCAI has published."),
        "ar": ("لم تتم فهرسة نصوص المقالات بعد، لذا لا يمكنني البحث فيها. هذه خطوة إعداد "
                "(index-articles) وليست نقصاً فيما نشره المجلس."),
    },
    "ungrounded_numbers": {
        "en": ("Note: the figures {numbers} above do not appear in the article excerpts I "
                "retrieved, so treat them with caution — check the linked articles directly."),
        "ar": ("ملاحظة: الأرقام {numbers} أعلاه لا ترد في مقتطفات المقالات التي استرجعتها، "
                "لذا تعامل معها بحذر — راجع المقالات المرتبطة مباشرة."),
    },
    "no_data_for_country": {
        "en": ("There is no approved data for {country} on \"{indicator}\". I won't show "
                "Qatar's figure in its place."),
        "ar": ("لا توجد بيانات معتمدة لـ {country} بخصوص \"{indicator}\". "
                "ولن أعرض رقم قطر بدلاً منه."),
    },
    "need_two_periods": {
        "en": ("I need two specific periods to compare — for example \"between Q1 2025 and "
                "Q4 2025\" or \"2023 versus 2024\". Both have to be the same kind of period."),
        "ar": ("أحتاج إلى فترتين محددتين للمقارنة — مثل \"بين الربع الأول 2025 والربع الرابع "
                "2025\" أو \"2023 مقابل 2024\". ويجب أن تكونا من النوع نفسه."),
    },
    "no_analysis": {
        "en": ("SCAI hasn't published written analysis for \"{indicator}\" in the approved "
                "data. I can give you its values or trend instead."),
        "ar": ("لم ينشر المجلس تحليلاً مكتوباً لـ \"{indicator}\" ضمن البيانات المعتمدة. "
                "يمكنني عرض قيمه أو اتجاهه بدلاً من ذلك."),
    },
    "measured_in": {
        "en": "\n\nMeasured in {unit}.",
        "ar": "\n\nوحدة القياس: {unit}.",
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
