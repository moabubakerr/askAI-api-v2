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


# Words too common in a data question to say anything about language. "Q2 2025"
# and "GDP 2024" are the same message in both halves of this product.
_LANGUAGE_NEUTRAL = {"gdp", "cpi", "fdi", "q1", "q2", "q3", "q4", "top", "vs",
                      "ok", "yes", "no", "pls", "plz"}


def carries_language_signal(text: str) -> bool:
    """Whether this message says anything about what language it is in.

    detect_language reads the script, which is the right test for a message that
    HAS script to read. A follow-up often does not: "2024", "Q2 2025", "top 3",
    "GDP" carry no Arabic characters and no meaningful English either, and the
    detector answered "en" for all of them — so an Arabic conversation switched
    to English the moment someone typed a year into it.

    Arabic script is always a signal. Otherwise it takes two or more ordinary
    words: one word can be an indicator name or an abbreviation that is written
    identically whichever language the reader is working in.
    """
    if not text:
        return False
    if _ARABIC_CHARS.search(text):
        return True
    words = [w for w in re.findall(r"[A-Za-z]{2,}", text)
             if w.lower() not in _LANGUAGE_NEUTRAL]
    return len(words) >= 2


def answers_in_language(text: str, language: str) -> bool:
    """Is this reply actually written in the language that was asked for?

    Only Arabic is checked, and only for its presence. An Arabic answer
    legitimately contains Latin script — indicator names are reproduced
    verbatim by design, so "بلغ Real GDP 185.17 مليار ر.ق" is correct — which
    means a ratio test would reject good answers. What cannot happen is an
    Arabic question answered with no Arabic in it at all.

    The English direction is deliberately not checked: an English answer
    quoting an Arabic indicator name is fine, and there is no comparable
    failure to catch.
    """
    if not is_arabic(language):
        return True
    return bool(text) and bool(_ARABIC_CHARS.search(text))


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
    # The same courtesy mid-conversation, without the introduction. Replying to
    # "thanks" on turn nine with the full five-line menu of what the assistant
    # covers reads as though the previous eight turns did not happen.
    "greeting_again": {
        "en": "You're welcome — ask me anything else about the published indicators.",
        "ar": "على الرحب والسعة — اسألني عن أي شيء آخر بخصوص المؤشرات المنشورة.",
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
        "en": ("No indicator in SCAI's approved data matches \"{phrase}\".\n\n"
                "Two things that usually help:\n"
                "• name the metric more plainly — \"inflation\", \"Real GDP\", "
                "\"trade balance\", \"government revenues\"\n"
                "• or ask \"what can you do?\" for the full list of what's published"),
        "ar": ("لا يوجد مؤشر في بيانات المجلس المعتمدة يطابق \"{phrase}\"\n\n"
                "عادةً ما يساعد أحد أمرين:\n"
                "• اذكر اسم المؤشر بصيغة أبسط — \"التضخم\"، \"الناتج المحلي الإجمالي الحقيقي\"، "
                "\"الميزان التجاري\"، \"الإيرادات الحكومية\"\n"
                "• أو اسأل \"ماذا يمكنك أن تفعل؟\" للاطلاع على القائمة الكاملة لما هو منشور"),
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
    # Appended when the missing period is in the FUTURE. Asking for 2026 and
    # being told the data stops at 2025-Q4 is accurate but incomplete: it reads
    # as a gap that might be filled, when the real answer is that this system
    # reports published figures and does not project. Without this, the same
    # user got a forecast explanation for "GDP forecast 2026" and a bare
    # date-range reply for "GDP growth for 2026" — two different answers to one
    # underlying limitation.
    "future_period_suffix": {
        "en": (" That period is in the future: I report published figures and don't "
                "forecast. Where SCAI has set a TARGET for a future period I can give "
                "that instead — it's a policy goal, not a prediction."),
        "ar": (" هذه الفترة مستقبلية: أعرض الأرقام المنشورة ولا أقدّم تنبؤات. وإذا كان "
                "المجلس قد حدد هدفاً لفترة مستقبلية يمكنني عرضه — وهو غاية سياسات لا تنبؤ."),
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
    # A calculation the user has done and wants confirmed, where the two
    # figures are measured over different populations.
    "derived_different_bases": {
        "en": ("No — those two figures cannot be multiplied together. \"{a}\" is "
                "measured over {base_a}, while \"{b}\" counts {base_b}. Those are "
                "different populations, so the result would assume something the "
                "data does not say. I can give you either figure on its own, or "
                "the indicator that measures what you are after if SCAI publishes "
                "one."),
        "ar": ("لا — لا يمكن ضرب هذين الرقمين معاً. فـ\"{a}\" يُقاس على {base_a}، "
                "بينما \"{b}\" يحصي {base_b}. وهما مجموعتان مختلفتان، لذا فإن الناتج "
                "يفترض ما لا تقوله البيانات. يمكنني تقديم أي من الرقمين منفرداً، أو "
                "المؤشر الذي يقيس ما تبحث عنه إن كان المجلس ينشره."),
    },
    "derived_not_supported": {
        "en": ("I report figures as SCAI publishes them and don't combine indicators "
                "to derive new ones, because the result depends on assumptions the "
                "data does not state — most often that two figures cover the same "
                "population or period. Ask me for either indicator on its own and "
                "I'll give you the published value."),
        "ar": ("أعرض الأرقام كما ينشرها المجلس ولا أجمع بين المؤشرات لاشتقاق أرقام "
                "جديدة، لأن الناتج يعتمد على افتراضات لا تذكرها البيانات — غالباً أن "
                "الرقمين يغطيان المجموعة أو الفترة نفسها. اسألني عن أي مؤشر منفرداً "
                "وسأعطيك القيمة المنشورة."),
    },
    # Asked to rank by performance with no group in play — either the question
    # named no sector, or it is a follow-up to a turn that was not a listing.
    "performance_needs_scope": {
        "en": ("Tell me which group to rank and I'll order it by progress against target — "
                "a sector such as Education, Tourism or Manufacturing, or an indicator type "
                "such as Economic Diversification Targets."),
        "ar": ("أخبرني بالمجموعة التي تريد ترتيبها وسأرتّبها حسب التقدّم نحو المستهدف — "
                "قطاع مثل التعليم أو السياحة أو الصناعة التحويلية، أو نوع مؤشر مثل "
                "مستهدفات التنويع الاقتصادي."),
    },
    # Replaces a bare "No indicator was mentioned." — technically true, useless
    # in practice, and what an unrecognised greeting used to be answered with.
    "no_indicator_in_question": {
        "en": ("I have the period you asked about, but not which indicator you want for it.\n\n"
                "Add the metric and I can answer — for example:\n"
                "• \"Real GDP in 2024\"\n"
                "• \"inflation in May 2025\"\n"
                "• \"government revenues last year\"\n\n"
                "Or ask \"what can you do?\" to see everything that's published."),
        "ar": ("لديّ الفترة التي سألت عنها، لكن ليس المؤشر الذي تريده لها.\n\n"
                "أضف اسم المؤشر وسأجيبك — مثل:\n"
                "• \"الناتج المحلي الإجمالي الحقيقي في 2024\"\n"
                "• \"التضخم في مايو 2025\"\n"
                "• \"الإيرادات الحكومية العام الماضي\"\n\n"
                "أو اسأل \"ماذا يمكنك أن تفعل؟\" للاطلاع على كل ما هو منشور."),
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
    # Notes attached to a country comparison. They were built as English
    # f-strings in the payload, which the Composer translates as prose but the
    # template fallback prints verbatim — so a rejected Arabic answer ended in an
    # English sentence about how the comparison had been aligned.
    "compared_at": {
        "en": ("Compared at {period}, the most recent period all of these "
                "countries report."),
        "ar": "تمت المقارنة عند {period}، وهي أحدث فترة تنشرها جميع هذه الدول.",
    },
    "no_common_period": {
        "en": ("These countries report no period in common, so each figure below "
                "is that country's own latest ({first} to {last}). This is not a "
                "like-for-like comparison."),
        "ar": ("لا تشترك هذه الدول في أي فترة، لذا فإن كل رقم أدناه هو أحدث قراءة "
                "لتلك الدولة ({first} إلى {last}). وهذه ليست مقارنة متكافئة."),
    },
    "measured_in": {
        "en": "\n\nMeasured in {unit}.",
        "ar": "\n\nوحدة القياس: {unit}.",
    },
    "no_country_breakdown": {
        "en": ("\"{indicator}\" is reported for Qatar as a whole — the approved data has no "
                "breakdown by country, so there is no {countries} figure to give. Ask me for "
                "the overall total and I can answer that."),
        "ar": ("\"{indicator}\" يُنشر لدولة قطر ككل — ولا تتضمن البيانات المعتمدة تفصيلاً "
                "حسب الدولة، لذا لا يوجد رقم خاص بـ {countries}. يمكنك أن تسألني عن الإجمالي "
                "وسأجيبك."),
    },
    "no_forecast": {
        "en": ("I don't produce forecasts — I report figures SCAI has already published, "
                "and the approved data contains no projection for {topic}. What it does "
                "carry for some indicators is SCAI's own TARGET for a future period, which "
                "is a policy goal rather than a prediction. Ask for a target, or for the "
                "latest actual reading, and I can answer either."),
        "ar": ("لا أقدّم تنبؤات — أعرض الأرقام التي نشرها المجلس، ولا تتضمن البيانات "
                "المعتمدة أي إسقاط لـ {topic}. ما تتضمنه لبعض المؤشرات هو هدف المجلس "
                "لفترة مستقبلية، وهو غاية سياسات لا تنبؤ. اسألني عن الهدف أو عن أحدث "
                "قراءة فعلية وسأجيبك."),
    },
    "none_in_period": {
        "en": ("None of the indicators you asked for have approved data for {period}. "
                "Try a period closer to the present, or ask for one of them on its own "
                "and I'll tell you the range it covers."),
        "ar": ("لا يوجد لدى أي من المؤشرات التي طلبتها بيانات معتمدة للفترة {period}. "
                "جرّب فترة أقرب إلى الحاضر، أو اسأل عن أحدها بمفرده وسأخبرك بالمدى الذي يغطيه."),
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
