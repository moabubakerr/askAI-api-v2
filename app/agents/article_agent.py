"""
Answers from SCAI article text.

Separate from the Composer because the guarantee is different. A numeric answer
is checked by comparing every number against a facts payload; prose has no such
payload, so the discipline here is grounding: the answer may only state what
the retrieved passages state, and each claim names the article it came from.

What is unchanged from the numeric path is the direction of the constraint: the
model is given text and asked to report it, never asked to recall. The passages
are the whole of its permitted knowledge, so "what does SCAI say about X" can
only ever be answered with what SCAI actually wrote.
"""
import json
from app.core.llm_client import chat, llm_client
from app.core.config import settings

SYSTEM_PROMPT = """You answer questions using ONLY the supplied excerpts from articles
published by Qatar's Supreme Council for Economic Affairs and Investment (SCAI).

ABSOLUTE RULES — violating any of these makes your answer unusable:
1. Use ONLY the supplied excerpts. You have no other knowledge of this subject.
   Do not add context, background, or well-known facts from your own training.
2. Every substantive claim must be attributable to an excerpt. Name the article
   you are drawing on in the prose, e.g. 'In "Financial Stability Implications of
   Tariffs and the Trade War", SCAI argues that ...'.
3. Copy figures, dates, programme names and organisation names exactly as they
   appear in the excerpts. Never adjust, round, convert or modernise them.
4. If the excerpts only partially address the question, answer the part they
   cover and say plainly what they do not. Do not fill the gap.
5. The excerpts were selected by relevance search and come from the article that
   best matches the question, so they are usually the right source even when the
   question's wording differs from theirs — "the trade war" and an article about
   tariffs are the same subject. Answer whenever they discuss the subject, even
   partially, and say which part you cannot cover.
   Refuse ONLY when they are plainly about a different subject: then say
   "SCAI's published articles don't cover this" and stop. Do not stretch a
   genuinely unrelated excerpt into an answer.
6. These are opinion and analysis pieces, not statistics. Attribute their claims
   to the article ("the article argues", "SCAI's analysis suggests") rather than
   stating them as established fact.
7. Do NOT write a "Sources:" section — one is appended automatically.
8. Answer in the requested language. The excerpts may be in either language;
   translate faithfully if they differ from the answer language, and do not add
   anything in the process.
9. Be direct. Lead with the answer, then the supporting detail.
"""


def answer_from_articles(question: str, passages: list[dict], language: str = "en") -> str:
    """passages: the retrieved chunks, each with title and content."""
    excerpts = [
        {
            "article_title": p.get("title_en") or p.get("title_ar"),
            "published": str(p.get("published_date") or p.get("article_date") or ""),
            "excerpt": p.get("content"),
        }
        for p in passages
    ]
    user_prompt = (
        f"Language for your answer: {language}\n\n"
        f"Question: {question}\n\n"
        f"Excerpts — the ONLY information you may use:\n"
        f"{json.dumps(excerpts, ensure_ascii=False, indent=2)}"
    )
    return chat(
        client=llm_client,
        model=settings.LLM_MODEL_NAME,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
    )
