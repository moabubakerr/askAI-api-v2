"""
Thin wrapper around the self-hosted vLLM server.
vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint, so the
official `openai` python client works unchanged — just point base_url
at your own server. Nothing leaves the network.
"""
from openai import OpenAI
from app.core.config import settings

# Main "reasoning" client — bigger model for analysis/answer generation
llm_client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
)

# Optional: a second, smaller model for cheap classification tasks (routing).
# Point this at a second vLLM instance/port if you're running two models,
# or reuse the same client if you only have one model deployed.
router_client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
)


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Greedy decoding has no way out of a loop: at temperature 0 a model that starts
# repeating itself keeps picking the same most-likely continuation forever, and
# Qwen did — an Arabic answer restated itself eight times and drifted into
# Chinese. A small frequency penalty makes each repetition slightly less likely
# than the last, which breaks the cycle without making the output random.
#
# Deliberately small. The answers here legitimately repeat: a unit after every
# figure, a period after every line, an indicator name in the lead and again in
# a bullet. Penalising that hard would push the model into synonyms, which is
# the one thing rule 4 forbids.
REPETITION_PENALTY = 0.2


def chat(client: OpenAI, model: str, system: str, user: str,
         temperature: float = None, max_tokens: int = None) -> str:
    """Single-turn helper. Agents call this instead of touching the SDK directly,
    so retry/logging/guardrail logic lives in one place."""
    resp = client.chat.completions.create(
        model=model,
        messages=_messages(system, user),
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        frequency_penalty=REPETITION_PENALTY,
    )
    return resp.choices[0].message.content


def chat_stream(client: OpenAI, model: str, system: str, user: str,
                 temperature: float = None, max_tokens: int = None):
    """The same call, yielding text as the model produces it.

    vLLM speaks the OpenAI streaming protocol, so this needs no server-side
    change — only `stream=True` and a loop over the deltas.

    Yields raw fragments, which are NOT safe to show anyone. They arrive
    mid-word, mid-number and mid-markdown, and a number is only a claim once it
    is complete. app/compute/streaming.py is what decides when a fragment has
    become something a reader may see; nothing should consume this generator
    without going through it.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=_messages(system, user),
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        frequency_penalty=REPETITION_PENALTY,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece
