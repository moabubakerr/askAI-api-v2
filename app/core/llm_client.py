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


def chat(client: OpenAI, model: str, system: str, user: str,
         temperature: float = None, max_tokens: int = None) -> str:
    """Single-turn helper. Agents call this instead of touching the SDK directly,
    so retry/logging/guardrail logic lives in one place."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content
