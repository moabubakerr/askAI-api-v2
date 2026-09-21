"""
Central configuration. All on-prem endpoints/paths live here so nothing
calls out to the public internet unless explicitly configured to.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM (self-hosted via vLLM, OpenAI-compatible server) ---
    LLM_BASE_URL: str = "http://localhost:8001/v1"   # your vLLM server
    LLM_MODEL_NAME: str = "Qwen2.5-72B-Instruct"      # or Llama-3.3-70B-Instruct
    LLM_API_KEY: str = "not-needed-for-vllm"          # vLLM ignores this by default
    LLM_TEMPERATURE: float = 0.1                       # low temp: this is data reporting, not creative writing
    LLM_MAX_TOKENS: int = 1500

    # A smaller/faster model for cheap tasks (routing/classification)
    ROUTER_MODEL_NAME: str = "Qwen2.5-7B-Instruct"

    # --- Embedding model (separate server) for semantic indicator matching ---
    # Plain string similarity cannot match paraphrases like "tourists arrived"
    # to the catalog's "Number of international visitors" — this needs a real
    # embedding model. bge-m3 or a similar multilingual model is recommended
    # since queries and indicator names both appear in English and Arabic.
    EMBEDDING_BASE_URL: str = "http://localhost:8002/v1"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # --- Databases ---
    POSTGRES_DSN: str = "postgresql://scai_ro:password@localhost:5432/scai_indicators"
    VECTOR_DB_PATH: str = "/data/scai/chroma_db"       # local Chroma persisted on disk

    # --- Safety / guardrails ---
    MAX_SQL_ROWS: int = 500
    ALLOWED_SQL_SCHEMAS: list[str] = ["public"]
    READ_ONLY_DB_ROLE: bool = True

    # --- Admin API ---
    # /admin/* reads the message log, which contains whatever users typed.
    # Left unset those routes refuse to serve: an empty key is a missing
    # decision, not a decision to publish.
    ADMIN_API_KEY: str = ""
    # Interactive login for the dashboard. One account for now; credentials
    # live here rather than in source so changing them is a config change and
    # not a deploy. ADMIN123 is a placeholder and must not survive contact with
    # real users — see the note in app/api/admin_auth.py.
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "ADMIN123"
    # How long a login lasts before the admin has to sign in again.
    ADMIN_SESSION_HOURS: int = 8
    # Rows per page by default, and the ceiling a caller may ask for. An
    # unbounded list over a table that grows with every question is a slow
    # query waiting to happen.
    ADMIN_PAGE_SIZE: int = 50
    ADMIN_MAX_PAGE_SIZE: int = 500

    # --- App ---
    APP_ENV: str = "on_prem"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
