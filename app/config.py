from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini/gemini-3.8-flash"
    GEMINI_FALLBACK_MODELS: str = "gemini/gemini-3.6-flash,gemini/gemini-3.5-flash"
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_BACKOFF_SECONDS: float = 2.0
    QDRANT_HOST: str = "localhost"

    # LangSmith Tracing Parameters
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "compliance-auditor"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
