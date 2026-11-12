from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    QDRANT_HOST: str = "localhost"

    # LangSmith Tracing Parameters
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "compliance-auditor"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
