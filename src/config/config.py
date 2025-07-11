import os


class Config:
    # Database configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "translation_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Translation settings
    DEFAULT_BATCH_SIZE: int = 10
    MAX_PARALLEL_BATCHES: int = 4

    # LLM provider configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    @property
    def LLM_MODEL(self) -> str:
        # Use different default models based on provider
        provider = self.LLM_PROVIDER.lower()
        if provider == "ollama":
            return os.getenv("LLM_MODEL", "gemma3:27b")
        elif provider in ["google", "vertexai"]:
            return os.getenv("LLM_MODEL", "gemini-2.5-pro")
        else:
            return os.getenv("LLM_MODEL", "gpt-4.1")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Google Vertex AI configuration
    # Path to service account key file. If not set, application default credentials are used.
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "")
    GOOGLE_LOCATION: str = os.getenv("GOOGLE_LOCATION", "us-central1")

    # Caching strategy
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() == "true"
    CACHE_EXPIRATION: int = 86400  # 24 hours in seconds

    @property
    def postgres_uri(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def redis_uri(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


config = Config()
