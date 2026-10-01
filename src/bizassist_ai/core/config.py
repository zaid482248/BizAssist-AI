"""
Application configuration.
Yeh file saari settings rakhti hai (app name, version, debug mode).
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "BizAssist_AI"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()