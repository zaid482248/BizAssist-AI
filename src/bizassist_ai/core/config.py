"""
Application configuration.
Yeh file saari settings rakhti hai (app name, version, upload limits, etc.).
"""

from pathlib import Path

from pydantic_settings import BaseSettings


# Project root directory (BizAssist_AI folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App info
    app_name: str = "BizAssist_AI"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # File upload settings
    max_upload_size_mb: int = 20  # 20 MB limit
    allowed_extensions: list[str] = [".pdf", ".txt", ".docx", ".md"]
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # Data directories
    data_dir: Path = PROJECT_ROOT / "data"
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()

# Ensure upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)