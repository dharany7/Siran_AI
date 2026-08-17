"""
config.py — centralised settings loaded from .env via python-dotenv.
Never hardcode secrets; always read from environment variables.

Gemini client usage (new google-genai SDK):
    from backend.config import get_gemini_client
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Hello!",
    )
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "Siren AI"
    debug: bool = False

    # --- Database ---
    database_url: str = "postgresql://siren:siren@localhost:5432/siren_ai"

    # --- Security ---
    app_secret_key: str = "change-me-in-dotenv"

    # --- JWT ---
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 7

    # --- Google Gemini (new google-genai SDK) ---
    # Set GOOGLE_GEMINI_API_KEY in your .env file — never hardcode.
    google_gemini_api_key: str = ""

    # Default model to use for generate_content calls
    gemini_model: str = "gemini-3.6-flash"  # confirmed working; override via GEMINI_MODEL in .env

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once)."""
    return Settings()


@lru_cache
def get_gemini_client():
    """
    Return a cached ``google.genai.Client`` instance.

    Uses the new ``google-genai`` package (``from google import genai``).
    The old ``google-generativeai`` / ``google.generativeai`` package is
    NOT used anywhere in this codebase.

    Example usage::

        from backend.config import get_gemini_client, get_settings
        client = get_gemini_client()
        response = client.models.generate_content(
            model=get_settings().gemini_model,
            contents="Describe this siren event.",
        )
        print(response.text)

    Raises:
        ValueError: if GOOGLE_GEMINI_API_KEY is not set in .env.
    """
    from google import genai  # google-genai package

    api_key = get_settings().google_gemini_api_key
    if not api_key:
        raise ValueError(
            "GOOGLE_GEMINI_API_KEY is not set. "
            "Add it to your .env file and never hardcode it."
        )
    return genai.Client(api_key=api_key)


# Convenience alias used throughout the codebase
settings: Settings = get_settings()
