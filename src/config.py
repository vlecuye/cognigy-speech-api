"""Configuration settings for Cognigy Gemini Speech Adapter."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallback."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Google Cloud & Vertex AI settings
    project_id: Optional[str] = "all-in-demo"
    location: str = "global"

    # Cognigy Voice Gateway authentication
    # If set, requests must contain 'Authorization: Bearer <cognigy_api_key>'
    cognigy_api_key: Optional[str] = None

    # TTS Model Configuration
    standard_tts_model: str = "gemini-3.8-flash-tts"
    live_tts_model: str = "gemini-3.1-flash-live-preview"

    # Default Telephony Audio Settings
    default_voice: str = "Puck"
    default_sample_rate: int = 8000  # Telephony standard 8000 Hz
    default_channels: int = 1         # Mono
    default_bits_per_sample: int = 16 # 16-bit linear PCM

    # Server settings
    port: int = 8080
    host: str = "0.0.0.0"
    log_level: str = "info"


settings = Settings()
