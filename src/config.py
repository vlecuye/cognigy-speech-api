"""Configuration settings for Cognigy Gemini Speech Adapter."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallback."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Google Cloud & Vertex AI settings
    # If not explicitly set via PROJECT_ID or .env, auto-detects from the ambient GCP environment
    project_id: Optional[str] = None
    location: str = "europe-west1"

    # Cognigy Voice Gateway authentication
    # If set, requests must contain 'Authorization: Bearer <cognigy_api_key>'
    cognigy_api_key: Optional[str] = None

    # TTS Model Configuration
    standard_tts_model: str = "gemini-2.5-flash-tts"
    live_tts_model: str = "gemini-live-2.5-flash-native-audio"

    # STT Provider Configuration ('google' or 'gemini')
    stt_provider: str = "google"

    # Gemini STT Model Configuration (Gemini 3.5 Transcribe Live in Europe)
    transcribe_model: str = "gemini-3.5-transcribe-live"
    transcribe_location: str = "eu"
    transcribe_language: str = "en-US"
    transcribe_sample_rate: int = 8000

    # Google Cloud Speech-to-Text Configuration (Chirp / Telephony)
    google_speech_model: str = "telephony"
    google_speech_location: str = "eu"


    # Default Telephony Audio Settings
    default_voice: str = "Puck"
    default_sample_rate: int = 8000  # Telephony standard 8000 Hz
    default_channels: int = 1         # Mono
    default_bits_per_sample: int = 16 # 16-bit linear PCM

    # Server settings
    port: int = 8080
    host: str = "0.0.0.0"
    log_level: str = "info"

    def model_post_init(self, __context) -> None:
        if not self.project_id:
            import os
            self.project_id = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.project_id:
                try:
                    import google.auth
                    _, detected_project = google.auth.default()
                    if detected_project:
                        self.project_id = detected_project
                except Exception:
                    pass


settings = Settings()

