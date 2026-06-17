from functools import lru_cache
from logging import Logger, getLogger
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

global_logger = getLogger()
global_logger.setLevel("INFO")

class Settings(BaseSettings):
    vantage_alpha_api_key: str
    vantage_alpha_base_url: str = "https://www.alphavantage.co/query"

    # Which LLM backend the agent talks to. Switch by setting MODEL_PROVIDER
    # in .env — no code changes needed.
    model_provider: Literal["google", "anthropic"] = "anthropic"
    google_model: str = "gemini-flash-latest"
    anthropic_model: str = "claude-opus-4-8"

    # Which market-data source tools pull from. Switch by setting
    # DATA_PROVIDER in .env. yfinance has no API key and a much looser rate
    # limit than Alpha Vantage's free tier.
    data_provider: Literal["alpha_vantage", "yfinance"] = "yfinance"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def model_id(self) -> str:
        return self.anthropic_model if self.model_provider == "anthropic" else self.google_model

    def get_logger(self) -> Logger:
        return global_logger

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]