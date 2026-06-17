from functools import lru_cache
from logging import Logger, getLogger 
from pydantic_settings import BaseSettings, SettingsConfigDict

global_logger = getLogger()
global_logger.setLevel("INFO")

class Settings(BaseSettings):
    vantage_alpha_api_key: str
    vantage_alpha_base_url: str = "https://www.alphavantage.co/query"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_logger(self) -> Logger:
        return global_logger

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]