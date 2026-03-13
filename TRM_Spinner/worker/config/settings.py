from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Look for .env in worker/ first, then parent TRM_Spinner/
_worker_dir = Path(__file__).resolve().parent.parent
_env_files = [_worker_dir / ".env", _worker_dir.parent / ".env"]
_env_file = next((f for f in _env_files if f.exists()), ".env")


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=str(_env_file), extra="ignore")

    appwrite_endpoint: str = "https://cloud.appwrite.io/v1"
    appwrite_project_id: str = ""
    appwrite_api_key: str = ""
    appwrite_database_id: str = "trm_spinner"
    redis_url: str = "redis://localhost:6379"
    admin_password: str = "Jack123123@!"
    openai_api_key: str = ""


settings = Settings()
