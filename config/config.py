"""Central configuration loader."""
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
import os
from typing import Dict, Any, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")

class Settings(BaseModel):
    sync_root: Path = Field(..., description="Local OneDrive/SharePoint folder")
    db_path: Path
    vector_path: Path
    chunk_size: int = 800
    overlap: int = 150
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    openai_api_key: Optional[str] = None

    @field_validator('sync_root', mode='before')
    @classmethod
    def expand_sync_root(cls, v: Any) -> Path:
        """Expand tilde and resolve path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v

def load_settings(overrides: Optional[Dict[str, Any]] = None) -> Settings:
    data: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        yaml_data = yaml.safe_load(CONFIG_PATH.read_text())
        if yaml_data:
            data.update(yaml_data)
    env_map: Dict[str, Optional[str]] = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "sync_root": os.getenv("SYNC_ROOT")
    }
    data.update({k: v for k, v in env_map.items() if v})
    if overrides:
        data.update(overrides)
    return Settings(**data)

settings = load_settings()
