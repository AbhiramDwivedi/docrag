"""Central configuration loader."""
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
import os
from typing import Dict, Any, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")

class Settings(BaseModel):
    sync_root: Path = Field(default=Path.home() / "Documents", description="Local document folder to watch and index")
    db_path: Path = Field(default=Path("backend/data/docmeta.db"))
    vector_path: Path = Field(default=Path("backend/data/vector.index"))
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

# Global settings instance - lazy loaded
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get settings instance, loading on first access."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings

# For backward compatibility - create a proxy object that behaves like the settings
class SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)

settings = SettingsProxy()
