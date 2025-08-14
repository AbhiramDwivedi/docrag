"""Central configuration loader."""
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
import os
from typing import Dict, Any, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")

class Settings(BaseModel):
    sync_root: Path = Field(default=Path.home() / "Documents", description="Local document folder to watch and index")
    db_path: Path = Field(default=Path("data/docmeta.db"))
    vector_path: Path = Field(default=Path("data/vector.index"))
    knowledge_graph_path: Path = Field(default=Path("data/knowledge_graph.db"))
    chunk_size: int = 800
    overlap: int = 150
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    openai_api_key: Optional[str] = None
    
    # Phase 1: Retrieval robustness parameters
    retrieval_k: int = Field(default=100, ge=1, description="Number of initial chunks to retrieve before MMR")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR balance: 1.0=pure relevance, 0.0=pure diversity")
    mmr_k: int = Field(default=20, ge=1, description="Final number of chunks to return after MMR selection")
    proper_noun_boost: float = Field(default=0.3, ge=0.0, le=1.0, description="Boost factor for metadata matches on proper noun queries")
    min_similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold for considering results relevant")
    enable_debug_logging: bool = Field(default=False, description="Enable detailed debug logging for retrieval pipeline")
    enable_debug_logging: bool = Field(default=False, description="Enable detailed debug logging for retrieval pipeline")

    @field_validator('mmr_k')
    @classmethod
    def validate_mmr_k_vs_retrieval_k(cls, v, info):
        """Ensure mmr_k <= retrieval_k."""
        if 'retrieval_k' in info.data and v > info.data['retrieval_k']:
            raise ValueError(f"mmr_k ({v}) cannot be greater than retrieval_k ({info.data['retrieval_k']})")
        return v

    @field_validator('sync_root', mode='before')
    @classmethod
    def expand_sync_root(cls, v: Any) -> Path:
        """Expand tilde and resolve path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v

    def resolve_storage_path(self, path: Path) -> Path:
        """Resolve storage paths to absolute, create directories if needed.
        
        Args:
            path: Storage path (relative or absolute)
            
        Returns:
            Resolved absolute path with directories created
        """
        if path.is_absolute():
            resolved = path
        else:
            resolved = Path.cwd() / path
        
        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

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
