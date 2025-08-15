"""Central configuration loader."""
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo
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
    embed_model_version: str = Field(default="1.0.0", description="Version identifier for embedding model compatibility")
    batch_size: int = 32
    openai_api_key: Optional[str] = None
    
    @field_validator('embed_model_version')
    @classmethod
    def validate_model_version_consistency(cls, v: str, info: ValidationInfo) -> str:
        """Validate that model version is consistent with model name."""
        # Validate version format (basic semantic versioning check)
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(f"Model version '{v}' must follow semantic versioning format (x.y.z)")
        
        # Only validate consistency if embed_model is also being set
        if 'embed_model' in info.data:
            model_name = info.data['embed_model']
            
            # Validate model_name is not empty
            if not model_name or not isinstance(model_name, str):
                raise ValueError("embed_model must be a non-empty string")
            
            # Define version mappings for known models
            known_model_versions = {
                "sentence-transformers/all-MiniLM-L6-v2": ["1.0.0"],
                "intfloat/e5-base-v2": ["2.0.0"],
                "intfloat/e5-small-v2": ["2.0.0"],
                "intfloat/e5-large-v2": ["2.0.0"],
                "BAAI/bge-small-en-v1.5": ["2.0.0"],
                "BAAI/bge-base-en-v1.5": ["2.0.0"],
                "BAAI/bge-large-en-v1.5": ["2.0.0"],
                "thenlper/gte-base": ["2.0.0"],
                "thenlper/gte-small": ["2.0.0"],
                "thenlper/gte-large": ["2.0.0"]
            }
            
            if model_name in known_model_versions:
                expected_versions = known_model_versions[model_name]
                if v not in expected_versions:
                    # Allow version override with warning, don't fail
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Model version '{v}' may not be compatible with model '{model_name}'. "
                        f"Expected versions: {expected_versions}. "
                        f"This may cause compatibility issues with embeddings and migration."
                    )
            else:
                # Unknown model - issue info warning
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Unknown embedding model '{model_name}' with version '{v}'. "
                    f"Ensure compatibility manually."
                )
        
        return v
    
    # Phase 1: Retrieval robustness parameters
    retrieval_k: int = Field(default=100, ge=1, description="Number of initial chunks to retrieve before MMR")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR balance: 1.0=pure relevance, 0.0=pure diversity")
    mmr_k: int = Field(default=20, ge=1, description="Final number of chunks to return after MMR selection")
    proper_noun_boost: float = Field(default=0.3, ge=0.0, le=1.0, description="Boost factor for metadata matches on proper noun queries")
    min_similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold for considering results relevant")
    enable_debug_logging: bool = Field(default=False, description="Enable detailed debug logging for retrieval pipeline")

    @field_validator('mmr_k')
    @classmethod
    def validate_mmr_k_vs_retrieval_k(cls, v: int, info: ValidationInfo) -> int:
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
    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

settings = SettingsProxy()
