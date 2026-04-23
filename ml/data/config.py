"""Configuration loader for the AquaSite ML pipeline.

Reads ml/config.yaml for structural config and .env for secrets (Supabase URL/key).
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, field_validator


# Project root: AquaSite/
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class LocalConfig(BaseModel):
    image_dirs: List[str]
    metadata_csv: Optional[str] = None
    labels_csv: Optional[str] = None


class SupabaseConfig(BaseModel):
    use_django_media: bool = True
    download_to: str = "./ml/outputs/supabase_cache"


class ModelConfig(BaseModel):
    name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    input_size: int = 224
    freeze_backbone: bool = True
    embedding_dim: int = 768


class TrainingConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.001
    epochs: int = 20
    device: str = "auto"
    checkpoint_dir: str = "./ml/outputs/checkpoints"
    log_dir: str = "./ml/outputs/logs"
    sample_mode: str = "box"            # "image" = one sample per AnalysisImage (majority label)
                                        # "box"   = one sample per labeled BoundingBox (crop)
    min_box_fraction: float = 0.005     # box mode: skip boxes below this fraction of image area

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v):
        if v == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

    @field_validator("sample_mode")
    @classmethod
    def check_sample_mode(cls, v):
        v = v.lower()
        if v not in {"image", "box"}:
            raise ValueError(f"training.sample_mode must be 'image' or 'box', got {v!r}")
        return v


class EmbeddingConfig(BaseModel):
    output_dir: str = "./ml/outputs/embeddings"
    batch_size: int = 64


class ClassifierConfig(BaseModel):
    type: str = "linear"  # "linear" | "mlp"
    hidden_dims: List[int] = []  # MLP only; ignored when type="linear"
    dropout: float = 0.3  # MLP only

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        v = v.lower()
        if v not in {"linear", "mlp"}:
            raise ValueError(f"classifier.type must be 'linear' or 'mlp', got {v!r}")
        return v


class MLConfig(BaseModel):
    data_source: str = "local"  # "local" | "supabase"
    local: LocalConfig = LocalConfig(image_dirs=[])
    supabase: SupabaseConfig = SupabaseConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    classifier: ClassifierConfig = ClassifierConfig()

    # Populated from .env at load time
    supabase_url: str = ""
    supabase_key: str = ""
    hf_token: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> MLConfig:
    """Load ML config from YAML file + environment secrets."""
    if config_path is None:
        config_path = str(BASE_DIR / "ml" / "config.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = MLConfig(**raw)

    # Load secrets from environment / .env
    # python-decouple is already installed; Django settings may have loaded .env
    try:
        from decouple import config as env_config
        config.supabase_url = env_config("SUPABASE_URL", default="")
        config.supabase_key = env_config("SUPABASE_KEY", default="")
        config.hf_token = env_config("HF_TOKEN", default=None)
    except Exception:
        config.supabase_url = os.environ.get("SUPABASE_URL", "")
        config.supabase_key = os.environ.get("SUPABASE_KEY", "")
        config.hf_token = os.environ.get("HF_TOKEN")

    # Resolve relative paths against BASE_DIR
    config.training.checkpoint_dir = str(
        (BASE_DIR / config.training.checkpoint_dir).resolve()
    )
    config.training.log_dir = str(
        (BASE_DIR / config.training.log_dir).resolve()
    )
    config.embedding.output_dir = str(
        (BASE_DIR / config.embedding.output_dir).resolve()
    )
    config.supabase.download_to = str(
        (BASE_DIR / config.supabase.download_to).resolve()
    )

    return config
