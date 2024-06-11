# Standard Imports
import os
from dataclasses import dataclass, field

# Third Party Imports

# Internal Imports
from utils.utils import load_yaml


@dataclass
class ModelConfig:
    name: str
    model_path: str = None
    arguments: dict = field(default_factory=dict)


@dataclass
class StoreConfig:
    store_name: str
    builder_name: str = None
    path: str = None
    arguments: dict = field(default_factory=dict)


@dataclass
class Configuration:

    detector_model: ModelConfig = None

    embedding_model: ModelConfig = None

    image_store: StoreConfig = None

    database_path: str = None


def create_configurations() -> Configuration:

    config_path = os.environ.get("APP_CONFIG_PATH", None)
    if not config_path:
        return Configuration()

    config = load_yaml(config_path)

    config = Configuration(
        **{
            "image_store": StoreConfig(**config.get("image_store", {})),
            "detector_model": ModelConfig(**config.get("detector_model", {})),
            "embedding_model": ModelConfig(**config.get("embedding_model", {})),
            "database_path": os.environ.get("DATABASE_PATH", None),
        }
    )
    return config


app_config = create_configurations()
