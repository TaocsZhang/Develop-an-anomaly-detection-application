from pydantic import BaseModel
from strictyaml import load
from typing import List


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    test_size: float
    VARS_WITH_MISSING: List[str]
    SCALED_VARS: List[str]

def fetch_config_from_yaml(cfg_path):
    """Parse YAML containing the package configuration."""

    with open(cfg_path, "r") as conf_file:
        parsed_config = load(conf_file.read())
        return parsed_config

cfg_path = 'config.yaml'
parsed_config = fetch_config_from_yaml(cfg_path)
config = ModelConfig(**parsed_config.data)