"""Unified configuration package.

Exposes two interfaces, both previously split across app/core/config.py and
src/config/:
  - Flask app config:  Config, get_config  (Flask config *classes*)
  - Runtime settings:  config, paths       (singletons with .CHUNK_SIZE, .DATA_DIR, ...)
"""

from ragbot.config.flask_config import (
    Config,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    get_config,
)
from ragbot.config.paths import PathConfig, paths
from ragbot.config.settings import AppConfig, config
from ragbot.config.utils import (
    ensure_environment_setup,
    load_environment,
    validate_environment,
)

__all__ = [
    "Config",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
    "get_config",
    "PathConfig",
    "paths",
    "AppConfig",
    "config",
    "ensure_environment_setup",
    "load_environment",
    "validate_environment",
]
