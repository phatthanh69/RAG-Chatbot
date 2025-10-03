"""
Configuration package for the RAG chatbot system.
Contains centralized configuration management for all input/output directories and settings.
"""

from .paths import PathConfig, paths
from .settings import AppConfig, config
from .utils import (
    load_environment,
    validate_environment,
    print_full_configuration,
    get_config_summary,
    create_sample_env_file,
    ensure_environment_setup,
    get_paths,
    get_config,
    get_data_dir,
    get_output_dir,
    get_processed_dir,
    get_embeddings_dir,
    get_chat_sessions_dir,
)

__all__ = [
    # Configuration classes
    "PathConfig",
    "AppConfig",
    
    # Global instances
    "paths",
    "config",
    
    # Utility functions
    "load_environment",
    "validate_environment",
    "print_full_configuration",
    "get_config_summary",
    "create_sample_env_file",
    "ensure_environment_setup",
    
    # Convenience functions
    "get_paths",
    "get_config",
    "get_data_dir",
    "get_output_dir",
    "get_processed_dir",
    "get_embeddings_dir",
    "get_chat_sessions_dir",
]
