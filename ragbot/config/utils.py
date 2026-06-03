"""
Configuration utilities for the RAG chatbot system.
Provides helper functions for configuration management and validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .paths import PathConfig
from .settings import AppConfig


def load_environment(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from .env file

    Args:
        env_file: Path to .env file (defaults to project root)
    """
    if env_file is None:
        # Look for .env in project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        logging.info(f"Loaded environment from: {env_file}")
    else:
        logging.warning(f"Environment file not found: {env_file}")
        logging.info("Using system environment variables")


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set

    Returns:
        bool: True if validation passes, False otherwise
    """
    logging.info("Validating environment configuration...")

    # Validate paths
    paths_valid = PathConfig.validate_configuration()

    # Validate app settings
    app_valid = AppConfig.validate_configuration()

    if paths_valid and app_valid:
        logging.info("Environment validation passed")
        return True
    else:
        logging.error("Environment validation failed")
        return False


def print_full_configuration() -> None:
    """Print complete configuration including paths and settings"""
    print("\n" + "=" * 80)
    print("🚀 RAG CHATBOT - COMPLETE CONFIGURATION")
    print("=" * 80)

    # Print paths
    PathConfig.print_configuration()
    print()

    # Print settings
    AppConfig.print_configuration()
    print()

    # Print environment summary
    print("🌍 ENVIRONMENT SUMMARY:")
    print("-" * 50)
    print(f"Project Root: {PathConfig.BASE_DIR}")
    print(f"Python Path: {PathConfig.SRC_DIR}")
    print(f"Data Location: {PathConfig.DATA_DIR}")
    print(f"Output Location: {PathConfig.OUTPUT_DIR}")
    print(
        f"Google AI: {'Vertex AI' if AppConfig.GOOGLE_GENAI_USE_VERTEXAI else 'API Key'}"
    )
    print(f"Debug Mode: {AppConfig.DEBUG}")
    print("=" * 80)


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration

    Returns:
        Dict containing configuration summary
    """
    return {
        "paths": {
            "base_dir": str(PathConfig.BASE_DIR),
            "data_dir": str(PathConfig.DATA_DIR),
            "output_dir": str(PathConfig.OUTPUT_DIR),
            "processed_dir": str(PathConfig.PROCESSED_DATA_DIR),
            "embeddings_dir": str(PathConfig.EMBEDDINGS_DIR),
            "chat_sessions_dir": str(PathConfig.CHAT_SESSIONS_DIR),
        },
        "settings": {
            "flask_env": AppConfig.FLASK_ENV,
            "debug": AppConfig.DEBUG,
            "server": f"{AppConfig.HOST}:{AppConfig.PORT}",
            "google_ai": (
                "Vertex AI" if AppConfig.GOOGLE_GENAI_USE_VERTEXAI else "API Key"
            ),
            "generation_model": AppConfig.GENERATION_MODEL,
            "embedding_model": AppConfig.EMBEDDING_MODEL,
            "chunk_size": AppConfig.CHUNK_SIZE,
            "chunk_overlap": AppConfig.CHUNK_OVERLAP,
        },
        "validation": {
            "paths_valid": PathConfig.validate_configuration(),
            "app_valid": AppConfig.validate_configuration(),
        },
    }


def create_sample_env_file(output_path: Optional[str] = None) -> str:
    """
    Create a sample .env file with all available configuration options

    Args:
        output_path: Path to save the sample .env file

    Returns:
        str: Path to the created file
    """
    if output_path is None:
        output_path = PathConfig.BASE_DIR / ".env.sample"

    sample_content = """# RAG Chatbot Configuration File
# Copy this file to .env and modify the values as needed

# Flask Configuration
FLASK_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here

# Server Configuration
HOST=0.0.0.0
PORT=5000

# CORS Configuration
CORS_ORIGINS=*

# File Upload Configuration
MAX_CONTENT_LENGTH=104857600

# Chatbot Configuration
DEFAULT_TOP_K=10
DEFAULT_MIN_SCORE=0.5
MAX_ANSWER_LENGTH=1000

# Google AI Configuration
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=false

# Generation Model Configuration
GENERATION_MODEL=gemini-2.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001

# AI Generation Parameters
TEMPERATURE=0.3
MAX_OUTPUT_TOKENS=3000
TOP_P=0.9
TOP_K=20

# Embedding Configuration
EMBEDDING_DIMENSIONS=1536
EMBEDDING_TASK_TYPE=RETRIEVAL_DOCUMENT

# Processing Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=120
BATCH_SIZE=10

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Metadata Ranking Configuration
ENABLE_METADATA_RANKING=true
SEMANTIC_WEIGHT=0.7
METADATA_WEIGHT=0.3

# Section-Aware Configuration
ENABLE_SECTION_AWARE=true
SECTION_PRIORITY_BOOST=1.5

# Font and Layout Configuration
LARGE_FONT_THRESHOLD=14.0
TOP_POSITION_THRESHOLD=200

# Performance Configuration
MAX_WORKERS=20
MAX_CHAR_BUFFER=1000

# Retry Configuration
MAX_RETRIES=3

# ==============================================================================
# DIRECTORY CONFIGURATION
# ==============================================================================

# Base Data Directory (relative to project root)
DATA_DIR=data

# Output Directory (relative to project root)
OUTPUT_DIR=output

# Chat Sessions Directory (relative to project root)
CHAT_SESSIONS_DIR=chat_sessions

# Token Logs Directory (relative to project root)
TOKEN_LOGS_DIR=token_logs

# Temporary Directory (relative to project root)
TEMP_DIR=temp

# Cache Directory (relative to project root)
CACHE_DIR=cache

# Logs Directory (relative to project root)
LOGS_DIR=logs

# Default Input Path (relative to project root)
DEFAULT_INPUT_PATH=data/raw_data
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    logging.info(f"Created sample .env file: {output_path}")
    return str(output_path)


def ensure_environment_setup() -> bool:
    """
    Ensure the environment is properly set up

    Returns:
        bool: True if setup is complete, False otherwise
    """
    logging.info("Setting up environment...")

    # Load environment
    load_environment()

    # Ensure directories exist
    PathConfig.ensure_directories()

    # Validate configuration
    is_valid = validate_environment()

    if is_valid:
        logging.info("Environment setup complete")
        return True
    else:
        logging.error("Environment setup failed")
        logging.error("Please check your .env file and required environment variables")
        return False


# Convenience functions for quick access
def get_paths() -> PathConfig:
    """Get paths configuration instance"""
    return PathConfig


def get_config() -> AppConfig:
    """Get application configuration instance"""
    return AppConfig


def get_data_dir() -> Path:
    """Get data directory path"""
    return PathConfig.DATA_DIR


def get_output_dir() -> Path:
    """Get output directory path"""
    return PathConfig.OUTPUT_DIR


def get_processed_dir() -> Path:
    """Get processed data directory path"""
    return PathConfig.PROCESSED_DATA_DIR


def get_embeddings_dir() -> Path:
    """Get embeddings directory path"""
    return PathConfig.EMBEDDINGS_DIR


def get_chat_sessions_dir() -> Path:
    """Get chat sessions directory path"""
    return PathConfig.CHAT_SESSIONS_DIR


if __name__ == "__main__":
    # Demo configuration
    print("🚀 RAG Chatbot Configuration Demo")
    print("=" * 50)

    # Setup environment
    ensure_environment_setup()

    # Print configuration
    print_full_configuration()

    # Create sample .env if it doesn't exist
    env_file = PathConfig.BASE_DIR / ".env"
    if not env_file.exists():
        print("\n📝 Creating sample .env file...")
        create_sample_env_file(env_file)
        print("Please edit .env file with your actual configuration values")
