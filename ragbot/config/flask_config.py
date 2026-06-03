"""
Application configuration management
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class"""

    # Flask configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # CORS configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

    # File upload configuration
    UPLOAD_FOLDER = Path("data/uploads")
    MAX_CONTENT_LENGTH = int(
        os.getenv("MAX_CONTENT_LENGTH", 100 * 1024 * 1024)
    )  # 100MB
    ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "md", "markdown"}

    # Processing configuration
    PROCESSED_FOLDER = Path("data/processed")
    TEMP_FOLDER = Path("temp")

    # Chatbot configuration
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.5"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "1000"))

    # Ensemble Search configuration
    USE_ENSEMBLE_RETRIEVER = (
        os.getenv("USE_ENSEMBLE_RETRIEVER", "true").lower() == "true"
    )
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))
    FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")  # "rrf" or "weighted"
    RRF_K = int(os.getenv("RRF_K", "60"))

    # AI Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    GOOGLE_GENAI_USE_VERTEXAI = (
        os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    )

    # Embedding configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    EMBEDDING_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")

    # Chunking configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/rag_chatbot"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_TIMEZONE = "Asia/Ho_Chi_Minh"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 10,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
        "connect_args": {"options": "-c timezone=Asia/Ho_Chi_Minh"},
    }

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = Path("logs/app.log")


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER = Path("tests/data/uploads")
    PROCESSED_FOLDER = Path("tests/data/processed")


# Configuration mapping
_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "development")
    return _CONFIG_MAP.get(env, _CONFIG_MAP["default"])
