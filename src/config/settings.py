"""
Application settings configuration for the RAG chatbot system.
All settings are configured via environment variables with sensible defaults.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AppConfig:
    """Centralized application configuration"""

    # Flask Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # File Upload Configuration
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "104857600"))  # 100MB

    # Chatbot Configuration
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.5"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "1000"))

    # Google AI Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    GOOGLE_GENAI_USE_VERTEXAI = (
        os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    )

    # Generation Model Configuration
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # AI Generation Parameters
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "3000"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    TOP_K = int(os.getenv("TOP_K", "20"))

    # Embedding Configuration
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    EMBEDDING_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")

    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Metadata Ranking Configuration
    ENABLE_METADATA_RANKING = (
        os.getenv("ENABLE_METADATA_RANKING", "true").lower() == "true"
    )
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    METADATA_WEIGHT = float(os.getenv("METADATA_WEIGHT", "0.3"))

    # Section-Aware Configuration
    ENABLE_SECTION_AWARE = os.getenv("ENABLE_SECTION_AWARE", "true").lower() == "true"
    SECTION_PRIORITY_BOOST = float(os.getenv("SECTION_PRIORITY_BOOST", "1.5"))

    # Font and Layout Configuration
    LARGE_FONT_THRESHOLD = float(os.getenv("LARGE_FONT_THRESHOLD", "14.0"))
    TOP_POSITION_THRESHOLD = float(os.getenv("TOP_POSITION_THRESHOLD", "200"))

    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "20"))
    MAX_CHAR_BUFFER = int(os.getenv("MAX_CHAR_BUFFER", "1000"))

    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    @classmethod
    def validate_configuration(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = []

        # Check if either Google API key or Vertex AI config is present
        if not cls.GOOGLE_API_KEY and not cls.GOOGLE_CLOUD_PROJECT:
            required_vars.append(
                "Either GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT must be set"
            )

        # Check if Vertex AI location is set when using Vertex AI
        if cls.GOOGLE_GENAI_USE_VERTEXAI and not cls.GOOGLE_CLOUD_LOCATION:
            required_vars.append(
                "GOOGLE_CLOUD_LOCATION must be set when using Vertex AI"
            )

        if required_vars:
            print("❌ Configuration validation failed:")
            for var in required_vars:
                print(f"   - {var}")
            return False

        return True

    @classmethod
    def print_configuration(cls) -> None:
        """Print current configuration"""
        print("⚙️  APPLICATION CONFIGURATION:")
        print("=" * 50)
        print(f"Flask Environment: {cls.FLASK_ENV}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Server: {cls.HOST}:{cls.PORT}")
        print(f"Max Upload Size: {cls.MAX_CONTENT_LENGTH / 1024 / 1024:.1f} MB")
        print(f"Default Top-K: {cls.DEFAULT_TOP_K}")
        print(f"Default Min Score: {cls.DEFAULT_MIN_SCORE}")
        print(f"Max Answer Length: {cls.MAX_ANSWER_LENGTH}")
        print(f"Use Vertex AI: {cls.GOOGLE_GENAI_USE_VERTEXAI}")
        print(f"Generation Model: {cls.GENERATION_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Enable Metadata Ranking: {cls.ENABLE_METADATA_RANKING}")
        print(f"Enable Section Aware: {cls.ENABLE_SECTION_AWARE}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 50)

    @classmethod
    def get_google_client_config(cls) -> dict:
        """Get configuration for Google AI client"""
        config = {}

        if cls.GOOGLE_GENAI_USE_VERTEXAI:
            config["vertexai"] = True
            config["project"] = cls.GOOGLE_CLOUD_PROJECT
            config["location"] = cls.GOOGLE_CLOUD_LOCATION
        else:
            config["api_key"] = cls.GOOGLE_API_KEY

        return config

    @classmethod
    def get_generation_config(cls) -> dict:
        """Get configuration for text generation"""
        return {
            "temperature": cls.TEMPERATURE,
            "max_output_tokens": cls.MAX_OUTPUT_TOKENS,
            "top_p": cls.TOP_P,
            "top_k": cls.TOP_K,
        }

    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get configuration for embeddings"""
        return {
            "task_type": cls.EMBEDDING_TASK_TYPE,
            "output_dimensionality": cls.EMBEDDING_DIMENSIONS,
        }


# Global instance for easy access
config = AppConfig()
