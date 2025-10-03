"""
Path configuration for the RAG chatbot system.
All input/output directories are configured via environment variables.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PathConfig:
    """Centralized path configuration for all input/output directories"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    
    # Data directories
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    UPLOADS_DIR = DATA_DIR / "uploads"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    
    # Output directories
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))
    RESULTS_DIR = OUTPUT_DIR / "results"
    CHAT_SESSIONS_DIR = Path(os.getenv("CHAT_SESSIONS_DIR", str(BASE_DIR / "chat_sessions")))
    TOKEN_LOGS_DIR = Path(os.getenv("TOKEN_LOGS_DIR", str(BASE_DIR / "token_logs")))
    
    # Temporary and cache directories
    TEMP_DIR = Path(os.getenv("TEMP_DIR", str(BASE_DIR / "temp")))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", str(BASE_DIR / "cache")))
    
    # Log directories
    LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
    
    # Default input paths (can be overridden)
    DEFAULT_INPUT_PATH = Path(os.getenv("DEFAULT_INPUT_PATH", str(BASE_DIR / "data" / "raw_data")))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.UPLOADS_DIR,
            cls.EMBEDDINGS_DIR,
            cls.OUTPUT_DIR,
            cls.RESULTS_DIR,
            cls.CHAT_SESSIONS_DIR,
            cls.TOKEN_LOGS_DIR,
            cls.TEMP_DIR,
            cls.CACHE_DIR,
            cls.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_processed_file_path(cls, filename: str) -> Path:
        """Get path for a processed file in the processed data directory"""
        return cls.PROCESSED_DATA_DIR / filename
    
    @classmethod
    def get_embedding_file_path(cls, filename: str) -> Path:
        """Get path for an embedding file in the embeddings directory"""
        return cls.EMBEDDINGS_DIR / filename
    
    @classmethod
    def get_output_file_path(cls, filename: str) -> Path:
        """Get path for an output file in the results directory"""
        return cls.RESULTS_DIR / filename
    
    @classmethod
    def get_chat_session_path(cls, session_id: str) -> Path:
        """Get path for a chat session file"""
        return cls.CHAT_SESSIONS_DIR / f"chat_session_{session_id}.json"
    
    @classmethod
    def get_token_log_path(cls, filename: str) -> Path:
        """Get path for a token log file"""
        return cls.TOKEN_LOGS_DIR / filename
    
    @classmethod
    def get_temp_file_path(cls, filename: str) -> Path:
        """Get path for a temporary file"""
        return cls.TEMP_DIR / filename
    
    @classmethod
    def get_cache_file_path(cls, filename: str) -> Path:
        """Get path for a cache file"""
        return cls.CACHE_DIR / filename
    
    @classmethod
    def get_log_file_path(cls, filename: str) -> Path:
        """Get path for a log file"""
        return cls.LOGS_DIR / filename
    
    @classmethod
    def resolve_input_path(cls, input_path: Optional[str] = None) -> Path:
        """Resolve input path, using default if none provided"""
        if input_path:
            return Path(input_path)
        return cls.DEFAULT_INPUT_PATH
    
    @classmethod
    def get_relative_path(cls, file_path: Path) -> str:
        """Get relative path from base directory"""
        try:
            return str(file_path.relative_to(cls.BASE_DIR))
        except ValueError:
            return str(file_path)
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """Validate that all required paths are accessible"""
        try:
            # Check if base directories exist or can be created
            cls.ensure_directories()
            return True
        except Exception as e:
            print(f"❌ Path validation failed: {e}")
            return False
    
    @classmethod
    def print_configuration(cls) -> None:
        """Print current path configuration"""
        print("📁 PATH CONFIGURATION:")
        print("=" * 50)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Source Directory: {cls.SRC_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Raw Data: {cls.RAW_DATA_DIR}")
        print(f"Processed Data: {cls.PROCESSED_DATA_DIR}")
        print(f"Uploads: {cls.UPLOADS_DIR}")
        print(f"Embeddings: {cls.EMBEDDINGS_DIR}")
        print(f"Output: {cls.OUTPUT_DIR}")
        print(f"Results: {cls.RESULTS_DIR}")
        print(f"Chat Sessions: {cls.CHAT_SESSIONS_DIR}")
        print(f"Token Logs: {cls.TOKEN_LOGS_DIR}")
        print(f"Temporary: {cls.TEMP_DIR}")
        print(f"Cache: {cls.CACHE_DIR}")
        print(f"Logs: {cls.LOGS_DIR}")
        print(f"Default Input: {cls.DEFAULT_INPUT_PATH}")
        print("=" * 50)


# Global instance for easy access
paths = PathConfig()
