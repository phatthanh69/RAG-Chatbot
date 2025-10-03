#!/usr/bin/env python3
"""
RAG Chatbot Flask Application Entry Point
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app import create_app
from app.core.config import get_config


# Configure logging
def setup_logging():
    """Setup application logging with reduced third-party noise"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = Path("logs/app.log")

    # Create logs directory
    log_file.parent.mkdir(exist_ok=True)

    # Configure main logging with UTF-8 encoding to handle Unicode characters
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Reduce noise from third-party libraries
    # Set httpx to WARNING to hide HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set Google AI libraries to WARNING to hide verbose debug info
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    # Set other noisy libraries to WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(
        logging.WARNING
    )  # Flask's built-in server logs

    # Keep our application logs at the configured level
    logging.getLogger("app").setLevel(getattr(logging, log_level))

    return logging.getLogger(__name__)


def main():
    """Main application entry point"""
    start_time = time.time()
    logger = setup_logging()
    logger.info("Starting RAG Chatbot Flask Application...")

    # Get configuration
    config_start = time.time()
    config = get_config()
    config_time = time.time() - config_start
    logger.info(f"Running in {os.getenv('FLASK_ENV', 'development')} mode")
    logger.info(f"Configuration loaded in {config_time:.2f}s")

    # Check for embedded files at startup (optimized)
    try:
        from pathlib import Path

        processed_dir = Path("data/processed")
        if processed_dir.exists():
            # Use os.scandir for faster directory scanning

            embedded_files = []
            try:
                with os.scandir(processed_dir) as entries:
                    for entry in entries:
                        if entry.name.endswith("_embedded.jsonl") and entry.is_file():
                            embedded_files.append(entry.name)
            except OSError:
                # Fallback to glob if scandir fails
                embedded_files = [
                    f.name for f in processed_dir.glob("*_embedded.jsonl")
                ]

            if embedded_files:
                logger.info(
                    f"Found {len(embedded_files)} embedded files for multi-document search"
                )
                # Only log file details if there are not too many files
                if len(embedded_files) <= 10:
                    for i, filename in enumerate(embedded_files, 1):
                        file_path = processed_dir / filename
                        try:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            logger.info(f"  {i}. {filename} ({size_mb:.1f} MB)")
                        except OSError:
                            logger.info(f"  {i}. {filename}")
                else:
                    logger.info("Chatbot will search across ALL embedded documents")
            else:
                logger.warning("No embedded files found in data/processed/")
                logger.info("Upload and process documents through the web interface")
        else:
            logger.warning("data/processed directory not found")
            logger.info("Directory will be created when first document is processed")
    except Exception as e:
        logger.error(f"Error checking embedded files: {e}")

    # Create Flask app
    app_start = time.time()
    app = create_app(config)
    app_time = time.time() - app_start
    logger.info(f"Flask app created in {app_time:.2f}s")

    # Get port and host from environment
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"

    total_time = time.time() - start_time
    logger.info(f"Application startup completed in {total_time:.2f}s")
    logger.info(f"Web Interface: http://{host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/api")
    logger.info("Multi-document RAG chatbot ready!")

    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
