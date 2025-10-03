"""
RAG Chatbot Flask Application
Main application factory and configuration
"""

import logging
import os
from pathlib import Path

from flask import Flask

from app.api.routes import register_blueprints
from app.core.config import Config
from app.core.extensions import init_extensions


logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    """
    Application factory pattern for Flask app creation
    """
    # Get the absolute path to the static folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(os.path.dirname(current_dir), "static")

    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")

    # Load configuration
    app.config.from_object(config_class)

    # Initialize extensions (CORS, etc.)
    init_extensions(app)

    # Register blueprints
    register_blueprints(app)

    # Register web interface routes
    register_web_routes(app)

    # Create necessary directories
    create_directories()

    # Create database tables (deferred - will be created on first database access)
    # create_tables(app)

    # Note: Pattern management now triggered by document processing
    # init_pattern_auto_management(app)  # Disabled - patterns refresh on document upload

    return app


def register_web_routes(app):
    """
    Register web interface routes
    """

    @app.route("/")
    def index():
        """Serve the main web interface"""
        try:
            return app.send_static_file("index.html")
        except Exception as e:
            app.logger.error("❌ Error serving index.html: %s", e)
            return f"Error: {e}", 500

    @app.route("/web")
    def web_interface():
        """Alternative route for web interface"""
        try:
            return app.send_static_file("index.html")
        except Exception as e:
            app.logger.error("❌ Error serving web interface: %s", e)
            return f"Error: {e}", 500


def create_directories():
    """Create necessary directories for the application"""
    directories = [
        Path("data/processed"),
        Path("data/uploads"),
        Path("logs"),
        Path("temp"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
