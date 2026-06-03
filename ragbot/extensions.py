"""
Flask extensions initialization
"""

import logging

from flask_cors import CORS

from ragbot.models.base import db
from ragbot.chat.orchestrator import ChatbotService

# Global chatbot service instance
chatbot_service_instance = None
logger = logging.getLogger(__name__)


def init_extensions(app):
    """
    Initialize Flask extensions
    """
    global chatbot_service_instance

    # Initialize SQLAlchemy
    db.init_app(app)

    # Configure CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": app.config["CORS_ORIGINS"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "max_age": 3600,
            }
        },
    )

    # Defer chatbot service initialization until first use
    # chatbot_service_instance will be initialized lazily when first accessed
    logger.info("ℹ️ Flask extensions initialized successfully!")


def get_chatbot_service() -> ChatbotService:
    """
    Get the global chatbot service instance (lazy initialization)

    Returns:
        ChatbotService: The singleton chatbot service instance
    """
    global chatbot_service_instance
    if chatbot_service_instance is None:
        logger.info("ℹ️ Initializing chatbot service...")
        chatbot_service_instance = ChatbotService()
        logger.info("ℹ️ Chatbot service initialized successfully!")
    return chatbot_service_instance


def create_tables(app):
    """
    Create database tables (called lazily on first database access)
    """
    with app.app_context():
        db.create_all()
        logger.info("ℹ️ Database tables created successfully!")
