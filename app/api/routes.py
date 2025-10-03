"""
API routes registration
"""

from flask import Blueprint
from app.api.document_routes import document_bp
from app.api.chatbot_routes import chatbot_bp
from app.api.health_routes import health_bp


def register_blueprints(app):
    """
    Register all API blueprints
    """
    # Register blueprints with URL prefixes
    app.register_blueprint(health_bp, url_prefix='/api/health')
    app.register_blueprint(document_bp, url_prefix='/api/documents')
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')

    # Root API endpoint
    @app.route('/api')
    def api_root():
        return {
            'message': 'RAG Chatbot API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/health',
                'documents': '/api/documents',
                'chatbot': '/api/chatbot'
            }
        }

    # Note: Web interface routes are handled in app/__init__.py
