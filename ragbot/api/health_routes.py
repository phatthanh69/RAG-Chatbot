"""
Health check API endpoints
"""

import os
from datetime import datetime

import psutil
from flask import Blueprint, jsonify

from ragbot.utils.response_helpers import get_vietnam_time

health_bp = Blueprint("health", __name__)


@health_bp.route("", methods=["GET"])
def health_check():
    """Basic health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": get_vietnam_time().isoformat(),
            "service": "RAG Chatbot API",
        }
    )


@health_bp.route("/detailed", methods=["GET"])
def detailed_health():
    """Detailed health check with system information"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": get_vietnam_time().isoformat(),
            "service": "RAG Chatbot API",
            "version": "1.0.0",
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_free": psutil.disk_usage("/").free,
                "uptime": os.popen("uptime").read() if os.name != "nt" else "N/A",
            },
        }
    )
