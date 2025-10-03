"""
Chatbot API endpoints
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session

from app.core.extensions import get_chatbot_service
from app.services.chatbot_service import ChatbotService
from app.services.database_service import DatabaseService
from app.services.vector_search_service import VectorSearchService
from app.utils.response_helpers import error_response, success_response

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint for asking questions"""
    try:
        data = request.get_json()

        if not data:
            return error_response("No JSON data provided", 400)

        question = data.get("question", "").strip()
        # Create session ID based on client IP address + timestamp for unique conversations
        client_ip = request.remote_addr or "unknown"
        # If frontend sends 'default' or no session_id, generate new unique session per conversation
        provided_session_id = data.get("session_id")
        if (
            not provided_session_id
            or provided_session_id == "default"
            or provided_session_id is None
        ):
            # Create short unique session ID using IP + short UUID for internal testing
            short_uuid = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
            session_id = f"{client_ip.replace('.', '_')}_{short_uuid}"
        else:
            session_id = provided_session_id
        current_app.logger.info(
            f"REQUEST FROM IP: {client_ip}, PROVIDED SESSION: {provided_session_id}, FINAL SESSION: {session_id}"
        )
        embedded_file = data.get("embedded_file")

        if not question:
            return error_response("Question is required", 400)

        # Get the singleton chatbot service instance
        chatbot_service = get_chatbot_service()

        # Process the question using the new database-backed service
        # Optional heading filter to guide retrieval
        active_heading_title = data.get("active_heading_title")
        # Convert to list for new API (backward compatibility)
        active_headings = [active_heading_title] if active_heading_title else None

        result = chatbot_service.ask_question(
            question=question,
            session_id=session_id,
            user_id=None,  # Could be extracted from authentication
            use_vector_search=True,
            active_headings=active_headings,
        )

        if "error" in result:
            return error_response(result["error"], 500)

        return success_response(
            {
                "session_id": session_id,
                "question": question,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "processing_time": result.get("processing_time", 0),
                "timestamp": datetime.utcnow().isoformat(),
                # Add debug information
                "original_question": result.get("original_question"),
                "rewritten_question": result.get("rewritten_question"),
                "heading_info": result.get("heading_info", {}),
                "question_type": result.get("question_type"),
                "classification": result.get("classification", {}),
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {str(e)}")
        return error_response(f"Chat failed: {str(e)}", 500)


@chatbot_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all chat sessions from database"""
    try:
        sessions = DatabaseService.get_all_chat_sessions()
        sessions_data = [session.to_dict() for session in sessions]
        return success_response({"sessions": sessions_data})

    except Exception as e:
        return error_response(f"Failed to list sessions: {str(e)}", 500)


@chatbot_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get a specific chat session from database"""
    try:
        chat_session = DatabaseService.get_chat_session(session_id)

        if not chat_session:
            return error_response("Session not found", 404)

        session_data = chat_session.to_dict()
        session_data["messages"] = [msg.to_dict() for msg in chat_session.messages]

        return success_response(
            {
                "session_id": session_id,
                "session_data": session_data,
                "total_interactions": len(chat_session.messages),
            }
        )

    except Exception as e:
        return error_response(f"Failed to get session: {str(e)}", 500)


@chatbot_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a chat session from database"""
    try:
        success = DatabaseService.delete_chat_session(session_id)

        if not success:
            return error_response("Session not found", 404)

        return success_response(
            {"message": f"Session {session_id} deleted successfully"}
        )

    except Exception as e:
        return error_response(f"Failed to delete session: {str(e)}", 500)


@chatbot_bp.route("/search", methods=["POST"])
def search_documents():
    """Search documents without full chatbot response"""
    try:
        data = request.get_json()

        if not data:
            return error_response("No JSON data provided", 400)

        query = data.get("query", "").strip()
        embedded_file = data.get(
            "embedded_file"
        )  # Optional - if not provided, search all files
        top_k = data.get("top_k", 10)
        min_score = data.get("min_score", 0.5)
        active_heading_title = data.get("active_heading_title")
        # Convert to list for new API (backward compatibility)
        active_headings = [active_heading_title] if active_heading_title else None

        if not query:
            return error_response("Search query is required", 400)

        # Get the singleton chatbot service instance
        chatbot_service = get_chatbot_service()

        # Generate embedding for the search query using the same method as chatbot service
        try:
            # Use the chatbot service's embedding generation method
            query_embedding = chatbot_service._get_question_embedding(query)
            if not query_embedding:
                current_app.logger.warning(
                    "Could not generate embedding for search query"
                )
                return error_response("Failed to generate query embedding", 500)
        except Exception as e:
            current_app.logger.error(f"Error generating query embedding: {str(e)}")
            return error_response(f"Embedding generation failed: {str(e)}", 500)

        # Perform vector search
        results = VectorSearchService.search_similar_chunks(
            embedding=query_embedding,
            limit=top_k,
            min_score=min_score,
            active_headings=active_headings,
        )

        # Format results for frontend compatibility
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                {
                    "rank": i,
                    "score": result["score"],
                    "content": result["content"],
                    "pdf_name": result["file_name"],
                    "page": result["page"] or "?",
                    "clean_pdf_name": result["clean_pdf_name"],
                    "meta": {
                        "page": result.get("page"),
                        "block_index": result.get("block_index"),
                        "bbox": result.get("bbox"),
                        "file_name": result.get("file_name"),
                        # Include heading metadata for UI filtering/highlighting
                        "heading_id": result.get("heading_id"),
                        "heading_title": result.get("heading_title"),
                        "heading_parent_id": result.get("heading_parent_id"),
                        "heading_level": result.get("heading_level"),
                    },
                }
            )

        return success_response(
            {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error in search endpoint: {str(e)}")
        return error_response(f"Search failed: {str(e)}", 500)


@chatbot_bp.route("/config", methods=["GET"])
def get_config():
    """Get current chatbot configuration"""
    try:
        # Get the singleton chatbot service instance
        chatbot_service = get_chatbot_service()

        config = chatbot_service.get_config()
        return success_response({"config": config})

    except Exception as e:
        return error_response(f"Failed to get config: {str(e)}", 500)


@chatbot_bp.route("/config", methods=["PUT"])
def update_config():
    """Update chatbot configuration"""
    try:
        data = request.get_json()

        if not data:
            return error_response("No JSON data provided", 400)

        new_config = data.get("config", {})

        # Get the singleton chatbot service instance
        chatbot_service = get_chatbot_service()

        success = chatbot_service.update_config(new_config)

        if not success:
            return error_response("Failed to update configuration", 500)

        return success_response(
            {
                "message": "Configuration updated successfully",
                "config": chatbot_service.get_config(),
            }
        )

    except Exception as e:
        return error_response(f"Failed to update config: {str(e)}", 500)


@chatbot_bp.route("/stats", methods=["GET"])
def get_statistics():
    """Get chatbot usage statistics"""
    try:
        # Get the singleton chatbot service instance
        chatbot_service = get_chatbot_service()

        stats = chatbot_service.get_statistics()
        return success_response({"statistics": stats})

    except Exception as e:
        return error_response(f"Failed to get statistics: {str(e)}", 500)


@chatbot_bp.route("/ensemble/status", methods=["GET"])
def get_ensemble_status():
    """Get ensemble retriever status and configuration"""
    try:
        chatbot_service = get_chatbot_service()
        status = chatbot_service.get_ensemble_status()
        return success_response({"ensemble_status": status})

    except Exception as e:
        return error_response(f"Failed to get ensemble status: {str(e)}", 500)


@chatbot_bp.route("/ensemble/weights", methods=["POST"])
def update_ensemble_weights():
    """Update ensemble retriever weights"""
    try:
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided", 400)

        bm25_weight = data.get("bm25_weight")
        vector_weight = data.get("vector_weight")

        if bm25_weight is None or vector_weight is None:
            return error_response(
                "Both bm25_weight and vector_weight are required", 400
            )

        if not isinstance(bm25_weight, (int, float)) or not isinstance(
            vector_weight, (int, float)
        ):
            return error_response("Weights must be numeric values", 400)

        if bm25_weight < 0 or vector_weight < 0:
            return error_response("Weights must be non-negative", 400)

        if bm25_weight + vector_weight == 0:
            return error_response("At least one weight must be positive", 400)

        chatbot_service = get_chatbot_service()
        success = chatbot_service.update_ensemble_weights(bm25_weight, vector_weight)

        if success:
            return success_response(
                {
                    "message": "Ensemble weights updated successfully",
                    "status": chatbot_service.get_ensemble_status(),
                }
            )
        else:
            return error_response("Failed to update ensemble weights", 500)

    except Exception as e:
        return error_response(f"Failed to update ensemble weights: {str(e)}", 500)


@chatbot_bp.route("/ensemble/toggle", methods=["POST"])
def toggle_ensemble_retriever():
    """Enable or disable ensemble retriever"""
    try:
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided", 400)

        enabled = data.get("enabled")
        if enabled is None:
            return error_response("'enabled' field is required", 400)

        if not isinstance(enabled, bool):
            return error_response("'enabled' must be a boolean value", 400)

        chatbot_service = get_chatbot_service()
        success = chatbot_service.toggle_ensemble_retriever(enabled)

        if success:
            return success_response(
                {
                    "message": f"Ensemble retriever {'enabled' if enabled else 'disabled'} successfully",
                    "status": chatbot_service.get_ensemble_status(),
                }
            )
        else:
            return error_response("Failed to toggle ensemble retriever", 500)

    except Exception as e:
        return error_response(f"Failed to toggle ensemble retriever: {str(e)}", 500)


@chatbot_bp.route("/ensemble/config", methods=["GET", "POST"])
def ensemble_config():
    """Get or update ensemble search configuration"""
    try:
        chatbot_service = get_chatbot_service()

        if request.method == "GET":
            # Return current ensemble config
            config = {
                "use_ensemble_retriever": getattr(
                    chatbot_service.config, "USE_ENSEMBLE_RETRIEVER", True
                ),
                "bm25_weight": getattr(chatbot_service.config, "BM25_WEIGHT", 0.3),
                "vector_weight": getattr(chatbot_service.config, "VECTOR_WEIGHT", 0.7),
                "fusion_method": getattr(
                    chatbot_service.config, "FUSION_METHOD", "rrf"
                ),
                "rrf_k": getattr(chatbot_service.config, "RRF_K", 60),
            }
            return success_response(config)

        elif request.method == "POST":
            # Update ensemble config
            data = request.get_json()

            if not data:
                return error_response("No JSON data provided", 400)

            # Update config values
            if "use_ensemble_retriever" in data:
                setattr(
                    chatbot_service.config,
                    "USE_ENSEMBLE_RETRIEVER",
                    bool(data["use_ensemble_retriever"]),
                )

            if "bm25_weight" in data:
                setattr(
                    chatbot_service.config, "BM25_WEIGHT", float(data["bm25_weight"])
                )

            if "vector_weight" in data:
                setattr(
                    chatbot_service.config,
                    "VECTOR_WEIGHT",
                    float(data["vector_weight"]),
                )

            if "fusion_method" in data:
                setattr(
                    chatbot_service.config, "FUSION_METHOD", str(data["fusion_method"])
                )

            if "rrf_k" in data:
                setattr(chatbot_service.config, "RRF_K", int(data["rrf_k"]))

            # Reinitialize ensemble retriever with new config if enabled
            if getattr(chatbot_service.config, "USE_ENSEMBLE_RETRIEVER", True):
                chatbot_service.ensemble_retriever = None
                chatbot_service._ensemble_init_attempted = False
                chatbot_service._initialize_ensemble_retriever_if_needed()

            return success_response(
                {
                    "message": "Ensemble config updated successfully",
                    "config": {
                        "use_ensemble_retriever": getattr(
                            chatbot_service.config, "USE_ENSEMBLE_RETRIEVER", True
                        ),
                        "bm25_weight": getattr(
                            chatbot_service.config, "BM25_WEIGHT", 0.3
                        ),
                        "vector_weight": getattr(
                            chatbot_service.config, "VECTOR_WEIGHT", 0.7
                        ),
                        "fusion_method": getattr(
                            chatbot_service.config, "FUSION_METHOD", "rrf"
                        ),
                        "rrf_k": getattr(chatbot_service.config, "RRF_K", 60),
                    },
                }
            )

        # Fallback (should not be reached due to methods restriction)
        return error_response("Method not allowed", 405)

    except Exception as e:
        return error_response(f"Failed to manage ensemble config: {str(e)}", 500)


@chatbot_bp.route("/ensemble/refresh", methods=["POST"])
def refresh_ensemble_indices():
    """Refresh ensemble retriever indices"""
    try:
        chatbot_service = get_chatbot_service()
        success = chatbot_service.refresh_ensemble_indices()

        if success:
            return success_response(
                {
                    "message": "Ensemble indices refreshed successfully",
                    "status": chatbot_service.get_ensemble_status(),
                }
            )
        else:
            return error_response("Failed to refresh ensemble indices", 500)

    except Exception as e:
        return error_response(f"Failed to refresh ensemble indices: {str(e)}", 500)
