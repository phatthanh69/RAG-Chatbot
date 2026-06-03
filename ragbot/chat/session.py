"""Chat-session lifecycle and history persistence.

`SessionManager` owns session creation and the session-metadata/history writes
that previously lived on `ChatbotService`. Bodies are moved verbatim; the few
`self.<x>` collaborators become injected dependencies:

  - config_provider() -> dict   (was self.config, copied into new sessions)
  - logger                       (was self.logger)
  - log_process_step / log_error_concise callbacks (concise logging helpers)

`DatabaseService` is used directly (its methods are static), exactly as before.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ragbot.db.database_service import DatabaseService


class SessionManager:
    """Chat-session lifecycle and history persistence."""

    def __init__(
        self,
        config_provider=None,
        logger=None,
        log_process_step=None,
        log_error_concise=None,
    ):
        self._config_provider = config_provider or (lambda: {})
        self.logger = logger or logging.getLogger(__name__)
        self._log_process_step = log_process_step or (lambda *a, **k: None)
        self._log_error_concise = log_error_concise or (lambda *a, **k: None)

    def get_or_create_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get or create a chat session (from database)

        Args:
            session_id: Session identifier
            user_id: Optional user identifier

        Returns:
            Session data
        """
        # Get or create session in database
        session = DatabaseService.get_or_create_chat_session(session_id, user_id)

        # Check if database operation failed
        if session is None:
            self.logger.error(
                f"DatabaseService.get_or_create_chat_session returned None for session_id: {session_id}"
            )
            return {}

        # Convert to dictionary format for compatibility
        session_data = session.to_dict()
        session_data["config"] = self._config_provider().copy()
        session_data["history"] = [msg.to_dict() for msg in session.messages]

        return session_data

    def update_active_headings(
        self, session_id: str, active_headings: List[str], heading_info: Dict[str, Any]
    ) -> None:
        """
        Cập nhật active_headings trong session metadata

        Args:
            session_id: Session identifier
            active_headings: headings hiện tại đang được nhắc tới
            heading_info: Thông tin phân tích heading
        """
        try:
            DatabaseService.update_session_metadata(
                session_id,
                {
                    "active_headings": active_headings,
                    "heading_info": heading_info,
                    "last_heading_update": time.time(),
                },
            )
            # Use concise logging instead of verbose
            heading_preview = active_headings[0] if active_headings else "None"
            if len(active_headings) > 1:
                heading_preview += f" (+{len(active_headings)-1} more)"
            self._log_process_step("Updated session headings", heading_preview)
        except Exception as e:
            self._log_error_concise("Session heading update", e, session_id)

    def update_active_entity(
        self, session_id: str, active_entity: str, entity_info: Dict[str, Any]
    ) -> None:
        """
        Cập nhật active_entity trong session metadata

        Args:
            session_id: Session identifier
            active_entity: Thực thể hiện tại đang được nhắc tới
            entity_info: Thông tin chi tiết về entities
        """
        try:
            # Cập nhật active_entity trong database session
            DatabaseService.update_session_metadata(
                session_id,
                {
                    "active_entity": active_entity,
                    "entity_info": entity_info,
                    "last_entity_update": time.time(),
                },
            )

            self.logger.info(
                f"Updated active_entity for session {session_id}: {active_entity}"
            )

        except Exception as e:
            self.logger.error(f"Error updating session active_entity: {str(e)}")

    def add_to_history(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add interaction to session history in database

        Args:
            session_id: Session identifier
            question: User question
            answer: AI answer
            sources: Source documents
        """
        try:
            # Prepare metadata with sources
            metadata = {
                "sources_count": len(sources),
                "sources": sources[:5],  # Store first 5 sources in metadata
            }

            if extra_metadata and isinstance(extra_metadata, dict):
                try:
                    metadata.update(extra_metadata)
                except Exception:
                    # Be safe; metadata should not break history persistence
                    pass

            # Add user message
            DatabaseService.add_chat_message(
                session_id=session_id,
                message_type="user",
                content=question,
                metadata=metadata,
            )

            # Add bot message
            DatabaseService.add_chat_message(
                session_id=session_id,
                message_type="bot",
                content=answer,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Error adding to history: {str(e)}")
