"""
Database service for handling database operations
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from ragbot.models.base import db
from ragbot.models.chat import ChatMessage, ChatSession
from ragbot.models.document import Document, DocumentChunk
from ragbot.models.model_pattern import ModelPattern

# Global flag to track if tables have been created
_tables_created = False
logger = logging.getLogger(__name__)


def ensure_tables_created():
    """
    Ensure database tables are created (called before first database operation)
    """
    global _tables_created
    if not _tables_created:
        from flask import current_app

        if current_app:
            with current_app.app_context():
                db.create_all()
                logger.info("ℹ️ Database tables created successfully!")
            _tables_created = True


class DatabaseService:
    """Service class for database operations"""

    @staticmethod
    def create_document(
        file_name: str,
        original_file_name: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """Create a new document record"""
        ensure_tables_created()
        document = Document()
        setattr(document, "file_name", file_name)
        setattr(document, "original_file_name", original_file_name)
        if file_path is not None:
            setattr(document, "file_path", file_path)
        if file_size is not None:
            setattr(document, "file_size", file_size)
        if mime_type is not None:
            setattr(document, "mime_type", mime_type)
        setattr(document, "doc_metadata", metadata or {})
        db.session.add(document)
        db.session.commit()
        return document

    @staticmethod
    def update_document_status(
        document_id: int, status: str, processing_date: Optional[datetime] = None
    ) -> bool:
        """Update document status"""
        document = Document.query.get(document_id)
        if document:
            document.status = status
            if processing_date:
                document.processing_date = processing_date
            db.session.commit()
            return True
        return False

    @staticmethod
    def add_document_chunks(
        document_id: int, chunks_data: List[Dict[str, Any]]
    ) -> None:
        """Add document chunks with embeddings to database"""
        for chunk_data in chunks_data:
            chunk = DocumentChunk()
            setattr(chunk, "document_id", document_id)
            setattr(chunk, "chunk_index", chunk_data["chunk_index"])
            setattr(chunk, "content", chunk_data["content"])
            setattr(chunk, "tokenized_content", chunk_data.get("tokenized_content"))
            setattr(chunk, "embedding", chunk_data["embedding"])
            setattr(chunk, "page_number", chunk_data.get("page"))
            setattr(chunk, "block_index", chunk_data.get("block_index"))
            setattr(chunk, "bbox", chunk_data.get("bbox"))
            setattr(
                chunk, "font_info", chunk_data.get("font_info", {}).get("font_size", {})
            )
            # New hierarchical heading fields (optional during migration)
            setattr(chunk, "heading_id", chunk_data.get("heading_id"))
            setattr(chunk, "heading_title", chunk_data.get("heading_title"))
            setattr(chunk, "heading_parent_id", chunk_data.get("heading_parent_id"))
            setattr(chunk, "heading_level", chunk_data.get("heading_level"))
            setattr(chunk, "entity", chunk_data.get("entity"))
            setattr(chunk, "section", chunk_data.get("section"))
            setattr(chunk, "is_heading", chunk_data.get("is_heading", False))
            setattr(chunk, "clean_pdf_name", chunk_data.get("clean_pdf_name"))
            db.session.add(chunk)
        db.session.commit()

    @staticmethod
    def get_document_by_id(document_id: int) -> Optional[Document]:
        """Get document by ID"""
        return Document.query.get(document_id)

    @staticmethod
    def get_all_documents() -> List[Document]:
        """Get all documents"""
        return Document.query.order_by(Document.upload_date.desc()).all()

    @staticmethod
    def get_all_documents_with_chunk_count() -> List[Dict[str, Any]]:
        """Get all documents with their chunk count (more efficient for listing)"""
        results = (
            db.session.query(
                Document, func.count(DocumentChunk.id).label("chunk_count")
            )
            .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
            .group_by(Document.id)
            .order_by(Document.upload_date.desc())
            .all()
        )

        documents_with_counts = []
        for doc, chunk_count in results:
            doc_dict = doc.to_dict()
            doc_dict["chunks_count"] = chunk_count
            documents_with_counts.append(doc_dict)

        return documents_with_counts

    @staticmethod
    def get_document_chunk_count(document_id: int) -> int:
        """Get chunk count for a specific document"""
        return DocumentChunk.query.filter_by(document_id=document_id).count()

    @staticmethod
    def get_documents_by_status(status: str) -> List[Document]:
        """Get documents by status"""
        return (
            Document.query.filter_by(status=status)
            .order_by(Document.upload_date.desc())
            .all()
        )

    @staticmethod
    def create_chat_session(
        session_id: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession()
        setattr(session, "session_id", session_id)
        if user_id is not None:
            setattr(session, "user_id", user_id)
        # Store into the correct JSON column
        setattr(session, "session_metadata", metadata or {})
        db.session.add(session)
        db.session.commit()
        return session

    @staticmethod
    def get_chat_session(session_id: str) -> Optional[ChatSession]:
        """Get chat session by session ID"""
        return ChatSession.query.filter_by(session_id=session_id).first()

    @staticmethod
    def get_or_create_chat_session(
        session_id: str, user_id: Optional[str] = None
    ) -> ChatSession:
        """Get existing chat session or create a new one"""
        ensure_tables_created()
        session = ChatSession.query.filter_by(session_id=session_id).first()
        if not session:
            session = DatabaseService.create_chat_session(session_id, user_id)
        return session

    @staticmethod
    def add_chat_message(
        session_id: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> ChatMessage:
        """Add a message to a chat session"""
        message = ChatMessage()
        setattr(message, "session_id", session_id)
        setattr(message, "message_type", message_type)
        setattr(message, "content", content)
        # Store into the correct JSON column
        setattr(message, "message_metadata", metadata or {})
        db.session.add(message)
        db.session.commit()
        return message

    @staticmethod
    def get_chat_messages(session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat messages for a session"""
        return (
            ChatMessage.query.filter_by(session_id=session_id)
            .order_by(ChatMessage.timestamp.asc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_all_chat_sessions() -> List[ChatSession]:
        """Get all chat sessions"""
        return ChatSession.query.order_by(ChatSession.created_at.desc()).all()

    @staticmethod
    def update_session_metadata(session_id: str, metadata: Dict) -> bool:
        """Update session metadata for a chat session"""
        try:
            session = ChatSession.query.filter_by(session_id=session_id).first()
            if session:
                # Merge new metadata with existing metadata
                current = session.session_metadata or {}
                # Reassign to trigger SQLAlchemy change tracking for JSON
                session.session_metadata = {**current, **(metadata or {})}
                db.session.commit()
                return True
            return False
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error updating session metadata: %s", e)
            return False

    @staticmethod
    def delete_chat_session(session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        session = ChatSession.query.filter_by(session_id=session_id).first()
        if session:
            db.session.delete(session)
            db.session.commit()
            return True
        return False

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get system statistics"""
        total_documents = Document.query.count()
        processed_documents = Document.query.filter_by(status="completed").count()
        total_chunks = DocumentChunk.query.count()
        total_sessions = ChatSession.query.count()
        total_messages = ChatMessage.query.count()

        return {
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "total_chunks": total_chunks,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
        }

    @staticmethod
    def clear_all_data() -> bool:
        """Clear all data from database (use with caution!)"""
        try:
            ChatMessage.query.delete()
            ChatSession.query.delete()
            DocumentChunk.query.delete()
            Document.query.delete()
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error clearing data: %s", e)
            return False

    @staticmethod
    def delete_document(document_id: int) -> bool:
        """Delete a document by ID"""
        try:
            document = Document.query.get(document_id)
            if document:
                db.session.delete(document)
                db.session.commit()
                return True
            return False
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting document: %s", e)
            return False

    @staticmethod
    def delete_document_chunks(document_id: int) -> int:
        """Delete all chunks for a specific document and return count of deleted chunks"""
        try:
            chunks = DocumentChunk.query.filter_by(document_id=document_id).all()
            chunk_count = len(chunks)

            for chunk in chunks:
                db.session.delete(chunk)

            db.session.commit()
            return chunk_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting document chunks: %s", e)
            return 0

    @staticmethod
    def delete_all_chunks() -> int:
        """Delete all chunks from the database and return count of deleted chunks"""
        try:
            chunk_count = DocumentChunk.query.count()
            DocumentChunk.query.delete()
            db.session.commit()
            return chunk_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting all chunks: %s", e)
            return 0

    @staticmethod
    def delete_all_documents() -> int:
        """Delete all documents from the database and return count of deleted documents"""
        try:
            document_count = Document.query.count()
            Document.query.delete()
            db.session.commit()
            return document_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting all documents: %s", e)
            return 0

    @staticmethod
    def delete_all_model_patterns() -> int:
        """Delete all model patterns from the database and return count of deleted patterns"""
        try:
            pattern_count = ModelPattern.query.count()
            ModelPattern.query.delete()
            db.session.commit()
            return pattern_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting all model patterns: %s", e)
            return 0

    @staticmethod
    def reset_document_processing_status() -> int:
        """Reset all document statuses to 'uploaded' and return count of updated documents"""
        try:
            documents = Document.query.filter_by(status="completed").all()
            document_count = len(documents)

            for document in documents:
                document.status = "uploaded"
                document.processing_date = None

            db.session.commit()
            return document_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error resetting document status: %s", e)
            return 0

    @staticmethod
    def delete_documents_by_status(status: str) -> int:
        """Delete all documents with a specific status and return count of deleted documents"""
        try:
            documents = Document.query.filter_by(status=status).all()
            document_count = len(documents)

            for document in documents:
                # Delete associated chunks first
                DatabaseService.delete_document_chunks(document.id)
                # Delete the document
                db.session.delete(document)

            db.session.commit()
            return document_count
        except Exception as e:
            db.session.rollback()
            logger.error("❌ Error deleting documents by status: %s", e)
            return 0

    @staticmethod
    @staticmethod
    def get_heading_context(limit: Optional[int] = 200) -> Dict[str, Any]:
        """Get distinct heading ids/titles and parent relationships for LLM guidance"""
        try:
            # Distinct headings - sắp xếp theo thứ tự logic: level, sau đó theo alphabet
            heading_query = (
                db.session.query(
                    DocumentChunk.heading_id,
                    DocumentChunk.heading_title,
                    DocumentChunk.heading_parent_id,
                    DocumentChunk.heading_level,
                    func.count(DocumentChunk.id).label("count"),
                )
                .filter(DocumentChunk.heading_id.isnot(None))
                .filter(DocumentChunk.heading_level <= 3)  # Chỉ lấy level 1 và 2
                .group_by(
                    DocumentChunk.heading_id,
                    DocumentChunk.heading_title,
                    DocumentChunk.heading_parent_id,
                    DocumentChunk.heading_level,
                )
                .order_by(
                    DocumentChunk.heading_level.asc(),  # Level 1 trước
                    DocumentChunk.heading_id.asc(),  # Theo thứ tự ID (thứ tự xuất hiện tự nhiên)
                )
            )
            if limit is not None:
                heading_query = heading_query.limit(limit)

            rows = heading_query.all()

            headings = [
                {
                    "id": h_id,
                    "title": title,
                    "parent_id": parent_id,
                    "level": level,
                    "count": int(count or 0),
                }
                for h_id, title, parent_id, level, count in rows
            ]

            # Collect a simple parent->children map
            parent_map: Dict[str, List[str]] = {}
            for h in headings:
                pid = h.get("parent_id")
                if pid:
                    parent_map.setdefault(pid, []).append(h["id"])

            distinct_heading_count = (
                db.session.query(func.count(func.distinct(DocumentChunk.heading_id)))
                .filter(DocumentChunk.heading_id.isnot(None))
                .scalar()
            )

            return {
                "headings": headings,
                "distinct_heading_count": int(distinct_heading_count or 0),
                "parents": parent_map,
                "truncated": limit is not None
                and len(headings) < (distinct_heading_count or 0),
            }
        except Exception as e:
            logger.error("❌ Error getting heading context: %s", e)
            return {
                "headings": [],
                "distinct_heading_count": 0,
                "parents": {},
                "truncated": False,
            }
