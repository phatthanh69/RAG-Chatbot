"""
Document models for storing processed documents and their chunks with embeddings
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from ragbot.models.base import db

# Timezone configuration for Vietnam (UTC+7)
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def get_vietnam_now():
    """Get current time in Vietnam timezone"""
    return datetime.now(VIETNAM_TIMEZONE)


class Document(db.Model):
    """Model for storing document metadata"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    original_file_name = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    upload_date = Column(DateTime, default=get_vietnam_now)
    processing_date = Column(DateTime)
    status = Column(
        String(50), default="uploaded"
    )  # uploaded, processing, completed, error
    doc_metadata = Column(JSON)

    # Relationship to chunks
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document {self.file_name}>"

    def to_dict(self):
        """Convert document to dictionary representation"""
        upload_date_val = getattr(self, "upload_date", None)
        processing_date_val = getattr(self, "processing_date", None)
        return {
            "id": self.id,
            "file_name": self.file_name,
            "original_file_name": self.original_file_name,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "upload_date": (
                upload_date_val.isoformat()
                if isinstance(upload_date_val, datetime)
                else None
            ),
            "processing_date": (
                processing_date_val.isoformat()
                if isinstance(processing_date_val, datetime)
                else None
            ),
            "status": self.status,
            "metadata": self.doc_metadata,
            "chunks_count": len(self.chunks),
        }


class DocumentChunk(db.Model):
    """Model for storing document chunks with their embeddings"""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    tokenized_content = Column(Text)  # Vietnamese tokenized content for BM25
    embedding = Column(Vector(1536))  # 1536-dimensional vector for embeddings
    page_number = Column(Integer)
    block_index = Column(Integer)
    bbox = Column(JSON)  # Bounding box coordinates
    font_info = Column(JSON)  # Font information
    # New hierarchical heading fields
    heading_id = Column(String(100))
    heading_title = Column(String(500))
    heading_parent_id = Column(String(100))
    heading_level = Column(Integer)
    is_heading = Column(Boolean, default=False)
    clean_pdf_name = Column(String(255))
    score = Column(Float)  # Relevance score for search results
    created_at = Column(DateTime, default=get_vietnam_now)

    # Relationship to document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk {self.document_id}:{self.chunk_index}>"

    def to_dict(self):
        """Convert chunk to dictionary representation"""
        created_at_val = getattr(self, "created_at", None)
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "tokenized_content": self.tokenized_content,
            "page_number": self.page_number,
            "block_index": self.block_index,
            "bbox": self.bbox,
            "font_info": self.font_info,
            "heading_id": self.heading_id,
            "heading_title": self.heading_title,
            "heading_parent_id": self.heading_parent_id,
            "heading_level": self.heading_level,
            "is_heading": self.is_heading,
            "clean_pdf_name": self.clean_pdf_name,
            "score": self.score,
            "created_at": (
                created_at_val.isoformat()
                if isinstance(created_at_val, datetime)
                else None
            ),
        }
