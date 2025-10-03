"""
Chat models for storing chat sessions and messages
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.base import db

# Timezone configuration for Vietnam (UTC+7)
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def get_vietnam_now():
    """Get current time in Vietnam timezone"""
    return datetime.now(VIETNAM_TIMEZONE)


class ChatSession(db.Model):
    """Model for storing chat sessions"""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(100))  # Optional user identifier
    created_at = Column(DateTime, default=get_vietnam_now)
    updated_at = Column(DateTime, default=get_vietnam_now, onupdate=get_vietnam_now)
    session_metadata = Column(JSON)  # Additional session metadata

    # Relationship to messages
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
    )

    def __repr__(self):
        return f"<ChatSession {self.session_id}>"

    def to_dict(self):
        """Convert session to dictionary representation"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.session_metadata,
            "messages_count": len(self.messages),
        }


class ChatMessage(db.Model):
    """Model for storing chat messages"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        String(100),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_type = Column(String(20), nullable=False)  # user, bot, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=get_vietnam_now)
    message_metadata = Column(JSON)  # Store sources, scores, etc.

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage {self.session_id}:{self.message_type}>"

    def to_dict(self):
        """Convert message to dictionary representation"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.message_metadata,
        }
