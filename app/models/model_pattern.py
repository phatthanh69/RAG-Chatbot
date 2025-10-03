#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model for storing LLM-extracted model patterns
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.models.base import db

# Vietnam timezone
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def get_vietnam_now():
    """Get current time in Vietnam timezone"""
    return datetime.now(VIETNAM_TIMEZONE)


class ModelPattern(db.Model):
    """Model for storing LLM-extracted model patterns"""

    __tablename__ = "model_patterns"

    id = Column(Integer, primary_key=True)
    pattern_regex = Column(String(255), nullable=False)  # The actual regex pattern
    pattern_name = Column(String(100), nullable=False)  # Human-readable name
    category = Column(
        String(50), nullable=False
    )  # Category like 'sensor_model', 'software_version'
    description = Column(Text, nullable=True)  # LLM-generated description
    examples = Column(JSONB, nullable=True)  # JSON array of example matches
    confidence_score = Column(
        Float, nullable=False, default=0.0
    )  # LLM confidence (0.0-1.0)
    usage_count = Column(
        Integer, nullable=False, default=0
    )  # How often this pattern is matched
    is_active = Column(
        Boolean, nullable=False, default=True
    )  # Whether to use this pattern
    extraction_method = Column(
        String(50), nullable=False, default="llm"
    )  # 'llm', 'manual', 'hybrid'
    llm_analysis_metadata = Column(JSONB, nullable=True)  # LLM analysis details
    created_at = Column(DateTime, nullable=False, default=get_vietnam_now)
    updated_at = Column(
        DateTime, nullable=False, default=get_vietnam_now, onupdate=get_vietnam_now
    )

    def __repr__(self):
        return f"<ModelPattern(id={self.id}, name='{self.pattern_name}', regex='{self.pattern_regex}')>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "pattern_regex": self.pattern_regex,
            "pattern_name": self.pattern_name,
            "category": self.category,
            "description": self.description,
            "examples": self.examples,
            "confidence_score": self.confidence_score,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "extraction_method": self.extraction_method,
            "llm_analysis_metadata": self.llm_analysis_metadata,
            "created_at": (
                self.created_at.isoformat()
                if hasattr(self, "created_at") and self.created_at
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if hasattr(self, "updated_at") and self.updated_at
                else None
            ),
        }

    def increment_usage(self):
        """Increment usage count when pattern is matched"""
        self.usage_count += 1
        self.updated_at = get_vietnam_now()

    @classmethod
    def get_active_patterns(cls, session, category=None):
        """Get all active patterns, optionally filtered by category"""
        query = session.query(cls).filter(cls.is_active == True)

        if category:
            query = query.filter(cls.category == category)

        return query.order_by(cls.confidence_score.desc(), cls.usage_count.desc()).all()

    @classmethod
    def get_pattern_regexes(cls, session, category=None):
        """Get just the regex patterns as a list"""
        patterns = cls.get_active_patterns(session, category)
        return [p.pattern_regex for p in patterns]
