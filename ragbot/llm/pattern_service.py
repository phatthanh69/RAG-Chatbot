#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-based Model Pattern Analysis Service
Uses Gemini 2.0 Flash to intelligently analyze heading titles and extract model patterns
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ragbot.models.base import db
from ragbot.models.model_pattern import ModelPattern
from ragbot.chat.rag_engine import generate_answer
from ragbot.llm.client import init_genai_client

VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


class ModelPatternAnalysisService:
    """Service to analyze headings using LLM and extract intelligent model patterns"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.genai_client = None

    def _get_genai_client(self):
        """Get or initialize genai client"""
        if self.genai_client is None:
            try:
                self.genai_client = init_genai_client()
                self.logger.info("Initialized Gemini client for pattern analysis")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                self.genai_client = None
        return self.genai_client

    def analyze_headings_with_llm(self, headings: List[str]) -> Dict[str, Any]:
        """
        Use Gemini 2.0 Flash to analyze heading titles and extract intelligent patterns

        Args:
            headings: List of heading titles from database

        Returns:
            Dict containing extracted patterns with metadata
        """
        try:
            client = self._get_genai_client()
            if not client:
                raise Exception("Gemini client not available")

            # Create analysis prompt
            headings_text = "\n".join([f"- {heading}" for heading in headings])

            analysis_prompt = f"""
You are an advanced AI pattern recognition specialist with expertise in industrial product identification and regex pattern engineering. You're analyzing technical product headings to extract precise model identification patterns.

CONTEXT: This is for an industrial IoT chatbot system that needs to identify specific products (sensors, weather stations, software, etc.) from user queries. The patterns will be used in real-time query processing.

HEADINGS TO ANALYZE ({len(headings)} total):
{headings_text}

ADVANCED ANALYSIS REQUIREMENTS:

1. **INTELLIGENT CATEGORIZATION**: 
   - Group products by function/domain (sensors, weather_equipment, software, controllers, etc.)
   - Identify manufacturer patterns (LS-BE series, WTX series, etc.)
   - Distinguish between model codes vs descriptive names

2. **PRECISION REGEX ENGINEERING**:
   - Create patterns that are NEITHER too broad NOR too narrow
   - Account for case variations and potential typos
   - Use Unicode-aware patterns for Vietnamese text if needed
   - Optimize for real-world query matching (users might not type exact format)

3. **CONFIDENCE ASSESSMENT**:
   - High confidence (0.9+): Clear, consistent patterns with multiple examples
   - Medium confidence (0.7-0.8): Recognizable but limited examples
   - Lower confidence (0.5-0.6): Potential patterns needing validation

4. **BUSINESS LOGIC**:
   - Prioritize frequently mentioned products
   - Consider common user query patterns (with/without hyphens, spaces)
   - Account for partial matches (user types "WTX" to find "WTX536")

OUTPUT FORMAT (STRICT JSON):
{{
    "patterns": [
        {{
            "pattern_regex": "[Ll][Ss]-?[Bb][Ee]-?\\d{{3}}",
            "pattern_name": "LS-BE Laser Distance Sensor Series",
            "category": "distance_sensor",
            "description": "Laser-based distance measurement sensors with 3-digit model numbers. Commonly used for industrial positioning and measurement applications.",
            "examples": ["LS-BE-001", "ls-be-002", "LS BE 003"],
            "confidence_score": 0.95,
            "reasoning": "Strong consistent pattern across multiple variants. Handles case variations and separator flexibility.",
            "business_priority": "high",
            "query_variations": ["ls be", "lsbe", "laser sensor"]
        }}
    ],
    "analysis_metadata": {{
        "total_headings_analyzed": {len(headings)},
        "unique_patterns_found": 0,
        "analysis_confidence": 0.0,
        "analysis_timestamp": "{datetime.now().isoformat()}",
        "categories_identified": [],
        "processing_notes": "Analysis completed using Gemini 2.0 Flash Experimental",
        "recommended_refresh_interval_days": 30
    }}
}}

ADVANCED GUIDELINES:
- **Flexibility over Exactness**: Patterns should match common user typing variations
- **Semantic Grouping**: Group by product function, not just string similarity  
- **Real-world Usage**: Consider how users actually search (abbreviated forms, partial matches)
- **Vietnamese Context**: Handle accented characters and Vietnamese naming conventions
- **Performance Optimization**: Prefer efficient regex patterns for real-time matching

EXECUTE COMPREHENSIVE ANALYSIS AND RETURN ONLY VALID JSON:
"""

            # Get LLM analysis using Gemini 2.0 Flash Experimental
            # Use the most advanced model for intelligent pattern analysis
            chat = client.models.generate_content(
                model="gemini-2.5-pro",  # Use the most advanced model
                contents=analysis_prompt,
                config={
                    "temperature": 0.1,  # Low temperature for consistent pattern extraction
                    "max_output_tokens": 10000,
                    "top_p": 0.9,
                    "top_k": 20,
                },
            )
            response = chat.text if chat.text else str(chat)

            # Parse JSON response
            response_text = (
                response.strip() if isinstance(response, str) else str(response).strip()
            )

            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis_result = json.loads(json_str)

                # Validate response structure
                if "patterns" in analysis_result and isinstance(
                    analysis_result["patterns"], list
                ):
                    self.logger.info(
                        f"LLM extracted {len(analysis_result['patterns'])} patterns"
                    )
                    return analysis_result
                else:
                    raise Exception("Invalid LLM response structure")
            else:
                raise Exception("Could not parse JSON from LLM response")

        except Exception as e:
            self.logger.error(f"Error in LLM pattern analysis: {e}")
            # Return empty analysis on error
            return {
                "patterns": [],
                "analysis_metadata": {
                    "total_headings_analyzed": len(headings),
                    "unique_patterns_found": 0,
                    "analysis_confidence": 0.0,
                    "error": str(e),
                    "analysis_timestamp": datetime.now().isoformat(),
                },
            }

    def validate_pattern(
        self, pattern_regex: str, examples: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that a regex pattern actually matches its claimed examples

        Args:
            pattern_regex: The regex pattern to test
            examples: List of example strings that should match

        Returns:
            Dict with validation results
        """
        try:
            compiled_pattern = re.compile(pattern_regex, re.IGNORECASE)

            matches = []
            non_matches = []

            for example in examples:
                if compiled_pattern.search(example.lower()):
                    matches.append(example)
                else:
                    non_matches.append(example)

            match_rate = len(matches) / len(examples) if examples else 0.0

            return {
                "is_valid": match_rate
                >= 0.6,  # Reduce threshold to 60% for more flexibility
                "match_rate": match_rate,
                "matches": matches,
                "non_matches": non_matches,
                "pattern_compiled": True,
            }

        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern_regex}': {e}")
            return {
                "is_valid": False,
                "match_rate": 0.0,
                "matches": [],
                "non_matches": examples,
                "pattern_compiled": False,
                "error": str(e),
            }

    def save_patterns_to_db(
        self, analysis_result: Dict[str, Any]
    ) -> List[ModelPattern]:
        """
        Save LLM-extracted patterns to database

        Args:
            analysis_result: Result from analyze_headings_with_llm()

        Returns:
            List of saved ModelPattern objects
        """
        try:
            saved_patterns = []

            for pattern_data in analysis_result.get("patterns", []):
                # Validate the pattern first
                examples = pattern_data.get("examples", [])
                validation = self.validate_pattern(
                    pattern_data["pattern_regex"], examples
                )

                if not validation["is_valid"]:
                    self.logger.warning(
                        f"Skipping invalid pattern: {pattern_data['pattern_regex']}"
                    )
                    continue

                # Check if pattern already exists
                existing = ModelPattern.query.filter_by(
                    pattern_regex=pattern_data["pattern_regex"]
                ).first()

                if existing:
                    # Update existing pattern with all available data
                    existing.pattern_name = pattern_data.get(
                        "pattern_name", existing.pattern_name
                    )
                    existing.category = pattern_data.get("category", existing.category)
                    existing.description = pattern_data.get(
                        "description", existing.description
                    )
                    existing.examples = pattern_data.get(
                        "examples", existing.examples or []
                    )
                    existing.confidence_score = pattern_data.get(
                        "confidence_score", existing.confidence_score
                    )
                    existing.llm_analysis_metadata = {  # type: ignore
                        "analysis_result": analysis_result.get("analysis_metadata", {}),
                        "validation": validation,
                        "reasoning": pattern_data.get("reasoning", ""),
                        "business_priority": pattern_data.get("business_priority", ""),
                        "query_variations": pattern_data.get("query_variations", []),
                        "updated_at": datetime.now(VIETNAM_TIMEZONE).isoformat(),
                    }
                    existing.updated_at = datetime.now(VIETNAM_TIMEZONE)
                    saved_patterns.append(existing)
                else:
                    # Create new pattern
                    new_pattern = ModelPattern(
                        pattern_regex=pattern_data["pattern_regex"],
                        pattern_name=pattern_data.get(
                            "pattern_name", f"Pattern {pattern_data['pattern_regex']}"
                        ),
                        category=pattern_data.get("category", "unknown"),
                        description=pattern_data.get("description", ""),
                        examples=pattern_data.get("examples", []),
                        confidence_score=pattern_data.get("confidence_score", 0.0),
                        extraction_method="llm",
                    )

                    # Set metadata separately
                    new_pattern.llm_analysis_metadata = {  # type: ignore
                        "analysis_result": analysis_result.get("analysis_metadata", {}),
                        "validation": validation,
                        "reasoning": pattern_data.get("reasoning", ""),
                        "business_priority": pattern_data.get("business_priority", ""),
                        "query_variations": pattern_data.get("query_variations", []),
                        "created_at": datetime.now(VIETNAM_TIMEZONE).isoformat(),
                    }

                    db.session.add(new_pattern)
                    saved_patterns.append(new_pattern)

            db.session.commit()
            self.logger.info(f"Saved {len(saved_patterns)} patterns to database")

            return saved_patterns

        except Exception as e:
            self.logger.error(f"Error saving patterns to database: {e}")
            db.session.rollback()
            return []

    def refresh_patterns_from_headings(
        self, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Main method: Extract all headings, analyze with LLM, and update database

        Args:
            force_refresh: If True, re-analyze even if recent analysis exists

        Returns:
            Dict with refresh results and statistics
        """
        try:
            # Check if we need to refresh (e.g., no patterns exist or patterns are old)
            if not force_refresh:
                recent_patterns = ModelPattern.query.filter(
                    ModelPattern.created_at
                    >= datetime.now(VIETNAM_TIMEZONE) - timedelta(days=7)
                ).count()

                if recent_patterns > 0:
                    self.logger.info(
                        "Recent patterns exist, skipping refresh. Use force_refresh=True to override."
                    )
                    return {
                        "refreshed": False,
                        "reason": "Recent patterns exist",
                        "existing_patterns": recent_patterns,
                    }

            # Get all headings from database
            from sqlalchemy import text

            result = db.session.execute(
                text(
                    """
                SELECT DISTINCT heading_title
                FROM document_chunks 
                WHERE heading_title IS NOT NULL AND heading_title != ''
                ORDER BY heading_title
            """
                )
            )

            headings = [row[0] for row in result.fetchall()]

            if not headings:
                return {
                    "refreshed": False,
                    "reason": "No headings found in database",
                    "headings_count": 0,
                }

            self.logger.info(
                f"Analyzing {len(headings)} headings with Gemini 2.0 Flash"
            )

            # Analyze headings with LLM
            analysis_result = self.analyze_headings_with_llm(headings)

            # Save patterns to database
            saved_patterns = self.save_patterns_to_db(analysis_result)

            return {
                "refreshed": True,
                "headings_analyzed": len(headings),
                "patterns_extracted": len(analysis_result.get("patterns", [])),
                "patterns_saved": len(saved_patterns),
                "analysis_metadata": analysis_result.get("analysis_metadata", {}),
                "saved_patterns": [p.to_dict() for p in saved_patterns],
            }

        except Exception as e:
            self.logger.error(f"Error refreshing patterns from headings: {e}")
            return {
                "refreshed": False,
                "error": str(e),
                "headings_analyzed": 0,
                "patterns_saved": 0,
            }

    def get_active_patterns(self, category: Optional[str] = None) -> List[str]:
        """
        Get active regex patterns from database

        Args:
            category: Optional category filter

        Returns:
            List of regex pattern strings
        """
        try:
            return ModelPattern.get_pattern_regexes(db.session, category)
        except Exception as e:
            self.logger.error(f"Error getting active patterns: {e}")
            return []
