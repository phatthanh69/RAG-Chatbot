#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct Model Name Service - Load actual model names from database instead of using regex
"""

import logging
import time
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz

from ragbot.models.base import db
from sqlalchemy import text

class DirectModelNameService:
    """Service to load and match actual model names from database directly"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache for model names
        self.model_names_cache = None
        self.model_names_cache_time = None
        self.cache_ttl = 1800  # 30 minutes cache
        
    def _extract_model_names_from_db(self) -> Dict[str, List[str]]:
        """Extract actual model names from database heading titles"""
        try:
            # Check cache first
            now = time.time()
            if (
                self.model_names_cache is not None
                and self.model_names_cache_time is not None
                and now - self.model_names_cache_time < self.cache_ttl
            ):
                return self.model_names_cache
            
            # Get all heading titles from database
            result = db.session.execute(text("""
                SELECT DISTINCT heading_title
                FROM document_chunks 
                WHERE heading_title IS NOT NULL AND heading_title != ''
                ORDER BY heading_title
            """))
            
            headings = [row[0] for row in result.fetchall()]
            
            # Categorize model names by pattern/type
            model_categories = {
                "exact_models": [],      # Exact model codes like LS-BE-001, WTX536
                "product_series": [],    # Product series names
                "software_names": [],    # Software products
                "equipment_codes": [],   # Equipment codes
                "all_headings": headings # All headings for fallback
            }
            
            for heading in headings:
                heading_lower = heading.lower()
                
                # Category 1: Exact model codes (alphanumeric with patterns)
                if any(pattern in heading_lower for pattern in [
                    'ls-be-', 'cls-be-', 'wtx', 'be-bas-', 'dcs', 'cr350'
                ]) or heading_lower.replace('-', '').replace(' ', '').isalnum():
                    if len(heading.replace(' ', '').replace('-', '')) <= 15:  # Reasonable model code length
                        model_categories["exact_models"].append(heading)
                
                # Category 2: Software names
                if any(term in heading_lower for term in [
                    'vnemisoft', 'software', 'cloud', 'bas', 'system'
                ]):
                    model_categories["software_names"].append(heading)
                
                # Category 3: Equipment/sensor types
                if any(term in heading_lower for term in [
                    'cảm biến', 'sensor', 'monitor', 'thiết bị'
                ]):
                    model_categories["equipment_codes"].append(heading)
                
                # Category 4: Product series (everything else)
                if heading not in model_categories["exact_models"]:
                    model_categories["product_series"].append(heading)
            
            # Cache the result
            self.model_names_cache = model_categories
            self.model_names_cache_time = now
            
            self.logger.info(f"Loaded {len(headings)} model names from database:")
            self.logger.info(f"  - Exact models: {len(model_categories['exact_models'])}")
            self.logger.info(f"  - Software: {len(model_categories['software_names'])}")
            self.logger.info(f"  - Equipment: {len(model_categories['equipment_codes'])}")
            
            return model_categories
            
        except Exception as e:
            self.logger.error(f"Error extracting model names from database: {e}")
            return {
                "exact_models": [],
                "product_series": [],
                "software_names": [],
                "equipment_codes": [],
                "all_headings": []
            }
    
    def find_matching_models(self, query: str, min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find model names that match the query using fuzzy matching
        
        Args:
            query: User query
            min_similarity: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of matching models with similarity scores
        """
        try:
            model_categories = self._extract_model_names_from_db()
            matches = []
            
            query_lower = query.lower()
            query_clean = query_lower.replace('-', '').replace(' ', '')
            
            # Check all categories
            for category, model_list in model_categories.items():
                if category == "all_headings":
                    continue
                    
                for model_name in model_list:
                    model_lower = model_name.lower()
                    model_clean = model_lower.replace('-', '').replace(' ', '')
                    
                    # Calculate different types of similarity
                    similarities = {
                        "exact": 1.0 if query_lower == model_lower else 0.0,
                        "exact_clean": 1.0 if query_clean == model_clean else 0.0,
                        "contains": 1.0 if query_lower in model_lower or model_lower in query_lower else 0.0,
                        "fuzzy_ratio": fuzz.ratio(query_lower, model_lower) / 100.0,
                        "fuzzy_partial": fuzz.partial_ratio(query_lower, model_lower) / 100.0,
                        "fuzzy_token": fuzz.token_ratio(query_lower, model_lower) / 100.0
                    }
                    
                    # Best similarity score
                    max_similarity = max(similarities.values())
                    
                    if max_similarity >= min_similarity:
                        matches.append({
                            "model_name": model_name,
                            "category": category,
                            "similarity": max_similarity,
                            "match_type": max(similarities, key=similarities.get),
                            "all_similarities": similarities
                        })
            
            # Sort by similarity score (descending)
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error finding matching models: {e}")
            return []
    
    def detect_specific_model_query(self, query: str) -> Dict[str, Any]:
        """
        Detect if query is asking for a specific model (not ambiguous)
        
        Args:
            query: User query
            
        Returns:
            Dict with detection results
        """
        try:
            matches = self.find_matching_models(query, min_similarity=0.7)
            
            if not matches:
                return {
                    "is_specific": False,
                    "reason": "No high-confidence matches found",
                    "matches": []
                }
            
            # Check for high-confidence exact/near-exact matches
            high_confidence_matches = [m for m in matches if m["similarity"] >= 0.85]
            
            if len(high_confidence_matches) == 1:
                return {
                    "is_specific": True,
                    "reason": "Single high-confidence match",
                    "best_match": high_confidence_matches[0],
                    "matches": matches[:5]  # Top 5 for reference
                }
            elif len(high_confidence_matches) > 1:
                # Multiple high-confidence matches - might be ambiguous
                top_score = high_confidence_matches[0]["similarity"]
                second_score = high_confidence_matches[1]["similarity"]
                
                if top_score - second_score >= 0.1:  # Clear winner
                    return {
                        "is_specific": True,
                        "reason": "Clear winner among high-confidence matches",
                        "best_match": high_confidence_matches[0],
                        "matches": matches[:5]
                    }
                else:
                    return {
                        "is_specific": False,
                        "reason": "Multiple high-confidence matches with similar scores",
                        "matches": matches[:5]
                    }
            else:
                return {
                    "is_specific": False,
                    "reason": "No high-confidence matches (best: {:.2f})".format(matches[0]["similarity"]),
                    "matches": matches[:5]
                }
                
        except Exception as e:
            self.logger.error(f"Error detecting specific model query: {e}")
            return {
                "is_specific": False,
                "reason": f"Error in detection: {e}",
                "matches": []
            }
    
    def get_all_model_names(self) -> List[str]:
        """Get all model names for fallback use"""
        try:
            model_categories = self._extract_model_names_from_db()
            return model_categories.get("all_headings", [])
        except Exception as e:
            self.logger.error(f"Error getting all model names: {e}")
            return []