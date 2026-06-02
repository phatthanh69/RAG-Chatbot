#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to populate LLM patterns and test the system
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ragbot.app import create_app
from ragbot.models.model_pattern import ModelPattern
from ragbot.llm.pattern_service import ModelPatternAnalysisService


def populate_and_test_llm_patterns():
    """Populate LLM patterns and test the integration"""

    # Create Flask app
    app = create_app()

    with app.app_context():
        try:
            print("=" * 60)
            print("POPULATING LLM PATTERNS")
            print("=" * 60)

            # Initialize service
            pattern_service = ModelPatternAnalysisService()

            # Check existing patterns
            existing_count = ModelPattern.query.count()
            print(f"\n1. Current patterns in database: {existing_count}")

            if existing_count > 0:
                print("   Existing patterns:")
                for pattern in ModelPattern.query.all():
                    print(f"   - {pattern.pattern_name}: {pattern.pattern_regex}")

            # Run LLM analysis to populate patterns
            print(f"\n2. Running LLM analysis to populate patterns...")
            print("-" * 40)

            result = pattern_service.refresh_patterns_from_headings(force_refresh=True)

            if result.get("refreshed"):
                print(f"✅ Analysis successful!")
                print(f"   Headings analyzed: {result.get('headings_analyzed', 0)}")
                print(f"   Patterns extracted: {result.get('patterns_extracted', 0)}")
                print(f"   Patterns saved: {result.get('patterns_saved', 0)}")
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return

            # Show newly created patterns
            print(f"\n3. LLM-Generated patterns:")
            print("-" * 40)

            patterns = ModelPattern.query.filter(ModelPattern.is_active == True).all()

            for pattern in patterns:
                print(f"\nPattern: {pattern.pattern_name}")
                print(f"  Regex: {pattern.pattern_regex}")
                print(f"  Category: {pattern.category}")
                print(f"  Confidence: {pattern.confidence_score:.2f}")
                print(f"  Examples: {pattern.examples}")

            # Test integration with chatbot service
            print(f"\n4. Testing chatbot service integration:")
            print("-" * 40)

            from ragbot.chat.orchestrator import ChatbotService

            chatbot = ChatbotService()

            # Get patterns through chatbot service
            chatbot_patterns = chatbot._get_model_patterns()
            print(f"Patterns loaded by chatbot: {len(chatbot_patterns)}")

            if chatbot_patterns:
                print("✅ LLM patterns successfully loaded!")
                print("First few patterns:")
                for i, pattern in enumerate(chatbot_patterns[:3], 1):
                    print(f"  {i}. {pattern}")
            else:
                print("❌ No patterns loaded - still using legacy extraction")

            # Test pattern matching
            print(f"\n5. Testing pattern matching:")
            print("-" * 40)

            test_queries = [
                "LS-BE-001 specifications",
                "WTX536 thông số",
                "CLS-BE-001 info",
                "05106 features",
            ]

            for query in test_queries:
                matches = []
                for pattern in chatbot_patterns:
                    import re

                    try:
                        if re.search(pattern, query.lower()):
                            matches.append(pattern)
                    except re.error:
                        # Skip invalid regex patterns
                        continue

                print(f"'{query}' → matches: {len(matches)} patterns")

            print(f"\n🎉 LLM Pattern Population Completed!")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    populate_and_test_llm_patterns()
