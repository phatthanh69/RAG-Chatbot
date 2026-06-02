#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Auto Pattern Manager - No Flask app dependency
Direct database access for pattern management
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def simple_pattern_check():
    """Simple pattern check without Flask app context"""

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv("config.env")

    # Database configuration
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:123456@localhost:5432/reeco_bot"
    )

    # SQLAlchemy setup without Flask
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    try:
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        print("🔍 Checking LLM patterns in database...")

        # Check pattern count
        result = session.execute(
            text(
                """
            SELECT COUNT(*) as pattern_count 
            FROM model_patterns 
            WHERE is_active = true
        """
            )
        )

        pattern_count = result.scalar()
        print(f"📊 Found {pattern_count} active LLM patterns")

        if pattern_count > 0:
            # Show sample patterns
            result = session.execute(
                text(
                    """
                SELECT pattern_name, pattern_regex, confidence_score
                FROM model_patterns 
                WHERE is_active = true
                ORDER BY confidence_score DESC
                LIMIT 5
            """
                )
            )

            patterns = result.fetchall()
            print("\n🎯 Sample patterns:")
            for i, (name, regex, confidence) in enumerate(patterns, 1):
                print(f"   {i}. {name}: {regex} ({confidence:.2f})")

        session.close()
        return {"pattern_count": pattern_count, "status": "ok"}

    except Exception as e:
        print(f"❌ Error checking patterns: {e}")
        return {"error": str(e), "status": "error"}


def manual_pattern_refresh():
    """Manual pattern refresh using populate script"""
    print("\n🔄 Running manual pattern refresh...")

    try:
        os.system("conda activate dolphin && python populate_llm_patterns.py")
        print("✅ Manual refresh completed")
        return {"refreshed": True}

    except Exception as e:
        print(f"❌ Manual refresh failed: {e}")
        return {"error": str(e), "refreshed": False}


if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE LLM PATTERN MANAGER")
    print("=" * 50)

    # Check current patterns
    status = simple_pattern_check()

    # If no patterns, offer to refresh
    if status.get("pattern_count", 0) == 0:
        print("\n❓ No patterns found. Refresh now? (y/n): ", end="")
        choice = input().lower().strip()

        if choice == "y":
            refresh_result = manual_pattern_refresh()
            if refresh_result.get("refreshed"):
                print("\n🎉 Checking patterns after refresh...")
                simple_pattern_check()
        else:
            print(
                "ℹ️ Skipping refresh. Use populate_llm_patterns.py to generate patterns."
            )

    print(f"\n✅ Pattern management completed!")
