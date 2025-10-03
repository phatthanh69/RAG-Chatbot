#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check database tables and model_patterns data
"""

from app import create_app
from app.core.extensions import db

app = create_app()

with app.app_context():
    from sqlalchemy import text

    # Check tables
    result = db.session.execute(
        text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
    )
    tables = [row[0] for row in result]

    print("Tables in database:")
    for table in sorted(tables):
        print(f"  - {table}")

    # Check model_patterns table
    if "model_patterns" in tables:
        result = db.session.execute(text("SELECT COUNT(*) FROM model_patterns"))
        count = result.scalar()
        print(f"\nmodel_patterns table has {count} records")

        # Show some records
        result = db.session.execute(
            text(
                "SELECT id, pattern_name, pattern_regex, confidence_score FROM model_patterns LIMIT 5"
            )
        )
        records = result.fetchall()
        print("\nSample records:")
        for record in records:
            print(f"  ID {record[0]}: {record[1]} - {record[2]} (conf: {record[3]})")
    else:
        print("\nmodel_patterns table does not exist!")

    # Check document_chunks for headings
    if "document_chunks" in tables:
        result = db.session.execute(
            text(
                "SELECT COUNT(DISTINCT heading_title) FROM document_chunks WHERE heading_title IS NOT NULL AND heading_title != ''"
            )
        )
        count = result.scalar()
        print(f"\nFound {count} unique, non-empty heading_titles in document_chunks")

        if count > 0:
            print("\nSample heading_titles:")
            result = db.session.execute(
                text(
                    "SELECT DISTINCT heading_title FROM document_chunks WHERE heading_title IS NOT NULL AND heading_title != '' LIMIT 5"
                )
            )
            records = result.fetchall()
            for record in records:
                print(f"  - {record[0]}")
    else:
        print("\ndocument_chunks table does not exist!")
