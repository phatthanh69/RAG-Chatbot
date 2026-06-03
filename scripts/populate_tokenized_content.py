#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to populate tokenized_content for existing chunks in database
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask

from ragbot.app import create_app
from ragbot.retrieval.bm25 import BM25Service


def populate_tokenized_content():
    """Populate tokenized_content for all existing chunks"""

    print("Populating tokenized content for existing chunks...")
    print("=" * 60)

    # Create Flask app context
    app = create_app()

    with app.app_context():
        # Initialize BM25 service
        bm25_service = BM25Service()

        # Update tokenized content for all chunks
        success = bm25_service.update_tokenized_content_for_all_chunks()

        if success:
            print("✅ Successfully populated tokenized content for all chunks!")

            # Test the improved performance
            print("\nTesting BM25 retriever with tokenized content...")

            success = bm25_service.initialize_retriever(force_rebuild=True)
            if success:
                print("✅ BM25 retriever initialized successfully!")

                # Test search
                test_queries = [
                    "tốc độ cập cảng",
                    "chi phí vận chuyển",
                    "quy trình xử lý container",
                ]

                for query in test_queries:
                    results = bm25_service.search_chunks(query, limit=3)
                    print(f"\nQuery: '{query}' -> Found {len(results)} results")

            else:
                print("❌ Failed to initialize BM25 retriever")
        else:
            print("❌ Failed to populate tokenized content")


if __name__ == "__main__":
    populate_tokenized_content()
