#!/usr/bin/env python3
"""
Script to import Q&A data from qa_samples folder to database
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import create_app
from app.models.base import db
from app.services.database_service import DatabaseService


def import_qa_data():
    """Import Q&A data from qa_samples folder to database"""

    # Create Flask app context
    app = create_app()

    with app.app_context():
        try:
            # Path to Q&A file
            qa_file = Path("data/qa_samples/Bo_cau_hoi_BAS_embeddings.jsonl")

            if not qa_file.exists():
                print(f"❌ Q&A file not found: {qa_file}")
                return False

            print(f"📂 Found Q&A file: {qa_file}")

            # Check if Q&A data already exists in database
            from sqlalchemy import text

            existing_chunks = db.session.execute(
                text(
                    "SELECT * FROM document_chunks WHERE clean_pdf_name = 'Bo_cau_hoi_BAS_embeddings'"
                )
            ).fetchall()

            if existing_chunks:
                print(
                    f"✅ Q&A data already exists in database ({len(existing_chunks)} chunks)"
                )
                return True

            # Create document record for Q&A data
            document = DatabaseService.create_document(
                file_name="Bo_cau_hoi_BAS_embeddings.jsonl",
                original_file_name="Bo_cau_hoi_BAS.jsonl",
                file_path=str(qa_file),
                file_size=qa_file.stat().st_size,
                mime_type="application/jsonl",
                metadata={
                    "source": "qa_samples",
                    "type": "q_and_a",
                    "description": "Bộ câu hỏi BAS - Berthing Aid System",
                },
            )

            print(f"📄 Created document record: {document.id}")

            # Read and parse Q&A embeddings file
            chunks_data = []
            with open(qa_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())

                        # Extract question and answer
                        question = data.get("question", "")
                        answer = data.get("answer", "")
                        content = f"Question: {question}\nAnswer: {answer}"

                        chunk_data = {
                            "chunk_index": line_num - 1,
                            "content": content,
                            "embedding": data.get("embedding", []),
                            "page": None,
                            "block_index": None,
                            "bbox": None,
                            "font_info": {},
                            "is_heading": False,
                            "entity": None,
                            "section": None,
                            "clean_pdf_name": "Bo_cau_hoi_BAS_embeddings",
                        }
                        chunks_data.append(chunk_data)

                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                        continue

            if chunks_data:
                # Store chunks in database
                DatabaseService.add_document_chunks(document.id, chunks_data)

                # Update document status
                DatabaseService.update_document_status(document.id, "completed")

                print(
                    f"✅ Successfully imported {len(chunks_data)} Q&A chunks to database"
                )
                print(f"📊 Document ID: {document.id}")
                print(f"🏷️  Q&A Source Name: Bo_cau_hoi_BAS_embeddings")

                return True
            else:
                print("❌ No valid Q&A data found in file")
                return False

        except Exception as e:
            print(f"❌ Error importing Q&A data: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("🚀 Starting Q&A data import...")
    success = import_qa_data()
    if success:
        print("✅ Q&A data import completed successfully!")
        print("\n💡 Now the chatbot can use Q&A data for answering questions.")
    else:
        print("❌ Q&A data import failed!")
        sys.exit(1)
