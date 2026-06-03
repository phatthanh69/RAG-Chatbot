"""
Enhanced Markdown Document Service
Full integration: Markdown → Chunks → Embeddings → Database
Similar to PDF extractor but for Markdown files
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths so local modules can be imported
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import app services
from ragbot.db.database_service import DatabaseService
from ragbot.utils.response_helpers import get_vietnam_time
from ragbot.ingestion.cleaner import normalize_text

# Import existing modules from src
from ragbot.ingestion.embedder import (
    create_embeddings,
    init_genai_client,
)
from ragbot.ingestion.markdown_extractor import (
    chunk_markdown_blocks,
    extract_markdown_blocks,
)

# Import BM25Service for tokenization
try:
    from ragbot.retrieval.bm25 import BM25Service as _BM25Service
except ImportError:  # pragma: no cover - optional dependency
    _BM25Service = None

BM25_AVAILABLE = _BM25Service is not None


class EnhancedMarkdownDocumentService:
    """
    Enhanced service for processing Markdown documents
    Complete pipeline: MD → chunks → embeddings → database
    """

    def __init__(self):
        """Initialize service with logger and client"""
        self.logger = logging.getLogger(__name__)
        self._genai_client = None
        self._bm25_service = None

        # Initialize BM25 service if available
        if BM25_AVAILABLE and _BM25Service is not None:
            try:
                self._bm25_service = _BM25Service()
                self.logger.info("BM25Service initialized for tokenization")
            except Exception as e:
                self.logger.warning(f"Failed to initialize BM25Service: {e}")

    @property
    def genai_client(self):
        """Lazy initialization of GenAI client"""
        if self._genai_client is None:
            try:
                self._genai_client = init_genai_client()
                self.logger.info("Google GenAI client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize GenAI client: {e}")
                raise
        return self._genai_client

    def process_markdown_file(
        self,
        file_path: str,
        output_dir: str = "data/processed",
        max_chunk_size: int = 1000,
        create_embeddings: bool = True,
        store_in_database: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a complete Markdown file

        Args:
            file_path: Path to the Markdown file
            output_dir: Output directory
            max_chunk_size: Maximum chunk size
            create_embeddings: Whether to create embeddings
            store_in_database: Whether to store in database

        Returns:
            Dict containing processing information
        """
        try:
            file_path_obj = Path(file_path)
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"🚀 Processing Markdown file: {file_path_obj.name}")

            # Step 1: Extract blocks from Markdown
            blocks = extract_markdown_blocks(file_path_obj)
            self.logger.info(f"📊 Extracted {len(blocks)} blocks")

            # Step 2: Create chunks
            chunks = chunk_markdown_blocks(blocks, max_chunk_size)
            self.logger.info(f"📦 Created {len(chunks)} chunks")

            # Step 3: Save chunks to JSON file
            chunks_file = output_dir_obj / f"{file_path_obj.stem}_chunks.json"
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            result = {
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj),
                "total_blocks": len(blocks),
                "total_chunks": len(chunks),
                "chunks_file": str(chunks_file),
                "embeddings_created": False,
                "stored_in_database": False,
                "document_id": None,
            }

            # Step 4: Create embeddings if requested
            if create_embeddings:
                embeddings_result = self._create_embeddings_for_chunks(
                    chunks, file_path_obj.stem, output_dir_obj
                )
                result.update(embeddings_result)

            # Step 5: Store in database if requested
            if store_in_database:
                db_result = self._store_in_database(
                    file_path_obj,
                    chunks,
                    embeddings_file=(
                        result.get("embeddings_file") if create_embeddings else None
                    ),
                )
                result.update(db_result)

            self.logger.info(f"✅ Successfully processed {file_path_obj.name}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path}: {e}")
            return {
                "error": str(e),
                "file_name": Path(file_path).name if file_path else "unknown",
                "success": False,
            }

    def _create_embeddings_for_chunks(
        self, chunks: List[Dict[str, Any]], base_name: str, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Create embeddings for the prepared chunks.

        Args:
            chunks: Collection of chunk dictionaries.
            base_name: Base name for the output file.
            output_dir: Directory where the embedding file will be written.

        Returns:
            Dictionary describing the embedding generation outcome.
        """
        try:
            self.logger.info("🔮 Creating embeddings for chunks...")

            # Extract content from chunks
            texts = []
            for chunk in chunks:
                content = chunk.get("content", "").strip()
                if content:
                    # Normalize text before generating embeddings
                    normalized_content = normalize_text(content)
                    texts.append(normalized_content)

            if not texts:
                self.logger.warning("No valid content found for embedding")
                return {"embeddings_created": False}

            # Create embeddings using existing embedder
            self.logger.info(f"Creating embeddings for {len(texts)} texts...")
            response = create_embeddings(
                client=self.genai_client,
                texts=texts,
                model="gemini-embedding-001",
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=1536,
            )

            # Process embeddings response
            embeddings = []
            if hasattr(response, "embeddings"):
                embeddings = [emb.values for emb in response.embeddings]
            else:
                # Handle different response format
                embeddings = response if isinstance(response, list) else []

            if len(embeddings) != len(chunks):
                self.logger.warning(
                    f"Embeddings count ({len(embeddings)}) doesn't match chunks count ({len(chunks)})"
                )
                return {"embeddings_created": False}

            # Create JSONL with embeddings
            embeddings_file = output_dir / f"{base_name}_embedded.jsonl"
            embedded_data = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                embedded_entry = {
                    "content": chunk.get("content", ""),
                    "embedding": embedding,
                    "meta": {
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "heading_id": chunk.get("heading_id"),
                        "heading_title": chunk.get("heading_title"),
                        "heading_parent_id": chunk.get("heading_parent_id"),
                        "heading_level": chunk.get("heading_level"),
                        "block_type": chunk.get("block_type"),
                        "is_heading": chunk.get("is_heading", False),
                        "file_name": base_name + ".md",
                        "page": 1,  # Markdown doesn't have pages
                        "chunk_index": i,
                    },
                }
                embedded_data.append(embedded_entry)

            # Save to JSONL
            with open(embeddings_file, "w", encoding="utf-8") as f:
                for entry in embedded_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self.logger.info(f"✅ Created embeddings for {len(embedded_data)} chunks")
            self.logger.info(f"📁 Embeddings saved to: {embeddings_file}")

            return {
                "embeddings_created": True,
                "embeddings_file": str(embeddings_file),
                "embeddings_count": len(embedded_data),
            }

        except Exception as e:
            self.logger.error(f"❌ Error creating embeddings: {e}")
            return {"embeddings_created": False, "embeddings_error": str(e)}

    def _store_in_database(
        self,
        file_path: Path,
        chunks: List[Dict[str, Any]],
        embeddings_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist the document and its chunks to the database.

        Args:
            file_path: Path to the original file.
            chunks: Collection of chunk dictionaries.
            embeddings_file: Optional path to the embeddings file.

        Returns:
            Dictionary describing the database write outcome.
        """
        try:
            self.logger.info("💾 Storing in database...")

            # Create document record
            document = DatabaseService.create_document(
                file_name=file_path.name,
                original_file_name=file_path.name,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                mime_type="text/markdown",
                metadata={
                    "processing_started": get_vietnam_time().isoformat(),
                    "total_chunks": len(chunks),
                },
            )

            document_id = getattr(document, "id", 0)
            self.logger.info(f"📄 Created document record ID: {document_id}")

            # Update document status to processing
            DatabaseService.update_document_status(document_id, "processing")

            # Store chunks with embeddings if available
            if embeddings_file and Path(embeddings_file).exists():
                self._store_embeddings_in_database(document_id, embeddings_file)
            else:
                # Store chunks without embeddings
                self._store_chunks_in_database(document_id, chunks)

            # Update document status to completed
            DatabaseService.update_document_status(
                document_id, "completed", get_vietnam_time()
            )

            self.logger.info(
                f"✅ Successfully stored document {document_id} in database"
            )

            return {
                "stored_in_database": True,
                "document_id": document_id,
                "chunks_count": len(chunks),
            }

        except Exception as e:
            self.logger.error(f"❌ Error storing in database: {e}")
            return {"stored_in_database": False, "database_error": str(e)}

    def _store_embeddings_in_database(
        self, document_id: int, embeddings_file: str
    ) -> None:
        """
        Load embeddings from file and save them to the database.
        Mirrors the behaviour of ``DocumentService._store_embeddings_in_database``.
        """
        try:
            chunks_data = []
            with open(embeddings_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        content = data.get("content", "")

                        # Generate tokenized content for BM25
                        tokenized_content = None
                        if self._bm25_service and content:
                            try:
                                processed_text = self._bm25_service._preprocess_text_for_bm25(
                                    content
                                )
                                tokenized_content = processed_text
                            except Exception as e:
                                self.logger.debug(
                                    f"BM25 tokenization failed for chunk {line_num}: {e}"
                                )

                        chunk_data = {
                            "chunk_index": data.get("meta", {}).get(
                                "chunk_index", line_num
                            ),
                            "content": content,
                            "tokenized_content": tokenized_content,
                            "embedding": data.get("embedding", []),
                            "page": data.get("meta", {}).get("page", 1),
                            "block_index": data.get("meta", {}).get(
                                "chunk_index", line_num
                            ),
                            "bbox": None,  # Markdown doesn't have bounding boxes
                            "font_info": {},  # Markdown doesn't have font info
                            # Hierarchical heading fields (Markdown specific)
                            "heading_id": data.get("meta", {}).get("heading_id"),
                            "heading_title": data.get("meta", {}).get("heading_title"),
                            "heading_parent_id": data.get("meta", {}).get(
                                "heading_parent_id"
                            ),
                            "heading_level": data.get("meta", {}).get("heading_level"),
                            "entity": data.get("meta", {}).get("entity"),
                            "section": data.get("meta", {}).get("section"),
                            "is_heading": data.get("meta", {}).get("is_heading", False),
                            "clean_pdf_name": data.get("meta", {})
                            .get("file_name", "")
                            .replace(".md", ""),
                        }
                        chunks_data.append(chunk_data)

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse line {line_num} in {embeddings_file}: {e}"
                        )
                        continue

            if chunks_data:
                DatabaseService.add_document_chunks(document_id, chunks_data)
                tokenized_count = sum(
                    1 for chunk in chunks_data if chunk.get("tokenized_content")
                )
                self.logger.info(
                    f"Stored {len(chunks_data)} chunks for document {document_id} "
                    f"({tokenized_count} with tokenized content)"
                )

        except Exception as e:
            self.logger.error(f"Error storing embeddings in database: {e}")
            raise

    def _store_chunks_in_database(
        self, document_id: int, chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Store chunks in the database when embeddings are not available.
        """
        try:
            chunks_data = []
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")

                # Generate tokenized content for BM25
                tokenized_content = None
                if self._bm25_service and content:
                    try:
                        tokenized_content = self._bm25_service._preprocess_text_for_bm25(
                            content
                        )
                    except Exception as e:
                        self.logger.debug(
                            f"BM25 tokenization failed for chunk {i}: {e}"
                        )

                chunk_data = {
                    "chunk_index": i,
                    "content": content,
                    "tokenized_content": tokenized_content,
                    "embedding": [],  # Empty embedding
                    "page": 1,  # Markdown doesn't have pages
                    "block_index": i,
                    "bbox": None,
                    "font_info": {},
                    # Hierarchical heading fields
                    "heading_id": chunk.get("heading_id"),
                    "heading_title": chunk.get("heading_title"),
                    "heading_parent_id": chunk.get("heading_parent_id"),
                    "heading_level": chunk.get("heading_level"),
                    "entity": chunk.get("entity"),
                    "section": chunk.get("section"),
                    "is_heading": chunk.get("is_heading", False),
                    "clean_pdf_name": chunk.get("file_name", "").replace(".md", ""),
                }
                chunks_data.append(chunk_data)

            if chunks_data:
                DatabaseService.add_document_chunks(document_id, chunks_data)
                self.logger.info(
                    f"Stored {len(chunks_data)} chunks for document {document_id}"
                )

        except Exception as e:
            self.logger.error(f"Error storing chunks in database: {e}")
            raise

    def batch_process_markdown_directory(
        self,
        input_dir: str,
        output_dir: str = "data/processed",
        max_chunk_size: int = 1000,
        create_embeddings: bool = True,
        store_in_database: bool = True,
        file_pattern: str = "*.md",
    ) -> Dict[str, Any]:
        """
        Process a batch of Markdown files within a directory.

        Args:
            input_dir: Directory containing Markdown files.
            output_dir: Output directory for generated artifacts.
            max_chunk_size: Maximum size for each chunk.
            create_embeddings: Whether embeddings should be generated.
            store_in_database: Whether results should be stored in the database.
            file_pattern: Glob pattern used to select files.

        Returns:
            Dictionary with details about the batch processing results.
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            # Find Markdown files
            markdown_files = list(input_path.glob(file_pattern))
            if not markdown_files:
                self.logger.warning(f"No Markdown files found in {input_dir}")
                return {
                    "success": [],
                    "errors": [
                        f"No Markdown files found matching pattern {file_pattern}"
                    ],
                    "total_files": 0,
                    "processed_files": 0,
                }

            self.logger.info(
                f"🚀 Batch processing {len(markdown_files)} Markdown files"
            )

            results = {
                "success": [],
                "errors": [],
                "total_files": len(markdown_files),
                "processed_files": 0,
            }

            for file_path in markdown_files:
                try:
                    self.logger.info(f"Processing: {file_path.name}")
                    result = self.process_markdown_file(
                        str(file_path),
                        output_dir=output_dir,
                        max_chunk_size=max_chunk_size,
                        create_embeddings=create_embeddings,
                        store_in_database=store_in_database,
                    )

                    if "error" not in result:
                        results["success"].append(
                            {"file": str(file_path), "result": result}
                        )
                        results["processed_files"] += 1
                    else:
                        results["errors"].append(
                            {"file": str(file_path), "error": result["error"]}
                        )

                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    results["errors"].append({"file": str(file_path), "error": str(e)})

            self.logger.info(
                f"✅ Batch processing completed: {results['processed_files']}/{results['total_files']} files"
            )
            return results

        except Exception as e:
            self.logger.error(f"❌ Error in batch processing: {e}")
            return {
                "success": [],
                "errors": [str(e)],
                "total_files": 0,
                "processed_files": 0,
            }

    def convert_to_jsonl(
        self, file_path: str, output_file: str, max_chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Convert a Markdown file to JSONL format (without embeddings).
        Output is compatible with the RAG system.

        Args:
            file_path: Path to the Markdown file.
            output_file: Destination JSONL file.
            max_chunk_size: Maximum size for each chunk.

        Returns:
            Dictionary describing the conversion result.
        """
        try:
            file_path_obj = Path(file_path)
            output_file_obj = Path(output_file)

            self.logger.info(f"🔄 Converting {file_path_obj.name} to JSONL...")

            # Extract blocks and create chunks
            blocks = extract_markdown_blocks(file_path_obj)
            chunks = chunk_markdown_blocks(blocks, max_chunk_size)

            # Create JSONL entries
            jsonl_entries = []
            for i, chunk in enumerate(chunks):
                entry = {
                    "content": chunk.get("content", ""),
                    "metadata": {
                        "source": file_path_obj.name,
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "heading_id": chunk.get("heading_id"),
                        "heading_title": chunk.get("heading_title"),
                        "heading_parent_id": chunk.get("heading_parent_id"),
                        "heading_level": chunk.get("heading_level"),
                        "block_type": chunk.get("block_type"),
                        "is_heading": chunk.get("is_heading", False),
                        "page": 1,
                        "chunk_index": i,
                    },
                }
                jsonl_entries.append(entry)

            # Write JSONL
            with open(output_file_obj, "w", encoding="utf-8") as f:
                for entry in jsonl_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self.logger.info(f"✅ Converted to JSONL: {output_file_obj}")

            return {
                "success": True,
                "input_file": str(file_path_obj),
                "output_file": str(output_file_obj),
                "total_chunks": len(chunks),
                "file_size": output_file_obj.stat().st_size,
            }

        except Exception as e:
            self.logger.error(f"❌ Error converting to JSONL: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_file": str(file_path) if "file_path" in locals() else "unknown",
            }


# Convenience helpers for common scenarios
def process_markdown_file(
    file_path: str,
    output_dir: str = "data/processed",
    create_embeddings: bool = True,
    store_in_database: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for processing a single Markdown file.
    """
    service = EnhancedMarkdownDocumentService()
    return service.process_markdown_file(
        file_path=file_path,
        output_dir=output_dir,
        create_embeddings=create_embeddings,
        store_in_database=store_in_database,
    )


def batch_process_markdown_files(
    input_dir: str,
    output_dir: str = "data/processed",
    create_embeddings: bool = True,
    store_in_database: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for batch-processing Markdown files.
    """
    service = EnhancedMarkdownDocumentService()
    return service.batch_process_markdown_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        create_embeddings=create_embeddings,
        store_in_database=store_in_database,
    )


if __name__ == "__main__":
    # Demo usage
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        logging.info("Usage: python enhanced_markdown_service.py <markdown_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = process_markdown_file(file_path)

    logging.info(json.dumps(result, indent=2, ensure_ascii=False))
