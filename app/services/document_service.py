"""
Document processing service
Handles document upload, processing, and embedding creation
Now integrated with database storage
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.database_service import DatabaseService
from app.services.model_pattern_service import ModelPatternAnalysisService
from app.utils.response_helpers import get_vietnam_time
from src import chunker, cleaner, embedder, extractor

# Import Enhanced Markdown Service
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from enhanced_markdown_service import EnhancedMarkdownDocumentService

    MARKDOWN_SERVICE_AVAILABLE = True
except ImportError:
    MARKDOWN_SERVICE_AVAILABLE = False
    EnhancedMarkdownDocumentService = None

# Import BM25Service for tokenization
try:
    from app.services.bm25_service import BM25Service

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class DocumentService:
    """Service for handling document processing operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_all_document_files(self, input_path_str: str) -> Dict[str, Any]:
        """
        Get all document files including Markdown
        Enhanced version of extractor.get_document_files
        """
        input_path = Path(input_path_str)

        if not input_path.exists():
            return {
                "pdf_files": [],
                "docx_files": [],
                "md_files": [],
                "errors": [f"Đường dẫn không tồn tại: {input_path_str}"],
            }

        pdf_files = []
        docx_files = []
        md_files = []

        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == ".pdf":
                pdf_files = [input_path]
            elif ext in [".docx", ".doc"]:
                docx_files = [input_path]
            elif ext in [".md", ".markdown"]:
                md_files = [input_path]
            else:
                return {
                    "pdf_files": [],
                    "docx_files": [],
                    "md_files": [],
                    "errors": [f"File không được hỗ trợ: {input_path.name}"],
                }
        else:
            pdf_files = list(input_path.glob("*.pdf"))
            docx_files = list(input_path.glob("*.docx")) + list(
                input_path.glob("*.doc")
            )
            md_files = list(input_path.glob("*.md")) + list(
                input_path.glob("*.markdown")
            )

        if not pdf_files and not docx_files and not md_files:
            return {
                "pdf_files": [],
                "docx_files": [],
                "md_files": [],
                "errors": [f"Không tìm thấy file được hỗ trợ trong: {input_path_str}"],
            }

        return {
            "pdf_files": pdf_files,
            "docx_files": docx_files,
            "md_files": md_files,
            "errors": [],
        }

    def process_documents(
        self,
        input_path: str,
        output_dir: str,
        files: Optional[List[str]] = None,
        extract_patterns: bool = False,
    ) -> Dict[str, Any]:
        """
        Process documents from input path and create embeddings, storing in database

        Args:
            input_path: Path to input documents
            output_dir: Output directory for processed files
            files: Optional list of specific files to process
            extract_patterns: Whether to extract LLM patterns from document headings

        Returns:
            Dict with success and error information
        """
        try:
            # Ensure output directory exists
            Path(output_dir).mkdir(exist_ok=True)

            # Get all document files (including Markdown)
            doc_info = self.get_all_document_files(input_path)
            if doc_info["errors"]:
                return {"success": [], "errors": doc_info["errors"]}

            pdf_files = doc_info["pdf_files"]
            docx_files = doc_info["docx_files"]
            md_files = doc_info["md_files"]

            all_files = pdf_files + docx_files + md_files

            # Filter files if specific files provided
            if files:
                file_names = [Path(f).name for f in files]
                all_files = [f for f in all_files if f.name in file_names]

            if not all_files:
                return {"success": [], "errors": ["No files found to process"]}

            success_files = []
            error_files = []
            processed_documents = []

            for file_path in all_files:
                try:
                    ext = file_path.suffix.lower()
                    if ext == ".pdf":
                        file_type = "PDF"
                    elif ext in [".md", ".markdown"]:
                        file_type = "MARKDOWN"
                    else:
                        file_type = "DOCX"
                    self.logger.info(f"Processing {file_type} file: {file_path}")

                    # Find existing document by file path or create new one
                    existing_docs = DatabaseService.get_all_documents()
                    document = None

                    # Look for existing document with same file path or name
                    for doc in existing_docs:
                        # Access actual values from SQLAlchemy model
                        doc_file_path = getattr(doc, "file_path", None) or ""
                        doc_file_name = getattr(doc, "file_name", None) or ""
                        doc_original_file_name = (
                            getattr(doc, "original_file_name", None) or ""
                        )

                        if (
                            doc_file_path == str(file_path)
                            or doc_file_name == file_path.name
                            or doc_original_file_name == file_path.name
                        ):
                            document = doc
                            break

                    if document is None:
                        # Create new document if not found
                        document = DatabaseService.create_document(
                            file_name=file_path.name,
                            original_file_name=file_path.name,
                            file_path=str(file_path),
                            file_size=file_path.stat().st_size,
                            mime_type=file_type.lower(),
                            metadata={
                                "processing_started": get_vietnam_time().isoformat()
                            },
                        )

                    # Check if document already has chunks (already processed)
                    doc_id = getattr(document, "id", 0)
                    chunk_count = DatabaseService.get_document_chunk_count(doc_id)
                    doc_status = getattr(document, "status", None) or ""
                    if chunk_count > 0 and doc_status == "completed":
                        self.logger.info(
                            f"Document {doc_id} is already completed with {chunk_count} chunks, skipping processing"
                        )
                        processed_documents.append(document.to_dict())
                        success_files.append(
                            str(file_path)
                        )  # Add to success for consistency
                        continue

                    if chunk_count > 0:
                        self.logger.info(
                            f"Document {doc_id} already has {chunk_count} chunks, deleting them for re-processing"
                        )
                        DatabaseService.delete_document_chunks(doc_id)

                    # Update document status to processing
                    DatabaseService.update_document_status(doc_id, "processing")

                    # Check if chunks file already exists
                    expected_jsonl = Path(output_dir) / f"{file_path.stem}.jsonl"
                    if expected_jsonl.exists():
                        self.logger.info(
                            f"Chunks file already exists: {expected_jsonl}, skipping chunking"
                        )
                        result = {"success": [str(expected_jsonl)], "errors": []}
                    else:
                        # Process document based on type
                        if (
                            file_type == "MARKDOWN"
                            and MARKDOWN_SERVICE_AVAILABLE
                            and EnhancedMarkdownDocumentService
                        ):
                            # Use Enhanced Markdown Service for .md files
                            try:
                                md_service = EnhancedMarkdownDocumentService()
                                md_result = md_service.process_markdown_file(
                                    file_path=str(file_path),
                                    output_dir=output_dir,
                                    create_embeddings=False,  # We'll handle embeddings separately
                                    store_in_database=False,  # We'll store in database here
                                )

                                if md_result.get("error"):
                                    result = {
                                        "success": [],
                                        "errors": [md_result["error"]],
                                    }
                                else:
                                    # Convert chunks to JSONL format
                                    chunks_file = Path(
                                        md_result.get("chunks_file", expected_jsonl)
                                    )
                                    if chunks_file.exists():
                                        # Convert JSON chunks to JSONL format for compatibility
                                        with open(
                                            chunks_file, "r", encoding="utf-8"
                                        ) as f:
                                            chunks = json.load(f)

                                        with open(
                                            expected_jsonl, "w", encoding="utf-8"
                                        ) as f:
                                            for chunk in chunks:
                                                jsonl_entry = {
                                                    "content": chunk["content"],
                                                    "meta": {
                                                        "source": "markdown",
                                                        "file_name": file_path.name,
                                                        "chunk_index": chunk.get(
                                                            "chunk_index", 0
                                                        ),
                                                        "line_number": chunk.get(
                                                            "line_number", 0
                                                        ),
                                                        "heading_id": chunk.get(
                                                            "heading_id"
                                                        ),
                                                        "heading_title": chunk.get(
                                                            "heading_title"
                                                        ),
                                                        "heading_parent_id": chunk.get(
                                                            "heading_parent_id"
                                                        ),
                                                        "heading_level": chunk.get(
                                                            "heading_level"
                                                        ),
                                                        "is_heading": chunk.get(
                                                            "is_heading", False
                                                        ),
                                                        "block_type": chunk.get(
                                                            "block_type", "text"
                                                        ),
                                                    },
                                                }
                                                f.write(
                                                    json.dumps(
                                                        jsonl_entry, ensure_ascii=False
                                                    )
                                                    + "\n"
                                                )

                                        result = {
                                            "success": [str(expected_jsonl)],
                                            "errors": [],
                                        }
                                    else:
                                        result = {
                                            "success": [],
                                            "errors": ["Failed to create chunks file"],
                                        }
                            except Exception as e:
                                self.logger.error(
                                    f"Error processing Markdown file: {e}"
                                )
                                result = {
                                    "success": [],
                                    "errors": [f"Markdown processing failed: {str(e)}"],
                                }
                        else:
                            # Process PDF/DOCX using existing chunker
                            result = chunker.pdf_to_chunks_jsonl(
                                str(file_path), output_dir
                            )

                    if result["errors"]:
                        DatabaseService.update_document_status(doc_id, "error")
                        error_files.append(
                            {"file": str(file_path), "error": result["errors"]}
                        )
                        continue

                    # Process embeddings for each chunk file and store in database
                    jsonl_file = None
                    for jsonl_path in result["success"]:
                        try:
                            jsonl_file = Path(jsonl_path)
                            embedding_output_file = jsonl_file.parent / (
                                jsonl_file.stem + "_embedded.jsonl"
                            )

                            self.logger.info(f"Creating embeddings for: {jsonl_file}")

                            if embedding_output_file.exists():
                                self.logger.info(
                                    f"Embeddings file already exists: {embedding_output_file}, skipping embedding creation"
                                )
                            else:
                                embedder.process_jsonl_embeddings(
                                    str(jsonl_file),
                                    output_file=str(embedding_output_file),
                                )

                            # Read the embedded data and store in database
                            self._store_embeddings_in_database(
                                doc_id, str(embedding_output_file)
                            )

                            self.logger.info(
                                f"Embeddings created and stored for: {embedding_output_file}"
                            )

                        except Exception as e:
                            error_file = (
                                str(jsonl_file) if jsonl_file else str(jsonl_path)
                            )
                            error_files.append(
                                {
                                    "file": error_file,
                                    "error": f"Embedding failed: {str(e)}",
                                }
                            )

                    # Update document status to completed
                    DatabaseService.update_document_status(
                        doc_id, "completed", get_vietnam_time()
                    )
                    processed_documents.append(document.to_dict())
                    success_files.extend(result["success"])

                except Exception as e:
                    error_files.append({"file": str(file_path), "error": str(e)})

            # Optional: Extract patterns from processed documents
            pattern_extraction_result = None
            if extract_patterns and success_files:
                self.logger.info("Extracting patterns from processed documents...")
                try:
                    pattern_result = self.extract_patterns_from_documents(
                        input_path, files
                    )
                    if pattern_result["success"]:
                        self.logger.info(
                            f"Successfully extracted {pattern_result['patterns_saved']} patterns"
                        )
                        pattern_extraction_result = {
                            "patterns_saved": pattern_result["patterns_saved"],
                            "headings_analyzed": pattern_result["headings_analyzed"],
                            "files_processed": pattern_result["files_processed"],
                        }
                    else:
                        self.logger.warning(
                            f"Pattern extraction failed: {pattern_result.get('errors', [])}"
                        )
                except Exception as e:
                    self.logger.error(f"Error during pattern extraction: {str(e)}")

            return {
                "success": success_files,
                "errors": error_files,
                "processed_documents": processed_documents,
                "pattern_extraction": pattern_extraction_result,
            }

        except Exception as e:
            self.logger.error(f"Error in document processing: {str(e)}")
            return {"success": [], "errors": [str(e)], "processed_documents": []}

    def _store_embeddings_in_database(
        self, document_id: int, embedding_file_path: str
    ) -> None:
        """
        Store embeddings from file into database

        Args:
            document_id: Database ID of the document
            embedding_file_path: Path to the embedding file
        """
        try:
            # Initialize BM25 service for tokenization
            bm25_service = None
            if BM25_AVAILABLE:
                try:
                    bm25_service = BM25Service()  # type: ignore
                    # self.logger.info("BM25Service initialized for tokenization")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize BM25Service: {e}")

            chunks_data = []
            with open(embedding_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        content = data.get("content", "")

                        # Generate tokenized content for BM25
                        tokenized_content = None
                        if bm25_service and content:
                            try:
                                tokenized_content = (
                                    bm25_service._preprocess_text_for_bm25(content)
                                )
                                # self.logger.debug(
                                #     f"Tokenized chunk {line_num}: '{content[:50]}...' -> '{tokenized_content[:50]}...'"
                                # )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to tokenize chunk {line_num}: {e}"
                                )

                        chunk_data = {
                            "chunk_index": data.get("chunk_index", line_num),
                            "content": content,
                            "tokenized_content": tokenized_content,  # Add tokenized content
                            "embedding": data.get("embedding", []),
                            "page": data.get("meta", {}).get("page"),
                            "block_index": data.get("meta", {}).get("block_index"),
                            "bbox": data.get("meta", {}).get("bbox"),
                            "font_info": {
                                "font_size": data.get("meta", {}).get("font_size", {}),
                                "font_sizes": data.get("meta", {}).get(
                                    "font_sizes", []
                                ),
                            },
                            # New hierarchical heading metadata
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
                            .replace(".pdf", ""),
                        }
                        chunks_data.append(chunk_data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse line {line_num} in {embedding_file_path}: {e}"
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

    def get_document_info(self, input_path: str) -> Dict[str, Any]:
        """
        Get information about available documents (including Markdown)

        Args:
            input_path: Path to check for documents

        Returns:
            Dict with document information
        """
        try:
            doc_info = self.get_all_document_files(input_path)
            return doc_info

        except Exception as e:
            self.logger.error(f"Error getting document info: {str(e)}")
            return {
                "pdf_files": [],
                "docx_files": [],
                "md_files": [],
                "errors": [str(e)],
            }

    def get_processing_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get processing status for a job (placeholder for future implementation)

        Args:
            job_id: Job identifier

        Returns:
            Dict with job status
        """
        # This would be implemented with a job tracking system
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Processing completed successfully",
        }

    def list_processed_files(
        self, processed_dir: str = "data/processed"
    ) -> Dict[str, Any]:
        """
        List processed files from database

        Args:
            processed_dir: Directory to scan for processed files (kept for compatibility)

        Returns:
            Dict with file information from database
        """
        try:
            # Get documents from database
            documents = DatabaseService.get_all_documents()

            files_info = []
            for doc in documents:
                upload_date = getattr(doc, "upload_date", None)
                processing_date = getattr(doc, "processing_date", None)

                files_info.append(
                    {
                        "id": doc.id,
                        "name": doc.file_name,
                        "original_name": doc.original_file_name,
                        "path": doc.file_path,
                        "size": doc.file_size,
                        "mime_type": doc.mime_type,
                        "status": doc.status,
                        "upload_date": (
                            upload_date.isoformat() if upload_date else None
                        ),
                        "processing_date": (
                            processing_date.isoformat() if processing_date else None
                        ),
                        "chunks_count": len(doc.chunks),
                        "metadata": doc.doc_metadata,
                    }
                )

            return {"files": files_info}

        except Exception as e:
            self.logger.error(f"Error listing processed files from database: {str(e)}")
            return {"files": [], "error": str(e)}

    def get_processed_documents_from_db(self) -> Dict[str, Any]:
        """
        Get processed documents from database with their chunks

        Returns:
            Dict with documents and their chunks information
        """
        try:
            documents = DatabaseService.get_all_documents()

            documents_info = []
            for doc in documents:
                doc_info = doc.to_dict()
                doc_info["chunks"] = [chunk.to_dict() for chunk in doc.chunks]
                documents_info.append(doc_info)

            return {"documents": documents_info, "total_count": len(documents_info)}

        except Exception as e:
            self.logger.error(
                f"Error getting processed documents from database: {str(e)}"
            )
            return {"documents": [], "total_count": 0, "error": str(e)}

    def extract_patterns_from_documents(
        self, input_path: str, files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract headings from documents and generate LLM patterns

        Args:
            input_path: Path to input documents
            files: Optional list of specific files to process

        Returns:
            Dict with pattern extraction results
        """
        try:
            self.logger.info("Starting pattern extraction from documents")

            # Get all document files (including Markdown)
            doc_info = self.get_all_document_files(input_path)
            if doc_info["errors"]:
                return {"success": False, "errors": doc_info["errors"]}

            pdf_files = doc_info["pdf_files"]
            docx_files = doc_info["docx_files"]
            md_files = doc_info["md_files"]

            all_files = pdf_files + docx_files + md_files

            # Filter files if specific files provided
            if files:
                file_names = [Path(f).name for f in files]
                all_files = [f for f in all_files if f.name in file_names]

            if not all_files:
                return {"success": False, "errors": ["No files found to process"]}

            all_headings = []

            # Extract headings from each document
            for file_path in all_files:
                try:
                    self.logger.info(f"Extracting headings from: {file_path}")

                    # Extract text from document based on file type
                    file_ext = file_path.suffix.lower()

                    if file_ext in [".md", ".markdown"] and MARKDOWN_SERVICE_AVAILABLE:
                        # Use Markdown extractor for .md files
                        from src.markdown_extractor import extract_markdown_blocks

                        blocks = extract_markdown_blocks(file_path)

                        # Extract heading titles from Markdown blocks
                        heading_titles = []
                        for block in blocks:
                            if block.is_heading and block.heading_title:
                                heading_title = block.heading_title.strip()
                                if heading_title and len(heading_title) > 3:
                                    heading_titles.append(heading_title)
                    else:
                        # Use existing extractor for PDF/DOCX files
                        blocks = extractor.extract_document_blocks(file_path)

                        # Extract heading titles directly from blocks (more efficient)
                        heading_titles = []
                        for block in blocks:
                            if block.get("is_heading", False) and block.get(
                                "heading_title"
                            ):
                                heading_title = block.get("heading_title", "").strip()
                                if (
                                    heading_title and len(heading_title) > 3
                                ):  # Filter out very short headings
                                    heading_titles.append(heading_title)

                    self.logger.info(
                        f"Extracted {len(heading_titles)} heading titles from {file_path}"
                    )

                    # Add to collection
                    all_headings.extend(heading_titles)

                except Exception as e:
                    self.logger.error(
                        f"Error extracting headings from {file_path}: {str(e)}"
                    )
                    continue

            if not all_headings:
                return {
                    "success": False,
                    "errors": ["No headings extracted from any documents"],
                }

            # Remove duplicates and sort
            unique_headings = list(set(all_headings))
            unique_headings.sort()

            self.logger.info(f"Total unique headings extracted: {len(unique_headings)}")

            # Use ModelPatternAnalysisService to generate patterns
            pattern_service = ModelPatternAnalysisService()

            # Analyze headings with LLM
            analysis_result = pattern_service.analyze_headings_with_llm(unique_headings)

            if not analysis_result.get("patterns"):
                return {
                    "success": False,
                    "errors": ["No patterns generated from headings"],
                    "headings_analyzed": len(unique_headings),
                }

            # Save patterns to database
            saved_pattern_objects = pattern_service.save_patterns_to_db(analysis_result)

            saved_patterns_count = len(saved_pattern_objects)

            self.logger.info(
                f"Pattern extraction completed. Saved {saved_patterns_count} patterns."
            )

            return {
                "success": True,
                "patterns_saved": saved_patterns_count,
                "headings_analyzed": len(unique_headings),
                "files_processed": len(all_files),
                "analysis_metadata": analysis_result.get("analysis_metadata", {}),
            }

        except Exception as e:
            self.logger.error(f"Error in pattern extraction: {str(e)}")
            return {
                "success": False,
                "errors": [str(e)],
                "headings_analyzed": 0,
                "files_processed": 0,
            }
