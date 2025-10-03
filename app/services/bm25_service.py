"""
BM25 service for keyword-based document search
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain.schema import Document as LangchainDocument
from langchain_community.retrievers import BM25Retriever
from rapidfuzz import fuzz
from sqlalchemy import func, text

try:
    from underthesea import word_tokenize

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False

from app.models.base import db
from app.models.document import Document, DocumentChunk


class BM25Service:
    """Service class for BM25 keyword search operations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._retriever = None
        self._documents = []
        self._chunks_data = []

    def _tokenize_vietnamese_text(self, text: str) -> str:
        """
        Tokenize Vietnamese text using underthesea library

        Args:
            text: Input text to tokenize

        Returns:
            Tokenized text with underscores connecting word components
        """
        if not UNDERTHESEA_AVAILABLE:
            self.logger.warning(
                "underthesea library not available, using default tokenization"
            )
            return text

        try:
            # Use underthesea to tokenize Vietnamese text
            # format="text" returns space-separated tokens with underscores for compound words
            tokenized = word_tokenize(text, format="text")  # type: ignore
            # Ensure we return a string
            if isinstance(tokenized, list):
                return " ".join(tokenized)
            return str(tokenized)
        except Exception as e:
            self.logger.warning(
                f"Error tokenizing Vietnamese text: {e}, using original text"
            )
            return text

    def _preprocess_text_for_bm25(self, text: str) -> str:
        """
        Preprocess text for better BM25 performance with Vietnamese

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text.strip())

        # Tokenize Vietnamese text
        tokenized_text = self._tokenize_vietnamese_text(text)

        return tokenized_text

    def _build_retriever(self, chunks_data: List[Dict[str, Any]]) -> BM25Retriever:
        """
        Build BM25 retriever from chunks data

        Args:
            chunks_data: List of chunk dictionaries with content and metadata

        Returns:
            BM25Retriever instance
        """
        try:
            # Convert chunks to Langchain documents
            langchain_docs = []
            for i, chunk in enumerate(chunks_data):
                content = chunk.get("content", "")
                tokenized_content = chunk.get("tokenized_content")

                # Use pre-tokenized content if available, otherwise tokenize on-the-fly
                if tokenized_content:
                    processed_content = tokenized_content
                    # self.logger.debug(f"Using pre-tokenized content for chunk {i}")
                else:
                    processed_content = self._preprocess_text_for_bm25(content)
                    # self.logger.debug(f"Tokenizing content on-the-fly for chunk {i}")

                metadata = {
                    "chunk_id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "page": chunk.get("page"),
                    "file_name": chunk.get("file_name"),
                    "original_file_name": chunk.get("original_file_name"),
                    "clean_pdf_name": chunk.get("clean_pdf_name"),
                    "entity": chunk.get("entity"),
                    "section": chunk.get("section"),
                    "is_heading": chunk.get("is_heading"),
                    "original_index": i,  # Keep track of original position
                    "original_content": content,  # Keep original content for display
                }

                doc = LangchainDocument(
                    page_content=processed_content, metadata=metadata
                )
                langchain_docs.append(doc)

            # Create BM25 retriever
            retriever = BM25Retriever.from_documents(langchain_docs)
            return retriever

        except Exception as e:
            self.logger.error(f"Error building BM25 retriever: {e}")
            raise

    def _load_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Load all document chunks from database

        Returns:
            List of chunk dictionaries
        """
        try:
            # Import Flask for app context
            from flask import current_app

            # Check if we're in an app context
            try:
                # Use hasattr to check for app context instead of _get_current_object
                hasattr(current_app, "app_context")
                in_app_context = True
            except RuntimeError:
                in_app_context = False

            if not in_app_context:
                self.logger.warning(
                    "Not in Flask app context, cannot load chunks from database"
                )
                return []

            results = (
                db.session.query(DocumentChunk)
                .join(Document)
                .filter(Document.status == "completed")
                .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                .all()
            )

            chunks_data = []
            for chunk in results:
                chunks_data.append(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "tokenized_content": chunk.tokenized_content,  # Pre-tokenized content
                        "page": chunk.page_number,
                        "block_index": chunk.block_index,
                        "bbox": chunk.bbox,
                        "font_info": chunk.font_info,
                        # heading metadata
                        "heading_id": getattr(chunk, "heading_id", None),
                        "heading_title": getattr(chunk, "heading_title", None),
                        "heading_parent_id": getattr(chunk, "heading_parent_id", None),
                        "heading_level": getattr(chunk, "heading_level", None),
                        "is_heading": chunk.is_heading,
                        "clean_pdf_name": chunk.clean_pdf_name,
                        "file_name": chunk.document.file_name,
                        "original_file_name": chunk.document.original_file_name,
                        "metadata": chunk.document.doc_metadata,
                    }
                )

            self.logger.debug(
                f"Successfully loaded {len(chunks_data)} chunks from database"
            )
            return chunks_data

        except Exception as e:
            self.logger.error(f"Error loading chunks from database: {e}")
            return []

    def initialize_retriever(self, force_rebuild: bool = False) -> bool:
        """
        Initialize or rebuild the BM25 retriever

        Args:
            force_rebuild: Force rebuilding even if retriever exists

        Returns:
            True if successful, False otherwise
        """
        try:
            if self._retriever is not None and not force_rebuild:
                return True

            self.logger.debug("Loading chunks for BM25 indexing...")
            chunks_data = self._load_all_chunks()

            if not chunks_data:
                self.logger.warning("No chunks found for BM25 indexing")
                return False

            self.logger.debug(
                f"Building BM25 retriever with {len(chunks_data)} chunks..."
            )
            self._retriever = self._build_retriever(chunks_data)
            self._chunks_data = chunks_data

            self.logger.info("BM25 retriever initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing BM25 retriever: {e}")
            return False

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        active_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using BM25 with actual BM25 scores

        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum score threshold (applied after BM25 scoring)
            active_headings: Filter results by specific headings (optional list)

        Returns:
            List of relevant chunks with actual BM25 scores
        """
        try:
            # Initialize retriever if not already done
            if self._retriever is None:
                if not self.initialize_retriever():
                    return []

            # Prepare base arrays for potential filtering
            retriever_to_use = self._retriever
            search_chunks_data = self._chunks_data

            # Filter chunks by heading BEFORE BM25 search if provided
            if active_headings:
                try:
                    # Normalize headings to lowercase for matching
                    headings_lower = [h.lower() for h in active_headings if h]
                    titles = [
                        c.get("heading_title")
                        for c in self._chunks_data
                        if c.get("heading_title")
                    ]
                    # Deduplicate titles while preserving case
                    seen = set()
                    unique_titles = []
                    for t in titles:
                        tl = (t or "").lower()
                        if tl not in seen:
                            seen.add(tl)
                            unique_titles.append(t)

                    # Find matching titles (exact match with any of the active headings)
                    resolved_titles = []
                    for h_lower in headings_lower:
                        for t in unique_titles:
                            if t and t.lower() == h_lower:
                                resolved_titles.append(t)
                                break

                    if resolved_titles:
                        filtered_chunks = [
                            c
                            for c in self._chunks_data
                            if c.get("heading_title") in resolved_titles
                        ]
                        if filtered_chunks:
                            temp_retriever = self._build_retriever(filtered_chunks)
                            retriever_to_use = temp_retriever
                            search_chunks_data = filtered_chunks
                            self.logger.info(
                                f"Applied heading filter: {resolved_titles} -> {len(filtered_chunks)} chunks"
                            )
                        else:
                            self.logger.warning(
                                f"No chunks found for headings: {resolved_titles}"
                            )
                    else:
                        self.logger.warning(
                            f"No matching headings found for: {active_headings}"
                        )
                except Exception as _hf_err:
                    self.logger.warning(
                        f"Heading title resolution failed; proceeding without filter: {_hf_err}"
                    )

            # Entity/section filtering removed; use heading_title-only pre-filter when provided

            # Preprocess the query for better Vietnamese tokenization
            processed_query = self._preprocess_text_for_bm25(query)
            self.logger.info(
                f"Original query: '{query}' -> Processed query: '{processed_query}'"
            )

            # Set the number of results to retrieve
            if retriever_to_use is not None:
                retriever_to_use.k = limit

                # Perform BM25 search and get actual BM25 scores
                retrieved_docs = retriever_to_use.get_relevant_documents(
                    processed_query
                )

                # Calculate actual BM25 scores using the vectorizer
                if retrieved_docs and hasattr(retriever_to_use, "vectorizer"):
                    try:
                        # Get the BM25 vectorizer instance
                        bm25 = retriever_to_use.vectorizer

                        # Tokenize query for BM25 scoring (same as how documents were tokenized)
                        query_tokens = processed_query.split()

                        # Get BM25 scores for the query against all documents in filtered set
                        all_scores = bm25.get_scores(query_tokens)

                        # Create mapping of scores to retrieved documents
                        doc_scores = []
                        for doc in retrieved_docs:
                            metadata = doc.metadata
                            original_index = metadata.get("original_index", 0)

                            if original_index < len(all_scores):
                                score = all_scores[original_index]
                                doc_scores.append((doc, float(score)))
                            else:
                                # Fallback score if index out of bounds
                                rank_score = 1.0 - (len(doc_scores) * 0.05)
                                rank_score = max(rank_score, 0.1)
                                doc_scores.append((doc, rank_score))

                        # Sort by score descending to maintain ranking order
                        doc_scores.sort(key=lambda x: x[1], reverse=True)
                        retrieved_docs_with_scores = doc_scores[:limit]

                        self.logger.info(
                            f"BM25 calculated actual scores for {len(retrieved_docs_with_scores)} documents"
                        )
                        if retrieved_docs_with_scores:
                            max_score = max(
                                score for _, score in retrieved_docs_with_scores
                            )
                            min_score = min(
                                score for _, score in retrieved_docs_with_scores
                            )
                            self.logger.info(
                                f"BM25 score range: {min_score:.4f} - {max_score:.4f}"
                            )

                    except Exception as e:
                        # Fallback to rank-based scoring if BM25 scoring fails
                        self.logger.warning(
                            f"BM25 score calculation failed: {e}, using rank-based scoring"
                        )
                        retrieved_docs_with_scores = []
                        for i, doc in enumerate(retrieved_docs):
                            rank_score = 1.0 - (i * 0.05)  # Smaller decrease per rank
                            rank_score = max(rank_score, 0.1)
                            retrieved_docs_with_scores.append((doc, rank_score))
                else:
                    # Fallback to rank-based scoring if vectorizer not available
                    self.logger.warning(
                        "BM25 vectorizer not available, using rank-based scoring"
                    )
                    retrieved_docs_with_scores = []
                    for i, doc in enumerate(retrieved_docs):
                        rank_score = 1.0 - (i * 0.05)
                        rank_score = max(rank_score, 0.1)
                        retrieved_docs_with_scores.append((doc, rank_score))
            else:
                self.logger.error("BM25 retriever is not initialized")
                return []

            # Convert back to our format
            results = []
            for doc, score in retrieved_docs_with_scores:
                metadata = doc.metadata
                original_index = metadata.get("original_index", 0)

                # Get the original chunk data from the correct chunks array
                if original_index < len(search_chunks_data):
                    chunk_data = search_chunks_data[original_index].copy()

                    # Use original content for display instead of processed content
                    if "original_content" in metadata:
                        chunk_data["content"] = metadata["original_content"]

                    # Use actual or calculated BM25 score
                    chunk_data["score"] = float(score)
                    results.append(chunk_data)

            self.logger.info(
                f"BM25 results: {len(results)}/{len(search_chunks_data)} (after optional heading title filter)"
            )
            return results

        except Exception as e:
            self.logger.error(f"Error in BM25 search: {e}")
            return []
        """
        Refresh the BM25 index with latest data

        Returns:
            True if successful, False otherwise
        """
        return self.initialize_retriever(force_rebuild=True)

    def update_tokenized_content_for_all_chunks(self) -> bool:
        """
        Update tokenized_content for all existing chunks in database.
        This is a one-time operation to populate the new column.

        Returns:
            True if successful, False otherwise
        """
        try:
            from flask import current_app

            # Check if we're in an app context
            try:
                # Use hasattr to check for app context instead of _get_current_object
                hasattr(current_app, "app_context")
                in_app_context = True
            except RuntimeError:
                in_app_context = False

            if not in_app_context:
                self.logger.warning(
                    "Not in Flask app context, cannot update tokenized content"
                )
                return False

            # Get all chunks that don't have tokenized_content yet
            chunks_without_tokenized = (
                db.session.query(DocumentChunk)
                .join(Document)
                .filter(
                    Document.status == "completed",
                    DocumentChunk.tokenized_content.is_(None),
                )
                .all()
            )

            if not chunks_without_tokenized:
                self.logger.info("All chunks already have tokenized content")
                return True

            self.logger.info(
                f"Updating tokenized content for {len(chunks_without_tokenized)} chunks"
            )

            updated_count = 0
            for chunk in chunks_without_tokenized:
                try:
                    # Tokenize the content - access the actual string value
                    content_value = getattr(chunk, "content", "") or ""
                    tokenized = self._preprocess_text_for_bm25(content_value)
                    setattr(chunk, "tokenized_content", tokenized)
                    updated_count += 1

                    # Commit every 100 chunks to avoid memory issues
                    if updated_count % 100 == 0:
                        db.session.commit()
                        self.logger.info(f"Updated {updated_count} chunks so far...")

                except Exception as e:
                    self.logger.error(f"Error tokenizing chunk {chunk.id}: {e}")
                    continue

            # Final commit
            db.session.commit()
            self.logger.info(
                f"Successfully updated tokenized content for {updated_count} chunks"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error updating tokenized content: {e}")
            db.session.rollback()
            return False
