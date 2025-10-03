"""
Vector search service for handling vector similarity search operations
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.models.base import db
from app.models.document import Document, DocumentChunk


class VectorSearchService:
    """Service class for vector search operations"""

    logger = logging.getLogger(__name__)

    @staticmethod
    def search_similar_chunks(
        embedding: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        active_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks using vector similarity

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            active_entity: Filter results by specific entity (optional)

        Returns:
            List of similar chunks with metadata
        """
        try:
            # Convert embedding to numpy array for validation
            query_embedding = np.array(embedding, dtype=np.float32)

            # Ensure embedding has correct dimensions
            if len(query_embedding) != 1536:
                raise ValueError(
                    f"Embedding must have 1536 dimensions, got {len(query_embedding)}"
                )

            # Build base query
            query = (
                db.session.query(
                    DocumentChunk,
                    (
                        1
                        - (DocumentChunk.embedding.cosine_distance(query_embedding) / 2)
                    ).label("similarity"),
                )
                .join(Document)
                .filter(
                    Document.status == "completed",
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                    <= (2 * (1 - min_score)),
                )
            )

            # Heading-based filtering (exact match for list)
            if active_headings:
                matched_titles = []
                try:
                    # Get all distinct non-empty headings
                    all_headings = (
                        db.session.query(DocumentChunk.heading_title)
                        .join(Document)
                        .filter(
                            Document.status == "completed",
                            DocumentChunk.heading_title.isnot(None),
                            DocumentChunk.heading_title != "",
                        )
                        .distinct()
                        .all()
                    )
                    available_titles = [h[0] for h in all_headings if h and h[0]]

                    # Exact match for each heading in the list
                    for heading in active_headings:
                        heading_lower = heading.lower()
                        for t in available_titles:
                            if t.lower() == heading_lower:
                                matched_titles.append(t)
                                break

                    if matched_titles:
                        query = query.filter(
                            DocumentChunk.heading_title.in_(matched_titles)
                        )
                    else:
                        pass  # No matching headings, continue without filter
                except Exception as _hf_err:
                    VectorSearchService.logger.warning(
                        "⚠️ Heading title resolution failed, skipping filter: %s",
                        _hf_err,
                    )

            # Entity/section filtering removed; rely on heading_title-only when provided

            # Execute query with ordering and limit
            results = (
                query.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
                .limit(limit)
                .all()
            )

            # Format results
            formatted_results = []
            for chunk, similarity in results:
                formatted_results.append(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
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
                        "score": float(similarity),
                        "file_name": chunk.document.file_name,
                        "original_file_name": chunk.document.original_file_name,
                        "metadata": chunk.document.doc_metadata,
                    }
                )

            return formatted_results

        except Exception:
            VectorSearchService.logger.exception("❌ Error in vector search")
            return []

    @staticmethod
    def get_chunk_by_id(chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID"""
        try:
            chunk = DocumentChunk.query.get(chunk_id)
            if chunk:
                return {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page": chunk.page_number,
                    "block_index": chunk.block_index,
                    "bbox": chunk.bbox,
                    "font_info": chunk.font_info,
                    "is_heading": chunk.is_heading,
                    "clean_pdf_name": chunk.clean_pdf_name,
                    "file_name": chunk.document.file_name,
                    "original_file_name": chunk.document.original_file_name,
                    "metadata": chunk.document.doc_metadata,
                }
            return None
        except Exception as e:
            VectorSearchService.logger.error("❌ Error getting chunk by ID: %s", e)
            return None

    @staticmethod
    def get_chunks_by_document(document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            chunks = (
                DocumentChunk.query.filter_by(document_id=document_id)
                .order_by(DocumentChunk.chunk_index.asc())
                .all()
            )

            return [VectorSearchService._format_chunk_result(chunk) for chunk in chunks]
        except Exception as e:
            VectorSearchService.logger.error(
                "❌ Error getting chunks by document: %s",
                e,
            )
            return []

    @staticmethod
    def get_similar_chunks_by_id(chunk_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Find chunks similar to a specific chunk"""
        try:
            chunk = DocumentChunk.query.get(chunk_id)
            if not chunk or not chunk.embedding:
                return []

            return VectorSearchService.search_similar_chunks(
                embedding=chunk.embedding,
                limit=limit + 1,  # +1 to exclude the chunk itself
            )[
                1:
            ]  # Skip the first result (itself)
        except Exception as e:
            VectorSearchService.logger.error(
                "❌ Error getting similar chunks: %s",
                e,
            )
            return []

    @staticmethod
    def _format_chunk_result(chunk: DocumentChunk) -> Dict[str, Any]:
        """Helper method to format chunk results consistently"""
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "page": chunk.page_number,
            "block_index": chunk.block_index,
            "bbox": chunk.bbox,
            "font_info": chunk.font_info,
            "is_heading": chunk.is_heading,
            "clean_pdf_name": chunk.clean_pdf_name,
            "file_name": chunk.document.file_name,
            "original_file_name": chunk.document.original_file_name,
            "metadata": chunk.document.doc_metadata,
        }

    @staticmethod
    def search_qa_chunks(
        embedding: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        qa_source_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for Q&A chunks using vector similarity

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            qa_source_name: Name of Q&A source (e.g., 'Bo_cau_hoi_BAS_embeddings')

        Returns:
            List of Q&A chunks with metadata
        """
        try:
            # Convert embedding to numpy array for validation
            query_embedding = np.array(embedding, dtype=np.float32)

            # Ensure embedding has correct dimensions
            if len(query_embedding) != 1536:
                raise ValueError(
                    f"Embedding must have 1536 dimensions, got {len(query_embedding)}"
                )

            # Build query
            query = db.session.query(
                DocumentChunk,
                (
                    1 - (DocumentChunk.embedding.cosine_distance(query_embedding) / 2)
                ).label("similarity"),
            ).join(Document)

            # Filter for completed documents
            query = query.filter(Document.status == "completed")

            # Filter for Q&A data if source name provided
            if qa_source_name:
                query = query.filter(DocumentChunk.clean_pdf_name == qa_source_name)

            # Apply similarity threshold and ordering
            results = (
                query.filter(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                    <= (2 * (1 - min_score))
                )
                .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
                .limit(limit)
                .all()
            )

            # Format results
            formatted_results = []
            for chunk, similarity in results:
                formatted_results.append(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "page": chunk.page_number,
                        "block_index": chunk.block_index,
                        "bbox": chunk.bbox,
                        "font_info": chunk.font_info,
                        "is_heading": chunk.is_heading,
                        "clean_pdf_name": chunk.clean_pdf_name,
                        "score": float(similarity),
                        "file_name": chunk.document.file_name,
                        "original_file_name": chunk.document.original_file_name,
                        "metadata": chunk.document.doc_metadata,
                    }
                )

            return formatted_results

        except Exception as e:
            VectorSearchService.logger.error("❌ Error in Q&A vector search: %s", e)
            return []
