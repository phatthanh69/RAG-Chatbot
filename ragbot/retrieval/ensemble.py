"""
Ensemble Retriever service that combines BM25 and vector search
Uses weighted scoring to merge results from both approaches
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from ragbot.models.document import DocumentChunk
from ragbot.retrieval.bm25 import BM25Service
from ragbot.retrieval.vector_search import VectorSearchService


class EnsembleRetrieverService:
    """Service class that combines BM25 and vector search using ensemble approach"""

    def __init__(
        self,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        enable_rrf: bool = True,
        rrf_k: int = 60,
    ):
        """
        Initialize ensemble retriever

        Args:
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            vector_weight: Weight for vector search scores (0.0 to 1.0)
            enable_rrf: Enable Reciprocal Rank Fusion for score combination
            rrf_k: RRF parameter (typically 60)
        """
        self.logger = logging.getLogger(__name__)

        # Normalize weights
        total_weight = bm25_weight + vector_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.vector_weight = vector_weight / total_weight
        else:
            self.bm25_weight = 0.3
            self.vector_weight = 0.7

        self.enable_rrf = enable_rrf
        self.rrf_k = rrf_k

        # Initialize services
        self.bm25_service = BM25Service()
        self.vector_service = VectorSearchService()

    def initialize(self) -> bool:
        """
        Initialize the ensemble retriever components

        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize BM25 service
            if not self.bm25_service.initialize_retriever():
                self.logger.warning("BM25 service initialization failed")
                return False

            self.logger.info("Ensemble retriever initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing ensemble retriever: {e}")
            return False

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to 0-1 range using min-max normalization

        Args:
            results: List of results with scores

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [result["score"] for result in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score - min_score == 0:
            for result in results:
                result["normalized_score"] = 1.0
            return results

        # Normalize scores
        for result in results:
            normalized = (result["score"] - min_score) / (max_score - min_score)
            result["normalized_score"] = normalized

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            k: RRF parameter (typically 60)

        Returns:
            Combined and ranked results
        """
        self.logger.info(
            f"RRF fusion: BM25={len(bm25_results)}, Vector={len(vector_results)}, k={k}"
        )

        # Create a dictionary to accumulate RRF scores
        rrf_scores = {}
        all_results = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            if result is None:
                continue
            doc_id = result.get("id")
            if doc_id is not None:
                rrf_score = 1.0 / (k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
                all_results[doc_id] = result.copy()
                all_results[doc_id]["bm25_rank"] = rank + 1
                all_results[doc_id]["bm25_score"] = result.get("score", 0.0)

        self.logger.info(f"RRF: Processed {len(all_results)} BM25 results")

        # Process vector results
        for rank, result in enumerate(vector_results):
            if result is None:
                continue
            doc_id = result.get("id")
            if doc_id is not None:
                rrf_score = 1.0 / (k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score

                if doc_id in all_results:
                    # Update existing result
                    all_results[doc_id]["vector_rank"] = rank + 1
                    all_results[doc_id]["vector_score"] = result.get("score", 0.0)
                else:
                    # Add new result
                    all_results[doc_id] = result.copy()
                    all_results[doc_id]["vector_rank"] = rank + 1
                    all_results[doc_id]["vector_score"] = result.get("score", 0.0)
                    all_results[doc_id]["bm25_rank"] = None
                    all_results[doc_id]["bm25_score"] = 0.0

        # Sort by RRF score and create final results
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        self.logger.info(f"RRF: Final sorted results count: {len(sorted_items)}")

        final_results = []
        for doc_id, rrf_score in sorted_items:
            result = all_results[doc_id].copy()
            result["score"] = rrf_score
            result["rrf_score"] = rrf_score
            final_results.append(result)

        self.logger.info(f"RRF: Returning {len(final_results)} final results")
        return final_results

    def _weighted_score_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine results using weighted score fusion

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search

        Returns:
            Combined and ranked results
        """
        # Normalize scores
        bm25_normalized = self._normalize_scores(bm25_results.copy())
        vector_normalized = self._normalize_scores(vector_results.copy())

        # Create dictionaries for efficient lookup
        bm25_dict = {result["id"]: result for result in bm25_normalized}
        vector_dict = {result["id"]: result for result in vector_normalized}

        # Get all unique document IDs
        all_doc_ids = set(bm25_dict.keys()) | set(vector_dict.keys())

        # Combine scores
        combined_results = []
        for doc_id in all_doc_ids:
            bm25_result = bm25_dict.get(doc_id)
            vector_result = vector_dict.get(doc_id)

            # Determine which result to use as base
            if bm25_result and vector_result:
                # Both results available - use vector result as base (usually more complete)
                result = vector_result.copy()
                bm25_score = bm25_result.get("normalized_score", 0.0)
                vector_score = vector_result.get("normalized_score", 0.0)
            elif bm25_result:
                # Only BM25 result
                result = bm25_result.copy()
                bm25_score = bm25_result.get("normalized_score", 0.0)
                vector_score = 0.0
            else:
                # Only vector result
                result = vector_result.copy() if vector_result else {}
                bm25_score = 0.0
                vector_score = (
                    vector_result.get("normalized_score", 0.0) if vector_result else 0.0
                )

            # Calculate combined score
            combined_score = (
                bm25_score * self.bm25_weight + vector_score * self.vector_weight
            )

            result["score"] = combined_score
            result["bm25_score"] = bm25_score
            result["vector_score"] = vector_score
            result["combined_score"] = combined_score

            combined_results.append(result)

        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        return combined_results

    def search(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        limit: int = 10,
        min_score: float = 0.0,
        search_multiplier: int = 2,
        expand_section_threshold: int = 3,
        active_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform ensemble search combining BM25 and vector search

        Args:
            query: Search query text
            embedding: Pre-computed query embedding (optional)
            limit: Maximum number of results to return
            min_score: Minimum score threshold
            search_multiplier: Multiplier for initial search to get more candidates
            expand_section_threshold: Minimum chunks from same section to trigger expansion
            active_headings: Filter results by specific headings (optional)

        Returns:
            Combined and ranked search results
        """
        try:
            # Calculate search limit for individual retrievers
            # Increased minimum from 20 to 50 for better RRF fusion quality
            individual_limit = max(limit * search_multiplier, 50)

            self.logger.info(
                f"🔧 Ensemble search config: final_limit={limit}, multiplier={search_multiplier}, individual_limit={individual_limit}"
            )

            # Perform BM25 search
            self.logger.debug(f"Performing BM25 search with query: '{query}'")
            bm25_results = self.bm25_service.search_chunks(
                query=query,
                limit=individual_limit,
                min_score=0.0,
                active_headings=active_headings,
            )
            self.logger.info(f"BM25 search returned {len(bm25_results)} results")

            # Perform vector search
            if embedding is None:
                # If no embedding provided, we need to get it from vector service
                # For now, we'll skip vector search if no embedding is provided
                self.logger.warning(
                    "No embedding provided for vector search, using only BM25 results"
                )
                vector_results = []
            else:
                self.logger.debug(
                    f"Performing vector search with embedding of length: {len(embedding)}"
                )
                vector_results = self.vector_service.search_similar_chunks(
                    embedding=embedding,
                    limit=individual_limit,
                    min_score=0.0,
                    active_headings=active_headings,
                )
                self.logger.info(
                    f"Vector search returned {len(vector_results)} results"
                )

            self.logger.info(
                f"Retrieved {len(bm25_results)} BM25 results and {len(vector_results)} vector results"
            )

            # Combine results
            if self.enable_rrf:
                self.logger.info(f"Using RRF fusion with k={self.rrf_k}")
                combined_results = self._reciprocal_rank_fusion(
                    bm25_results, vector_results, self.rrf_k
                )
            else:
                self.logger.info("Using weighted score fusion")
                combined_results = self._weighted_score_fusion(
                    bm25_results, vector_results
                )

            self.logger.info(f"Combined results count: {len(combined_results)}")

            # Apply minimum score threshold and limit
            self.logger.info(
                f"Score distribution: max={max([r['score'] for r in combined_results]) if combined_results else 0:.4f}, min={min([r['score'] for r in combined_results]) if combined_results else 0:.4f}"
            )

            filtered_results = [
                result for result in combined_results if result["score"] >= min_score
            ]

            self.logger.info(
                f"After min_score filter ({min_score}): {len(filtered_results)} results"
            )

            # Return top results
            final_results = filtered_results[:limit]

            self.logger.info(
                f"Ensemble search returned {len(final_results)} final results "
                f"(after filtering and limiting from {len(combined_results)} combined)"
            )

            # Ưu tiên mở rộng theo heading (level thấp nhất được ưu tiên) nếu phát hiện heading
            try:
                # Lấy document_ids từ results hiện tại để giới hạn phạm vi tìm kiếm
                document_ids: List[str] = [
                    str(result.get("document_id"))
                    for result in final_results
                    if result.get("document_id") is not None
                ]
                document_ids = list(set(document_ids))

                # If we have active_headings, expand by those headings first
                if active_headings:
                    expanded_by_active_headings = (
                        self._expand_results_by_active_headings(
                            final_results, active_headings, document_ids, query
                        )
                    )
                    if expanded_by_active_headings is not None:
                        self.logger.info(
                            f"Active headings expansion applied: {len(expanded_by_active_headings) - len(final_results)} chunks added"
                        )
                        final_results = expanded_by_active_headings
                    else:
                        # Fallback to automatic heading detection
                        expanded_by_heading = self._expand_results_by_heading(
                            final_results, document_ids
                        )
                        if expanded_by_heading is not None:
                            self.logger.info(
                                f"Heading-first expansion applied: {len(expanded_by_heading) - len(final_results)} chunks added"
                            )
                            final_results = expanded_by_heading
                else:
                    # No active_headings, use automatic heading detection
                    expanded_by_heading = self._expand_results_by_heading(
                        final_results, document_ids
                    )
                    if expanded_by_heading is not None:
                        self.logger.info(
                            f"Heading-first expansion applied: {len(expanded_by_heading) - len(final_results)} chunks added"
                        )
                        final_results = expanded_by_heading
                # Keep heading-first results as-is
            except Exception as _h_err:
                # Heading-first expansion failed; keep current results as-is.
                self.logger.warning(
                    f"Heading expansion failed softly; keeping current results: {_h_err}"
                )
                # If heading expansion fails, return current final_results without any fallback

            return final_results

        except Exception as e:
            self.logger.error(f"Error in ensemble search: {e}")
            return []

    def _expand_results_deprecated(
        self,
        results: List[Dict[str, Any]],
        entity_threshold: int = 3,
        section_threshold: int = 3,
        active_entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Entity/section expansion has been replaced by heading-based expansion.
        This method now returns results unchanged.
        """
        self.logger.debug(
            "Entity/section expansion called but is deprecated, returning unchanged results"
        )
        return results

    def _format_chunk_result(self, chunk: DocumentChunk) -> Optional[Dict[str, Any]]:
        """
        Format chunk result để phù hợp với format của search results

        Args:
            chunk: DocumentChunk object

        Returns:
            Dict với format chuẩn
        """
        try:
            return {
                "id": chunk.id,
                "content": chunk.content,
                "score": getattr(chunk, "score", 0.0) or 0.0,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "page": chunk.page_number,
                "block_index": chunk.block_index,
                "bbox": chunk.bbox,
                "font_info": chunk.font_info,
                # heading metadata
                "heading_id": getattr(chunk, "heading_id", None),
                "heading_title": getattr(chunk, "heading_title", None),
                "heading_parent_id": getattr(chunk, "heading_parent_id", None),
                "heading_level": getattr(chunk, "heading_level", None),
                # entity and section removed - replaced by headings
                "entity": None,
                "section": None,
                "is_heading": chunk.is_heading,
                "clean_pdf_name": chunk.clean_pdf_name,
                "file_name": chunk.document.file_name,
                "original_file_name": chunk.document.original_file_name,
                "metadata": chunk.document.doc_metadata,
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi format chunk {chunk.id}: {e}")
            return None

    def _expand_results_by_heading(
        self,
        results: List[Dict[str, Any]],
        document_ids: List[str],
        top_k_scan: int = 10,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Mở rộng kết quả dựa trên heading_title/heading_level thay thế cho entity/section.

        Quy tắc:
        - Quét top-k kết quả đầu để tìm các chunks có is_heading=True và có heading_id/heading_title.
        - Ưu tiên heading có level nhỏ nhất (level càng thấp càng được ưu tiên). Nếu có nhiều heading cùng level,
          chọn heading có điểm cao nhất trong kết quả hiện tại.
        - Mở rộng subtree: lấy tất cả các heading con (dựa vào heading_parent_id) và
          toàn bộ chunks thuộc các heading_ids này.

        Trả về danh sách kết quả đã mở rộng; nếu không phát hiện heading hợp lệ, trả về None.
        """
        try:
            if not results:
                return None

            # Lấy top-k để phát hiện heading
            scan = results[: max(1, top_k_scan)]

            # Thu thập heading candidates: (level, -score, heading_id, title)
            candidates: List[Tuple[int, float, str, str]] = []
            for r in scan:
                is_heading = bool(r.get("is_heading"))
                hid = r.get("heading_id")
                htitle = r.get("heading_title")
                hlevel = r.get("heading_level")
                score = float(r.get("score", 0.0) or 0.0)
                if is_heading and hid and htitle and hlevel is not None:
                    try:
                        lvl = int(hlevel)
                    except Exception:
                        continue
                    # Ưu tiên level thấp hơn (level nhỏ hơn được ưu tiên), sort theo (level asc, -score desc)
                    candidates.append((lvl, -score, str(hid), str(htitle)))

            if not candidates:
                # No headings present in the top-k results
                return None

            # Chọn heading theo ưu tiên level thấp nhất; nếu có nhiều heading cùng level, ưu tiên score cao hơn
            candidates.sort(key=lambda t: (t[0], t[1]))
            best_level, _, root_heading_id, root_title = candidates[0]
            self.logger.info(
                f"Heading detected for expansion: '{root_title}' (id={root_heading_id}, level={best_level})"
            )

            # Xây map heading_id -> (parent_id, level, title) trong phạm vi document_ids
            heading_map = self._build_heading_map(document_ids)
            if not heading_map:
                self.logger.info("No heading map built (empty), skip heading expansion")
                return None

            # Thu thập toàn bộ subtree ids từ root_heading_id
            subtree_ids = self._collect_heading_subtree(root_heading_id, heading_map)
            if not subtree_ids:
                self.logger.info(
                    f"No subtree ids found for heading_id={root_heading_id}, skip heading expansion"
                )
                return None

            # Tránh duplicate
            seen_chunk_ids = {r.get("id") for r in results if r.get("id") is not None}

            # Query tất cả chunks thuộc các heading_ids này trong phạm vi document_ids
            try:
                q = DocumentChunk.query
                q = q.filter(DocumentChunk.heading_id.in_(list(subtree_ids)))
                if document_ids:
                    q = q.filter(
                        DocumentChunk.document_id.in_(
                            [int(d) for d in document_ids if str(d).isdigit()]
                        )
                    )
                expand_chunks = q.all()
            except Exception as _q_err:
                self.logger.warning(f"Heading expansion query failed: {_q_err}")
                return None

            added = 0
            for ch in expand_chunks:
                if ch.id in seen_chunk_ids:
                    continue
                ch_dict = self._format_chunk_result(ch)
                if ch_dict:
                    results.append(ch_dict)
                    seen_chunk_ids.add(ch.id)
                    added += 1

            self.logger.info(
                f"Heading expansion added {added} chunks for heading '{root_title}' (level {best_level})"
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in heading-first expansion: {e}")
            return None

    def _expand_results_by_active_headings(
        self,
        results: List[Dict[str, Any]],
        active_headings: List[str],
        document_ids: List[str],
        query: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Expand results by fetching all chunks under the active_headings.

        Args:
            results: Current search results
            active_headings: List of heading titles to expand by
            document_ids: Document IDs to limit search scope

        Returns:
            Expanded results or None if expansion fails
        """
        try:
            if not active_headings or not results:
                return None

            self.logger.info(
                f"Attempting heading expansion with active_headings: {active_headings}, document_ids: {document_ids}"
            )

            # Build heading map for the documents
            heading_map = self._build_heading_map(document_ids)
            if not heading_map:
                self.logger.warning(
                    "Cannot build heading map for active headings expansion"
                )
                return None

            # Collect all subtree IDs for all active headings
            all_subtree_ids = set()
            for heading_title in active_headings:
                # Find ALL heading_ids for this title (since heading_id is not unique)
                matching_heading_ids = []
                for hid, (parent_id, level, title) in heading_map.items():
                    if title == heading_title:
                        matching_heading_ids.append(hid)

                if matching_heading_ids:
                    self.logger.info(
                        f"Found heading '{heading_title}' with ids={matching_heading_ids}"
                    )
                    # Collect subtrees for all matching heading_ids
                    for heading_id in matching_heading_ids:
                        subtree_ids = self._collect_heading_subtree(
                            heading_id, heading_map
                        )
                        all_subtree_ids.update(subtree_ids)
                else:
                    self.logger.warning(
                        f"Heading '{heading_title}' not found in heading_map, available titles: {list(set(title for _, _, title in heading_map.values() if title))}"
                    )

            if not all_subtree_ids:
                self.logger.warning("No subtree IDs found for active headings")
                return None

            # Avoid duplicates
            seen_chunk_ids = {r.get("id") for r in results if r.get("id") is not None}

            # Query all chunks under these heading IDs
            try:
                q = DocumentChunk.query
                q = q.filter(DocumentChunk.heading_id.in_(list(all_subtree_ids)))
                if document_ids:
                    q = q.filter(
                        DocumentChunk.document_id.in_(
                            [int(d) for d in document_ids if str(d).isdigit()]
                        )
                    )
                expand_chunks = q.all()
            except Exception as _q_err:
                self.logger.warning(f"Active headings expansion query failed: {_q_err}")
                return None

            added = 0
            for ch in expand_chunks:
                if ch.id in seen_chunk_ids:
                    continue
                ch_dict = self._format_chunk_result(ch)
                if ch_dict:
                    results.append(ch_dict)
                    seen_chunk_ids.add(ch.id)
                    added += 1

            self.logger.info(
                f"Heading expansion added {added} chunks for active_headings: {active_headings}"
            )

            # Phase 2: Intent-based filtering
            if results:
                results = self._apply_phase2_filtering(results, active_headings, query)

            return results

        except Exception as e:
            self.logger.error(f"Error in active headings expansion: {e}")
            return None

    def _apply_phase2_filtering(
        self,
        results: List[Dict[str, Any]],
        active_headings: List[str],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Smart filtering để tối ưu chi phí LLM bằng cách chỉ lấy chunks thực sự cần thiết.

        Logic:
        - Phát hiện intent của query (comparison, specific info, general info)
        - Lọc chunks theo intent để giảm noise và tối ưu chi phí
        - Ưu tiên chunks có heading relevance cao với query intent

        Args:
            results: Current results after expansion
            active_headings: The active headings that were expanded
            query: The original search query

        Returns:
            Smart filtered results
        """
        try:
            if not results or not query:
                return results

            original_count = len(results)

            # 1. Phát hiện intent của query
            query_intent = self._detect_query_intent(query)

            # 2. Áp dụng filtering theo intent
            filtered_results = self._filter_by_intent(
                results, query_intent, active_headings, query
            )

            self.logger.info(
                f"Phase 2: Intent '{query_intent['type']}' filtering: {original_count} → {len(filtered_results)} chunks "
                f"(saved {original_count - len(filtered_results)} chunks, {((original_count - len(filtered_results)) / original_count * 100):.1f}% reduction)"
            )

            return filtered_results

        except Exception as e:
            self.logger.error(f"Error in phase 2 filtering: {e}")
            return results

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Phát hiện intent của query để áp dụng filtering phù hợp.

        Args:
            query: Search query

        Returns:
            Dict chứa thông tin intent: {
                'type': 'comparison'|'specific_info'|'general_listing'|'technical_specs',
                'entities': [list of entities detected],
                'focus_keywords': [list of focus keywords]
            }
        """
        try:
            query_lower = query.lower().strip()

            # 1. Comparison queries
            comparison_keywords = [
                "so sánh",
                "compare",
                "khác biệt",
                "difference",
                "versus",
                "vs",
                "giữa",
                "between",
                "và",
                "and",
                "với",
                "against",
            ]

            # 2. Technical specs queries
            tech_spec_keywords = [
                "thông số",
                "specification",
                "specs",
                "kỹ thuật",
                "technical",
                "tính năng",
                "feature",
                "đặc điểm",
                "characteristic",
                "parameter",
            ]

            # 3. General listing queries
            listing_keywords = [
                "những loại",
                "các loại",
                "list",
                "danh sách",
                "all types",
                "types of",
                "kinds of",
                "loại nào",
                "có gì",
                "bao gồm",
            ]

            # Extract entities (product codes like LS-BE-001, LDM301)
            import re

            entities = re.findall(
                r"[A-Z]{2,}[-_]?[A-Z0-9]{2,}[-_]?[A-Z0-9]*", query.upper()
            )

            # Detect intent type
            is_comparison = (
                any(kw in query_lower for kw in comparison_keywords)
                and len(entities) >= 2
            )
            is_tech_specs = any(kw in query_lower for kw in tech_spec_keywords)
            is_listing = any(kw in query_lower for kw in listing_keywords)

            if is_comparison:
                intent_type = "comparison"
            elif is_tech_specs:
                intent_type = "technical_specs"
            elif is_listing:
                intent_type = "general_listing"
            else:
                intent_type = "specific_info"

            # Extract focus keywords
            focus_keywords = []
            if is_tech_specs:
                focus_keywords.extend(
                    ["thông số", "specification", "kỹ thuật", "tính năng"]
                )
            if is_comparison:
                focus_keywords.extend(["so sánh", "compare", "khác biệt"])

            result = {
                "type": intent_type,
                "entities": entities,
                "focus_keywords": focus_keywords,
                "is_multi_entity": len(entities) > 1,
            }

            self.logger.debug(f"Query intent detection: '{query}' → {result}")

            return result

        except Exception as e:
            self.logger.error(f"Error in query intent detection: {e}")
            return {
                "type": "specific_info",
                "entities": [],
                "focus_keywords": [],
                "is_multi_entity": False,
            }

    def _fuzzy_match_score(self, text1: str, text2: str) -> float:
        """
        Tính điểm fuzzy matching giữa 2 chuỗi sử dụng RapidFuzz.

        Returns:
            Float từ 0.0 - 1.0, cao hơn là giống nhau hơn
        """
        try:
            if not text1 or not text2:
                return 0.0

            text1_clean = text1.lower().strip()
            text2_clean = text2.lower().strip()

            # Sử dụng RapidFuzz (nhanh và chính xác)
            return fuzz.ratio(text1_clean, text2_clean) / 100.0
        except Exception:
            return 0.0

    def _find_child_headings_with_focus_match(
        self,
        active_headings: List[str],
        focus_keywords: List[str],
        document_ids: List[str],
        min_fuzzy_score: float = 0.7,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Tìm các heading con của active_headings và match với focus_keywords sử dụng RapidFuzz.

        Args:
            active_headings: Các heading gốc đã được filter ở Phase 1
            focus_keywords: Keywords cần match (ví dụ: "thông số", "so sánh")
            document_ids: Giới hạn tìm kiếm
            min_fuzzy_score: Điểm tối thiểu cho fuzzy matching

        Returns:
            Dict[heading_title -> List[matched_children_info]]
        """
        try:
            if not active_headings or not focus_keywords:
                return {}

            # Xây dựng heading map
            heading_map = self._build_heading_map(document_ids)
            if not heading_map:
                return {}

            results = {}

            for parent_heading in active_headings:
                # Tìm tất cả heading_ids của parent heading
                parent_heading_ids = []
                for hid, (pid, level, title) in heading_map.items():
                    if title == parent_heading:
                        parent_heading_ids.append(hid)

                if not parent_heading_ids:
                    continue

                # Tìm tất cả heading con
                child_headings = []
                for parent_id in parent_heading_ids:
                    subtree_ids = self._collect_heading_subtree(parent_id, heading_map)

                    for child_id in subtree_ids:
                        if child_id == parent_id:  # Bỏ qua chính nó
                            continue

                        child_info = heading_map.get(child_id)
                        if child_info:
                            _, child_level, child_title = child_info
                            if child_title:  # Chỉ lấy các heading có title
                                child_headings.append(
                                    {
                                        "id": child_id,
                                        "title": child_title,
                                        "level": child_level,
                                    }
                                )

                # So sánh các heading con với focus keywords
                matched_children = []
                for child in child_headings:
                    child_title = child["title"]
                    best_match_score = 0.0
                    best_keyword = ""

                    for keyword in focus_keywords:
                        # Exact match (cao nhất)
                        if keyword.lower() in child_title.lower():
                            score = 1.0
                        else:
                            # Fuzzy match
                            score = self._fuzzy_match_score(keyword, child_title)

                        if score > best_match_score:
                            best_match_score = score
                            best_keyword = keyword

                    # Chỉ lấy các match tốt
                    if best_match_score >= min_fuzzy_score:
                        matched_children.append(
                            {
                                "heading_id": child["id"],
                                "heading_title": child["title"],
                                "heading_level": child["level"],
                                "match_score": best_match_score,
                                "matched_keyword": best_keyword,
                            }
                        )

                # Sắp xếp theo điểm match
                matched_children.sort(key=lambda x: x["match_score"], reverse=True)
                results[parent_heading] = matched_children

            return results

        except Exception as e:
            self.logger.error(f"Error finding child headings with focus match: {e}")
            return {}

    def _filter_by_intent(
        self,
        results: List[Dict[str, Any]],
        intent: Dict[str, Any],
        active_headings: List[str],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Lọc chunks dựa trên intent và heading con của active headings để tối ưu chi phí LLM.

        Logic mới:
        - Vì Phase 1 đã filter theo active_headings
        - Phase 2 chỉ cần so sánh focus_keywords với heading con sử dụng RapidFuzz
        - ƪu tiên chunks thuộc heading con match với focus keywords

        Args:
            results: Danh sách chunks sau expansion (chỉ từ active_headings)
            intent: Intent info từ _detect_query_intent
            active_headings: Active headings đã filter ở Phase 1
            query: Original query

        Returns:
            Filtered chunks
        """
        try:
            if not results or not active_headings:
                return results

            intent_type = intent.get("type", "specific_info")
            entities = intent.get("entities", [])
            focus_keywords = intent.get("focus_keywords", [])

            self.logger.debug(
                f"Phase 2 filtering: {intent_type}, active_headings: {active_headings}, focus: {focus_keywords}"
            )

            # Nếu không có focus keywords, giữ tất cả
            if not focus_keywords:
                return self._apply_basic_filtering(results, intent_type)

            # Lấy document_ids từ results
            document_ids = list(
                set(
                    str(chunk.get("document_id"))
                    for chunk in results
                    if chunk.get("document_id")
                )
            )

            # Tìm heading con match với focus keywords
            matched_children = self._find_child_headings_with_focus_match(
                active_headings, focus_keywords, document_ids
            )

            if not matched_children:
                # No matching child heading found -> keep basic filtering
                return self._apply_basic_filtering(results, intent_type)

            # Áp dụng filtering dựa trên matched heading children
            filtered = self._filter_by_matched_headings(
                results, matched_children, intent_type
            )

            # Fallback protection
            if len(filtered) < 3 and len(results) > 3:
                self.logger.warning(
                    f"Heading-based filter too aggressive ({len(filtered)} chunks), applying fallback"
                )
                fallback = self._apply_basic_filtering(results, intent_type)
                filtered = fallback[: max(len(filtered), 8)]  # Lấy ít nhất 8 chunks

            return filtered

        except Exception as e:
            self.logger.error(f"Error in Phase 2 intent filtering: {e}")
            return results

    def _apply_basic_filtering(
        self, results: List[Dict[str, Any]], intent_type: str
    ) -> List[Dict[str, Any]]:
        """
        Áp dụng basic filtering khi không có focus keywords hoặc không tìm thấy heading con phù hợp.
        """
        try:
            # Sắp xếp theo score
            sorted_results = sorted(
                results, key=lambda x: x.get("score", 0), reverse=True
            )

            # Adaptive limit dựa trên intent và số lượng
            if intent_type == "comparison":
                limit = min(15, len(results))
            elif intent_type == "technical_specs":
                limit = min(12, len(results))
            elif intent_type == "general_listing":
                limit = len(results)  # Giữ tất cả
            else:
                # specific_info
                if len(results) > 30:
                    limit = 10
                elif len(results) > 15:
                    limit = 12
                else:
                    limit = len(results)

            self.logger.info(
                f"Basic filtering: {len(results)} → {limit} chunks (intent: {intent_type})"
            )
            return sorted_results[:limit]

        except Exception as e:
            self.logger.error(f"Error in basic filtering: {e}")
            return results[:10]

    def _filter_by_matched_headings(
        self,
        results: List[Dict[str, Any]],
        matched_children: Dict[str, List[Dict[str, Any]]],
        intent_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Lọc chunks dựa trên các heading con đã match với focus keywords.

        Args:
            results: Tất cả chunks
            matched_children: Dict[parent_heading -> matched_child_headings]
            intent_type: Loại intent
        """
        try:
            # Thu thập tất cả heading_ids đã match
            matched_heading_ids = set()
            for parent, children in matched_children.items():
                for child in children:
                    matched_heading_ids.add(child["heading_id"])

            # Phân loại chunks
            high_priority = []  # Chunks thuộc matched headings
            medium_priority = []  # Chunks thuộc parent headings nhưng không match
            low_priority = []  # Chunks khác

            for chunk in results:
                chunk_heading_id = chunk.get("heading_id")

                if chunk_heading_id and str(chunk_heading_id) in matched_heading_ids:
                    high_priority.append(chunk)
                elif chunk.get("heading_title") in matched_children.keys():
                    medium_priority.append(chunk)
                else:
                    low_priority.append(chunk)

            # Sắp xếp mỗi nhóm theo score
            high_priority.sort(key=lambda x: x.get("score", 0), reverse=True)
            medium_priority.sort(key=lambda x: x.get("score", 0), reverse=True)
            low_priority.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Kết hợp với tỷ lệ phù hợp
            filtered = []

            # ƪu tiên lấy tất cả high priority
            filtered.extend(high_priority)

            # Sau đó lấy một số medium priority
            remaining_slots = max(0, 15 - len(filtered))
            if remaining_slots > 0:
                filtered.extend(medium_priority[: remaining_slots // 2])

            # Cuối cùng lấy một ít low priority nếu còn chỗ
            remaining_slots = max(0, 15 - len(filtered))
            if remaining_slots > 0 and intent_type != "comparison":
                filtered.extend(low_priority[:remaining_slots])

            # Ensure at least 3 chunks
            if len(filtered) < 3:
                all_chunks = high_priority + medium_priority + low_priority
                filtered = all_chunks[: max(3, min(8, len(all_chunks)))]

            self.logger.info(
                f"Heading-based filtering: {len(results)} → {len(filtered)} chunks "
                f"(high: {len(high_priority)}, medium: {len(medium_priority)}, low: {len(low_priority)})"
            )

            return filtered

        except Exception as e:
            self.logger.error(f"Error in matched headings filtering: {e}")
            return results[:10]

    def _build_heading_map(
        self, document_ids: List[str]
    ) -> Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]]:
        """
        Tạo map heading_id -> (parent_id, level, title) trong phạm vi các documents cho nhanh.
        """
        try:
            q = (
                DocumentChunk.query.with_entities(
                    DocumentChunk.heading_id,
                    DocumentChunk.heading_parent_id,
                    DocumentChunk.heading_level,
                    DocumentChunk.heading_title,
                )
                .filter(DocumentChunk.heading_id.isnot(None))
                .distinct()
            )
            if document_ids:
                q = q.filter(
                    DocumentChunk.document_id.in_(
                        [int(d) for d in document_ids if str(d).isdigit()]
                    )
                )
            rows = q.all()

            mapping: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}
            for hid, pid, lvl, title in rows:
                if not hid:
                    continue
                mapping[str(hid)] = (
                    pid if pid else None,
                    int(lvl) if lvl is not None else None,
                    title,
                )
            return mapping
        except Exception as e:
            self.logger.error(f"Failed building heading map: {e}")
            return {}

    def _collect_heading_subtree(
        self,
        root_heading_id: str,
        heading_map: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]],
    ) -> set:
        """
        Thu thập tất cả heading_ids trong subtree (bao gồm root) dựa trên parent_id links.
        """
        try:
            root = str(root_heading_id)
            if root not in heading_map:
                # Nếu root chưa có trong map (ví dụ top result từ file khác), vẫn thêm root
                return {root}

            children_map: Dict[str, List[str]] = {}
            for hid, (pid, _lvl, _title) in heading_map.items():
                if not pid:
                    continue
                parent = str(pid)
                children_map.setdefault(parent, []).append(hid)

            # BFS
            result_ids = {root}
            frontier = [root]
            while frontier:
                current = frontier.pop(0)
                for child in children_map.get(current, []):
                    if child not in result_ids:
                        result_ids.add(child)
                        frontier.append(child)
            return result_ids
        except Exception:
            return {str(root_heading_id)}

    def refresh_indices(self) -> bool:
        """
        Refresh both BM25 and vector indices

        Returns:
            True if successful, False otherwise
        """
        try:
            # Refresh BM25 index (use initialize_retriever to rebuild)
            if not self.bm25_service.initialize_retriever(force_rebuild=True):
                self.logger.warning("Failed to refresh/reinitialize BM25 index")
                return False

            self.logger.info("Ensemble retriever indices refreshed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error refreshing ensemble retriever indices: {e}")
            return False

    def update_weights(self, bm25_weight: float, vector_weight: float) -> None:
        """
        Update the weights for BM25 and vector search

        Args:
            bm25_weight: New weight for BM25 scores
            vector_weight: New weight for vector search scores
        """
        total_weight = bm25_weight + vector_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.vector_weight = vector_weight / total_weight

            self.logger.info(
                f"Updated weights - BM25: {self.bm25_weight:.2f}, Vector: {self.vector_weight:.2f}"
            )
        else:
            self.logger.warning("Invalid weights provided, keeping current weights")
