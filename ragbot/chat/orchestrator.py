"""
Chatbot service
Handles chatbot operations and session management
Now integrated with database storage and vector search
"""

import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix import path to avoid circular imports
try:
    # First try to import from the project root
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ragbot.chat.rag_engine import (
        EnhancedRAGChatbot,
        RetrievalResult,
        generate_answer,
        retrieve,
    )
    from ragbot.llm.client import init_genai_client

except ImportError as e:
    # Fallback to sys.path manipulation
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ragbot.chat.rag_engine import (
        retrieve,
        generate_answer,
        RetrievalResult,
        EnhancedRAGChatbot,
    )
    from ragbot.llm.client import init_genai_client

from ragbot.db.database_service import DatabaseService
from ragbot.retrieval.ensemble import EnsembleRetrieverService
from ragbot.retrieval.vector_search import VectorSearchService
from ragbot.chat.classification import QuestionClassifier
from ragbot.chat.session import SessionManager
from ragbot.chat.prompt_builder import build_prompt, get_recent_context
from ragbot.utils.text import dedupe_preserve_order as _dedupe
from ragbot.utils.text import normalize_text as _normalize_text


class ChatbotService:
    """Service for handling chatbot operations"""

    @staticmethod
    def normalize_text(s: str) -> str:
        return _normalize_text(s)

    @staticmethod
    def dedupe_preserve_order(items):
        return _dedupe(items)

    # === Logging Utility Methods ===
    def _log_process_step(self, step: str, details: str = "", session_id: str = ""):
        """Log a processing step with consistent format"""
        session_info = f" [Session: {session_id}]" if session_id else ""
        self.logger.info(
            f"{step}{session_info} - {details}" if details else f"{step}{session_info}"
        )

    def _log_search_result(self, search_type: str, count: int, details: str = ""):
        """Log search results with consistent format"""
        self.logger.info(
            f"{search_type}: {count} results" + (f" - {details}" if details else "")
        )

    def _log_question_processing(
        self, original: str, rewritten: Optional[str] = None, session_id: str = ""
    ):
        """Log question processing with truncated content"""
        question_preview = original[:50] + "..." if len(original) > 50 else original
        session_info = f" [Session: {session_id}]" if session_id else ""

        if rewritten and rewritten != original:
            rewritten_preview = (
                rewritten[:50] + "..." if len(rewritten) > 50 else rewritten
            )
            self.logger.info(
                f"Question{session_info}: '{question_preview}' → '{rewritten_preview}'"
            )
        else:
            self.logger.info(f"Question{session_info}: '{question_preview}'")

    def _log_classification_result(self, classification: dict):
        """Log question classification result concisely"""
        q_type = classification.get("question_type", "unknown")
        confidence = classification.get("confidence", 0.0)
        needs_search = classification.get("needs_vector_search", False)
        self.logger.info(
            f"Classification: {q_type} (confidence: {confidence:.1f}, search: {needs_search})"
        )

    def _log_heading_analysis(
        self, heading_info: dict, context_maintained: bool = True
    ):
        """Log heading analysis result concisely"""
        headings = heading_info.get("active_headings", [])
        confidence = heading_info.get("confidence", 0.0) or 0.0
        if headings:
            heading_preview = (
                headings[0]
                if len(headings) == 1
                else f"{headings[0]} (+{len(headings)-1} more)"
            )
            self.logger.info(
                f"Active headings: {heading_preview} (context: {'maintained' if context_maintained else 'updated'}, confidence: {confidence:.2f})"
            )
        else:
            self.logger.info(f"No active headings found (confidence: {confidence:.2f})")

    def _log_error_concise(
        self, operation: str, error: Exception, session_id: str = ""
    ):
        """Log errors concisely without full stacktrace"""
        session_info = f" [Session: {session_id}]" if session_id else ""
        self.logger.error(f"{operation} failed{session_info}: {str(error)}")

    def _log_performance(self, operation: str, duration: float, extra_info: str = ""):
        """Log performance metrics"""
        extra = f" - {extra_info}" if extra_info else ""
        self.logger.info(f"{operation}: {duration:.2f}s{extra}")

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_chatbots: Dict[str, EnhancedRAGChatbot] = {}
        self.config = {
            "top_k": 10,
            "min_score": 0.5,
            "max_answer_length": 1000,
            "show_sources": True,
            "use_context": True,
            "debug_mode": False,
            "similarity_threshold": 0.75,
            "qa_source_name": "Bo_cau_hoi_BAS_embeddings",  # Q&A source name for database search
            # Entity and section expansion thresholds
            "expand_entity_threshold": 3,
            "expand_section_threshold": 3,
        }

        # Q&A data cache
        self.qa_items = None
        self.qa_file_loaded = None

        # Genai client for Q&A similarity check
        self.genai_client = None

        # Cache for entity-section context (legacy) and heading context
        self.entity_section_context = None  # legacy
        self.context_cache_time = None
        self.context_cache_ttl = 3600  # 1 hour cache TTL
        self.heading_context_cache = None
        self.heading_context_cache_time = None
        self.heading_context_cache_ttl = 300  # 5 minutes cache TTL for heading context

        # Cache for dynamically extracted model patterns
        self.model_patterns_cache = None
        self.model_patterns_cache_time = None
        self.model_patterns_cache_ttl = (
            7200  # 2 hour cache TTL (patterns change less frequently)
        )

        # Cache for dynamically extracted model patterns
        self.model_patterns_cache = None
        self.model_patterns_cache_time = None
        self.model_patterns_cache_ttl = (
            7200  # 2 hour cache TTL (patterns change less frequently)
        )

        # Hybrid heading selection configuration
        self.heading_selection_config = {
            "ambiguity_threshold": 0.15,  # Score difference threshold for ambiguity
            "min_candidates": 2,  # Minimum candidates for ambiguity
            "max_candidates": 4,  # Maximum candidates to show in clarification
            "confidence_threshold": 0.6,  # Minimum confidence for direct selection
        }

        # Ambiguity patterns for detection
        self.ambiguous_patterns = {
            "generic_sensor": {
                "keywords": ["cảm biến", "sensor", "thiết bị đo", "đo lường"],
                "potential_matches": ["LS-BE-001", "WTX536", "cảm biến khí tượng"],
                "clarification_type": "sensor_selection",
            },
            "generic_specs": {
                "keywords": ["thông số", "specification", "đặc tính", "parameters"],
                "requires_subject": True,
                "clarification_type": "subject_specification",
            },
            "generic_product": {
                "keywords": ["sản phẩm", "product", "thiết bị", "equipment"],
                "potential_matches": [],  # Will be filled from available headings
                "clarification_type": "product_selection",
            },
        }

        # Question understanding collaborator (classification + heading analysis)
        self._classifier = QuestionClassifier(
            genai_client_provider=self._get_genai_client,
            heading_provider=self._get_heading_context,
            logger=self.logger,
            log_classification_result=self._log_classification_result,
            log_heading_analysis=self._log_heading_analysis,
            log_error_concise=self._log_error_concise,
        )

        # Session lifecycle + history persistence collaborator
        self._sessions = SessionManager(
            config_provider=lambda: self.config,
            logger=self.logger,
            log_process_step=self._log_process_step,
            log_error_concise=self._log_error_concise,
        )

    def _get_heading_context(self, limit: Optional[int] = 200) -> Dict[str, Any]:
        """Get and cache heading context (titles/ids/parents) from database for heading-first prompts."""
        import time

        now = time.time()
        if (
            self.heading_context_cache is not None
            and self.heading_context_cache_time is not None
            and now - self.heading_context_cache_time < self.heading_context_cache_ttl
        ):
            return self.heading_context_cache

        try:
            from ragbot.db.database_service import DatabaseService

            ctx = DatabaseService.get_heading_context(limit=limit)
            headings = ctx.get("headings", []) or []
            titles = []
            seen = set()
            for h in headings:
                t = (h.get("title") or "").strip()
                if not t:
                    continue
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    titles.append(t)

            self.heading_context_cache = {
                "headings": headings,
                "titles": titles,
                "distinct_heading_count": ctx.get("distinct_heading_count", 0),
                "parents": ctx.get("parents", {}),
                "truncated": ctx.get("truncated", False),
            }
            self.heading_context_cache_time = now
            self.logger.info(
                f"Cached heading context: returned={len(headings)} headings, distinct={ctx.get('distinct_heading_count', 0)}"
            )
            return self.heading_context_cache

        except Exception as e:
            self.logger.error(f"Error getting heading context: {e}")
            return {
                "headings": [],
                "titles": [],
                "distinct_heading_count": 0,
                "parents": {},
                "truncated": False,
            }

    def get_or_create_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._sessions.get_or_create_session(session_id, user_id)

    def _get_genai_client(self):
        """Get or initialize genai client"""
        if self.genai_client is None:
            try:
                self.genai_client = init_genai_client()
                self.logger.info("Initialized genai client")
            except Exception as e:
                self.logger.error(f"Failed to initialize genai client: {e}")
                self.genai_client = None
        return self.genai_client

    def _classify_question_type(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._classifier.classify(question, session_data)

    def _analyze_heading_and_rewrite(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._classifier.analyze_and_rewrite(question, session_data)

    def _fallback_question_classification(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._classifier.fallback_classify(question, session_data)

    def _process_single_source(
        self, source: Any, include_rank: bool = False, rank: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single source (RetrievalResult or dict) into standardized format

        Args:
            source: Source object to process
            include_rank: Whether to include rank field
            rank: Rank value if include_rank is True

        Returns:
            Processed source dictionary or None if processing failed
        """
        try:
            # Handle RetrievalResult objects
            if (
                hasattr(source, "score")
                and hasattr(source, "content")
                and hasattr(source, "get_pdf_name")
                and hasattr(source, "get_clean_pdf_name")
            ):
                result = {
                    "score": source.score,
                    "content": source.content,
                    "pdf_name": source.get_pdf_name(),
                    "page": source.get_page(),
                    "clean_pdf_name": source.get_clean_pdf_name(),
                    "meta": source.meta,
                }
                if include_rank:
                    result["rank"] = rank
                return result

            # Handle dictionary sources
            elif isinstance(source, dict):
                meta = source.get("meta", {})
                pdf_name = (
                    meta.get("pdf_name")
                    or meta.get("file_name")
                    or meta.get("source")
                    or "Unknown"
                )
                page = meta.get("page", "?")

                # Clean up the PDF name for better display (same logic as RetrievalResult.get_clean_pdf_name)
                if pdf_name == "Unknown":
                    clean_pdf_name = pdf_name
                else:
                    import os

                    filename = os.path.basename(pdf_name)
                    filename = re.sub(
                        r"\.(pdf|docx|jsonl)$", "", filename, flags=re.IGNORECASE
                    )
                    clean_pdf_name = filename.replace("_", " ").replace("-", " ")
                    if clean_pdf_name.startswith("(") and clean_pdf_name.endswith(")"):
                        clean_pdf_name = clean_pdf_name[1:-1]

                result = {
                    "score": source.get("score", 0.0),
                    "content": source.get("content", ""),
                    "pdf_name": pdf_name,
                    "page": str(page) if page is not None else "?",
                    "clean_pdf_name": clean_pdf_name,
                    "meta": meta,
                }
                if include_rank:
                    result["rank"] = rank
                return result

            else:
                self.logger.warning(f"Unknown source type: {type(source)}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing source: {str(e)}")
            return None

    def _convert_sources_to_dict(self, sources: List) -> List[Dict[str, Any]]:
        """
        Convert sources from RetrievalResult objects or dictionaries to standardized dict format

        Args:
            sources: List of sources (RetrievalResult objects or dicts)

        Returns:
            List of standardized source dictionaries
        """
        try:
            frontend_sources = []
            for i, source in enumerate(sources):
                try:
                    # self.logger.debug(f"Processing source {i}: {type(source)}")
                    processed_source = self._process_single_source(source)
                    if processed_source:
                        frontend_sources.append(processed_source)
                    else:
                        # Fallback processing for unknown types
                        try:
                            if hasattr(source, "score"):
                                score = source.score
                            elif hasattr(source, "__getitem__"):
                                score = (
                                    source.get("score", 0.0)
                                    if hasattr(source, "get")
                                    else 0.0
                                )
                            else:
                                score = 0.0

                            if hasattr(source, "content"):
                                content = source.content
                            elif hasattr(source, "__getitem__"):
                                content = (
                                    source.get("content", "")
                                    if hasattr(source, "get")
                                    else str(source)
                                )
                            else:
                                content = str(source)

                            frontend_sources.append(
                                {
                                    "score": score,
                                    "content": content,
                                    "pdf_name": "Unknown",
                                    "page": "?",
                                    "clean_pdf_name": "Unknown",
                                    "meta": {},
                                }
                            )
                        except Exception as fallback_error:
                            self.logger.error(
                                f"Fallback processing failed for source {i}: {str(fallback_error)}"
                            )
                            continue
                except Exception as e:
                    self.logger.error(f"Error processing source {i}: {str(e)}")
                    self.logger.error(f"Source details: {source}")
                    continue

            self.logger.info(f"Successfully processed {len(frontend_sources)} sources")
            return frontend_sources

        except Exception as e:
            self.logger.error(f"Error in source conversion: {str(e)}")
            return []

    def _check_qa_similarity(
        self,
        question: str,
        qa_source_name: Optional[
            str
        ] = None,  # Thay đổi từ qa_file_path thành qa_source_name
        similarity_threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        Check if question has similarity with Q&A data from database

        Args:
            question: User question
            qa_source_name: Optional Q&A source name (e.g., 'Bo_cau_hoi_BAS_embeddings')
            similarity_threshold: Optional similarity threshold (uses config if not provided)

        Returns:
            Answer from Q&A data if similar question found, None otherwise
        """
        try:
            # Use provided parameters or fallback to config
            qa_source = qa_source_name or self.config.get("qa_source_name")
            threshold = similarity_threshold or self.config.get(
                "similarity_threshold", 0.75
            )

            if not qa_source:
                self.logger.debug(
                    "No Q&A source name configured, skipping Q&A similarity check"
                )
                return None

            # Get genai client
            client = self._get_genai_client()
            if not client:
                self.logger.warning(
                    "No genai client available, skipping Q&A similarity check"
                )
                return None

            # Perform similarity check using database
            # Import VectorSearchService
            from ragbot.retrieval.vector_search import VectorSearchService

            # Import embed_query function
            from ragbot.chat.rag_engine import embed_query

            # Tạo embedding cho câu hỏi
            query_embedding = embed_query(client, question)

            # Tìm kiếm Q&A chunks từ database
            qa_results = VectorSearchService.search_qa_chunks(
                embedding=query_embedding,
                limit=5,  # Lấy top 5 kết quả
                min_score=threshold,
                qa_source_name=qa_source,
            )

            if not qa_results:
                self.logger.info(f"No similar Q&A found (threshold: {threshold})")
                return None

            # Lấy kết quả tốt nhất
            best_result = qa_results[0]
            best_similarity = best_result["score"]

            self.logger.info(
                f"Best Q&A similarity: {best_similarity:.3f} (threshold: {threshold})"
            )

            if best_similarity > threshold:
                # Trích xuất answer từ content
                content = best_result["content"]
                # Content format: "Question: {question}\nAnswer: {answer}"
                if "Answer:" in content:
                    answer = content.split("Answer:", 1)[1].strip()
                    self.logger.info(
                        "Found similar question in Q&A data, returning direct answer"
                    )
                    return answer
                else:
                    # Nếu không có format chuẩn, trả về toàn bộ content
                    self.logger.info(
                        "Found similar question in Q&A data, returning direct answer"
                    )
                    return content
            else:
                self.logger.info("No similar question found in Q&A data")
                return None

        except Exception as e:
            self.logger.error(f"Error in Q&A similarity check: {e}")
            return None

    def ask_question(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None,
        use_vector_search: bool = True,
        active_headings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a question and return answer using vector search

        Args:
            question: User question
            session_id: Session identifier
            user_id: Optional user identifier
            use_vector_search: Whether to use vector search (default True)
            active_headings: Optional list of headings to filter results by

        Returns:
            Dict with answer and metadata
        """
        try:
            # Validate input parameters
            if not question or not isinstance(question, str):
                return {
                    "error": "Invalid question parameter - must be a non-empty string"
                }

            if not session_id or not isinstance(session_id, str):
                return {
                    "error": "Invalid session_id parameter - must be a non-empty string"
                }

            question = question.strip()
            if not question:
                return {"error": "Question cannot be empty"}

            self._log_question_processing(question, session_id=session_id)
            start_time = time.time()

            # Get or create session data
            session = self.get_or_create_session(session_id, user_id)

            # Check if session creation failed
            if session is None:
                self._log_error_concise(
                    "Session creation",
                    Exception("Failed to create/get session"),
                    session_id,
                )
                return {"error": "Failed to create session"}

            # Get session metadata with proper null handling
            if session is None:
                self._log_error_concise(
                    "Session metadata", Exception("Session is None"), session_id
                )
                return {"error": "Session lost during processing"}

            session_metadata = session.get("metadata", {}) or {}

            # Step 1: Phân loại câu hỏi
            classification = self._classify_question_type(question, session_metadata)

            # Check if classification is None and provide a fallback
            if classification is None:
                self._log_error_concise(
                    "Question classification",
                    Exception("Classification returned None"),
                    session_id,
                )
                classification = {
                    "question_type": "BlueEco_BAS",
                    "confidence": 0.5,
                    "response": "Đang tìm kiếm thông tin...",
                    "needs_vector_search": True,
                    "needs_qa_check": True,
                }

            self._log_classification_result(classification)

            # Step 2: Nếu câu hỏi cần xử lý tiếp, phân tích heading và viết lại câu hỏi
            heading_info: Dict[str, Any] = {}
            rewritten_question = None

            if classification.get("question_type") in ["BlueEco_BAS", "company_info"]:
                # Use LLM to analyze heading and rewrite question
                heading_analysis = self._analyze_heading_and_rewrite(question, session)
                heading_info = heading_analysis.get("heading_info", {})
                rewritten_question = heading_analysis.get("rewritten_question")

                self._log_heading_analysis(
                    heading_info, heading_info.get("context_maintained", True)
                )
                if rewritten_question and rewritten_question != question:
                    self._log_question_processing(
                        question, rewritten_question, session_id
                    )
            else:
                # Cho các loại câu hỏi khác, giữ heading cũ
                heading_info = {
                    "active_headings": session_metadata.get("active_headings") or [],
                    "context_maintained": True,
                    "confidence": 0.5,
                }
            new_active_headings = heading_info.get("active_headings") or []

            # Guardrail: validate headings are in DB and preserve context if needed
            try:
                # Get valid headings from DB
                heading_ctx = self._get_heading_context(limit=300)
                raw_headings = heading_ctx.get("titles", []) or []
                valid_headings = set(self.dedupe_preserve_order(raw_headings))

                # Normalize for comparison
                def normalize(s: str) -> str:
                    return self.normalize_text(s)

                normalized_valid = {normalize(h) for h in valid_headings}

                # Validate new headings
                validated_headings = []
                for h in new_active_headings:
                    normalized_h = normalize(h)
                    if normalized_h in normalized_valid:
                        # Find exact match from DB
                        for vh in valid_headings:
                            if normalize(vh) == normalized_h:
                                validated_headings.append(vh)
                                break
                    else:
                        self.logger.warning(f"Heading '{h}' not found in DB, skipping")

                if validated_headings:
                    heading_info["active_headings"] = validated_headings
                    new_active_headings = validated_headings
                else:
                    # No valid headings, preserve previous if context_maintained
                    if heading_info.get("context_maintained", True):
                        prev_headings = session_metadata.get("active_headings") or []
                        if isinstance(prev_headings, str):
                            prev_headings = [prev_headings] if prev_headings else []
                        heading_info["active_headings"] = prev_headings
                        new_active_headings = prev_headings
                        self.logger.info(
                            f"Preserving previous headings: {prev_headings}"
                        )
                    else:
                        heading_info["active_headings"] = []
                        new_active_headings = []

            except Exception as _hg_guard_err:
                self.logger.warning(
                    f"Heading validation guard failed softly: {_hg_guard_err}"
                )

            # Step 2: Cập nhật active_headings trong session metadata nếu có thay đổi
            context_maintained = heading_info.get("context_maintained", True)
            prev_active_before_update = session_metadata.get("active_headings") or []

            if (
                new_active_headings
                and (
                    set(new_active_headings) != set(prev_active_before_update)
                    or not prev_active_before_update
                )
                and not context_maintained
            ):
                self._update_session_active_headings(
                    session_id, new_active_headings, heading_info
                )
                session_metadata["active_headings"] = new_active_headings
                session_metadata["heading_info"] = heading_info
                self._log_process_step(
                    f"Updated headings",
                    f"from {len(prev_active_before_update)} to {len(new_active_headings)} items",
                )
            elif (
                new_active_headings
                and not prev_active_before_update
                and context_maintained
            ):
                self._update_session_active_headings(
                    session_id, new_active_headings, heading_info
                )
                session_metadata["active_headings"] = new_active_headings
                session_metadata["heading_info"] = heading_info
                self._log_process_step(
                    "Set initial headings", f"{len(new_active_headings)} items"
                )

            # Step 3: Sử dụng rewritten_question từ entity analysis
            original_question = question

            # Sử dụng rewritten_question nếu có và khác với câu hỏi gốc
            if rewritten_question and rewritten_question != question:
                question = rewritten_question

            # Xử lý dựa trên loại câu hỏi
            question_type = classification.get("question_type", "BlueEco_BAS")

            # Nếu là greeting, off_topic, capability, trả về phản hồi ngay lập tức
            if question_type in ["greeting", "off_topic", "capability", "quotation"]:
                processing_time = time.time() - start_time

                # Add to history for tracking (store minimal context)
                self._add_to_history_db(
                    session_id,
                    original_question,
                    classification["response"],
                    [],
                    extra_metadata={
                        "heading_info": heading_info,
                        "active_headings": session_metadata.get("active_headings")
                        or [],
                        "original_question": original_question,
                        "rewritten_question": None,
                    },
                )

                return {
                    "answer": classification["response"],
                    "sources": [],
                    "processing_time": processing_time,
                    "question_type": question_type,
                    "classification": classification,
                    "heading_info": heading_info,
                    "original_question": original_question,
                    "rewritten_question": (
                        question if question != original_question else None
                    ),
                }

            # Nếu là BlueEco_BAS hoặc company_info, bỏ qua Q&A check và đi thẳng vào vector search
            if question_type in ["BlueEco_BAS", "company_info"]:
                # Step 4: Bỏ qua Q&A check, đi thẳng vào vector search cho cả 2 loại câu hỏi
                # (Có thể kích hoạt lại Q&A check trong tương lai nếu cần)
                """
                # Q&A check logic (commented out)
                qa_answer = self._check_qa_similarity(question)
                if qa_answer:
                    processing_time = time.time() - start_time
                    # Add to history for tracking
                    self._add_to_history_db(session_id, original_question, qa_answer, [])
                    return {
                        "answer": qa_answer,
                        "sources": [],
                        "processing_time": processing_time,
                        "question_type": question_type,
                        "classification": classification,
                        "source_type": "qa_similarity",
                        "qa_match": True,
                        "heading_info": heading_info,
                        "original_question": original_question,
                        "rewritten_question": question if question != original_question else None,
                    }
                """
                pass

            # Step 5: Thực hiện vector search cho cả BlueEco_BAS và company_info
            if classification.get("needs_vector_search", True):
                self._log_process_step(
                    "Starting vector search",
                    f"type: {classification.get('question_type')}",
                )
                sources = []
                answer = ""

                # Additional session validation before vector search
                if session is None:
                    self._log_error_concise(
                        "Vector search", Exception("Session became None"), session_id
                    )
                    return {"error": "Session lost during processing"}

                if use_vector_search:
                    # Use active_entity resolved from entity_info or session_metadata (avoid relying on session object)
                    if session is None:
                        self.logger.error(
                            "Session is None during vector search, cannot retrieve active_entity"
                        )
                        sources = []
                    else:
                        # Use provided active_headings override if passed; otherwise use resolved heading_info
                        if not active_headings:
                            active_headings = heading_info.get("active_headings") or []
                        sources = self._vector_search(
                            question, session, active_headings=active_headings
                        )

                    # Log vector search results
                    if not sources:
                        self.logger.warning("Vector search returned no results")
                else:
                    # Fallback to file-based search for compatibility
                    chatbot = self._get_or_create_chatbot()
                    if chatbot:
                        answer, sources = self._process_question_file_based(
                            chatbot, question, session
                        )

                # Generate answer using AI if we have sources
                if sources and not answer:
                    answer = self._generate_answer_with_sources(
                        question, sources, session
                    )

                # Calculate processing time
                processing_time = time.time() - start_time
                self._log_performance(
                    "Total processing",
                    processing_time,
                    f"{len(sources)} sources, answer: {len(answer)} chars",
                )

                # Add to session history in database (lưu câu hỏi gốc) + entity context
                self._add_to_history_db(
                    session_id,
                    original_question,
                    answer,
                    sources,
                    extra_metadata={
                        "heading_info": heading_info,
                        "active_headings": session_metadata.get("active_headings")
                        or [],
                        "original_question": original_question,
                        "rewritten_question": (
                            question if question != original_question else None
                        ),
                    },
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "processing_time": processing_time,
                    "question_type": question_type,
                    "classification": classification,
                    "source_type": "vector_search",
                    "qa_match": False,
                    "heading_info": heading_info,
                    "original_question": original_question,
                    "rewritten_question": (
                        question if question != original_question else None
                    ),
                }

            # Trường hợp không xác định, fallback to traditional processing
            self._log_process_step(
                "Fallback to traditional processing", "needs_vector_search=False"
            )
            sources = []
            answer = ""

            # Additional session validation before fallback vector search
            if session is None:
                self._log_error_concise(
                    "Fallback vector search",
                    Exception("Session became None"),
                    session_id,
                )
                return {"error": "Session lost during processing"}

            if use_vector_search:
                # Use vector search to find relevant documents
                if session is None:
                    self._log_error_concise(
                        "Fallback vector search",
                        Exception("Session is None"),
                        session_id,
                    )
                    sources = []
                else:
                    # Use provided active_headings override if passed; otherwise use resolved heading_info
                    if not active_headings:
                        active_headings = heading_info.get("active_headings") or []
                    sources = self._vector_search(
                        question, session, active_headings=active_headings
                    )

                # Log fallback search results
                if not sources:
                    self.logger.warning("Fallback vector search returned no results")
            else:
                # Fallback to file-based search for compatibility
                chatbot = self._get_or_create_chatbot()
                if chatbot:
                    answer, sources = self._process_question_file_based(
                        chatbot, question, session
                    )

            # Generate answer using AI if we have sources
            if sources and not answer:
                answer = self._generate_answer_with_sources(question, sources, session)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add to session history in database (lưu câu hỏi gốc) + entity context
            self._add_to_history_db(
                session_id,
                original_question,
                answer,
                sources,
                extra_metadata={
                    "heading_info": heading_info,
                    "active_headings": session_metadata.get("active_headings") or [],
                    "original_question": original_question,
                    "rewritten_question": (
                        question if question != original_question else None
                    ),
                },
            )

            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "question_type": "BlueEco_BAS",
                "classification": classification,
                "source_type": "vector_search",
                "qa_match": False,
                "heading_info": heading_info,
                "original_question": original_question,
                "rewritten_question": (
                    question if question != original_question else None
                ),
            }

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Emergency fallback response
            return {
                "error": f"Error processing question: {str(e)}",
                "answer": "Xin lỗi, đã xảy ra lỗi kỹ thuật. Vui lòng thử lại hoặc liên hệ hỗ trợ.",
                "sources": [],
                "processing_time": 0,
                "question_type": "error",
                "classification": None,
                "heading_info": None,
                "original_question": question,
                "rewritten_question": None,
            }

    def _update_session_active_headings(
        self, session_id: str, active_headings: List[str], heading_info: Dict[str, Any]
    ) -> None:
        return self._sessions.update_active_headings(
            session_id, active_headings, heading_info
        )

    def _update_session_active_entity(
        self, session_id: str, active_entity: str, entity_info: Dict[str, Any]
    ) -> None:
        return self._sessions.update_active_entity(
            session_id, active_entity, entity_info
        )

    def _generate_company_info_fallback(self, question: str) -> str:
        """
        Generate fallback response for company_info questions when no database sources found

        Args:
            question: Original user question

        Returns:
            Company information response
        """
        try:
            # Normalize question for matching
            question_lower = question.lower().strip()

            # Basic company information that should always be available
            company_info = {
                "contact": {
                    "keywords": [
                        "liên lạc",
                        "contact",
                        "liên hệ",
                        "thông tin liên lạc",
                    ],
                    "response": """Thông tin liên lạc của REECOTECH:

📧 Email: info@reecotech.com.vn
📞 Hotline Kinh doanh: 0938 696 131  
📞 Hotline Kỹ thuật: 0901 880 386

Chúng tôi luôn sẵn sàng hỗ trợ bạn!""",
                },
                "address": {
                    "keywords": ["địa chỉ", "address", "văn phòng", "trụ sở"],
                    "response": """Địa chỉ liên hệ:
📧 Email: info@reecotech.com.vn
📞 Hotline Kinh doanh: 0938 696 131
📞 Hotline Kỹ thuật: 0901 880 386

Để biết thông tin chi tiết về địa chỉ văn phòng, vui lòng liên hệ qua các kênh trên.""",
                },
                "about": {
                    "keywords": ["về công ty", "giới thiệu", "reecotech", "about"],
                    "response": """REECOTECH chuyên cung cấp các giải pháp:
• Hệ thống BAS (Berthing Aid System)  
• Giải pháp quan trắc môi trường
• Khí tượng thủy hải văn
• Lắp đặt và chuyển giao hệ thống trang thiết bị khảo sát

📧 Email: info@reecotech.com.vn
📞 Hotline: 0938 696 131 | 0901 880 386""",
                },
                "services": {
                    "keywords": ["dịch vụ", "service", "làm gì", "chuyên về"],
                    "response": """Dịch vụ của REECOTECH:
• Hệ thống BAS (Berthing Aid System)
• Giám sát môi trường (Đất, Nước, Không khí)  
• Giám sát khí tượng thủy văn
• Hệ thống cảnh báo sớm
• Hỗ trợ vận hành an toàn ngành hàng hải

📧 Email: info@reecotech.com.vn  
📞 Hotline: 0938 696 131 | 0901 880 386""",
                },
            }

            # Try to match question with available info
            for info_type, info_data in company_info.items():
                for keyword in info_data["keywords"]:
                    if keyword in question_lower:
                        return info_data["response"]

            # Default company response
            return """Thông tin về REECOTECH:
Chúng tôi chuyên cung cấp các giải pháp quan trắc môi trường, khí tượng thủy hải văn, và hệ thống BAS.

📧 Email: info@reecotech.com.vn
📞 Hotline Kinh doanh: 0938 696 131
📞 Hotline Kỹ thuật: 0901 880 386

Vui lòng liên hệ để biết thêm thông tin chi tiết!"""

        except Exception as e:
            self.logger.error(f"Error in company info fallback: {str(e)}")
            # Final fallback
            return """Thông tin liên lạc REECOTECH:
📧 Email: info@reecotech.com.vn
📞 Hotline: 0938 696 131 | 0901 880 386"""

    def _generate_polite_redirect_response(
        self, question: str, validation_result: Dict[str, Any]
    ) -> str:
        """
        Generate a polite response for off-topic questions

        Args:
            question: User question
            validation_result: Topic validation result

        Returns:
            Polite redirect response
        """
        try:
            confidence = validation_result.get("confidence", 0.5)

            # Generate contextual polite response
            if confidence > 0.7:
                response = f"""Xin chào! Cảm ơn bạn đã quan tâm đến REECOTECH. 

Tuy nhiên, câu hỏi "{question}" có vẻ không thuộc lĩnh vực chuyên môn của chúng tôi. REECOTECH chuyên về:
- Hệ thống BAS (Berthing Aid System)
- Giải pháp phân tích và đo lường cho các ngành công nghiệp sản xuất và R&D
- Giám sát môi trường (Đất, Nước, Không khí)
- Giám sát khí tượng thủy văn và hệ thống cảnh báo sớm
- Giải pháp hỗ trợ vận hành an toàn cho ngành hàng hải và ngoài khơi

Bạn có câu hỏi nào về các lĩnh vực này không? Tôi rất sẵn lòng hỗ trợ bạn! 😊"""
            else:
                response = f"""Chào bạn! Tôi là trợ lý AI của REECOTECH, Chuyên cung cấp các giải pháp quan trắc môi trường, khí tượng thủy hải văn, lắp đặt và chuyển giao hệ thống trang thiết bị khảo sát, nghiên cứu.

Câu hỏi của bạn có thể nằm ngoài phạm vi chuyên môn của tôi. Tôi có thể giúp bạn tìm hiểu về:
- Hệ thống BAS (Berthing Aid System)
- Giải pháp phân tích và đo lường cho các ngành công nghiệp sản xuất và R&D
- Giám sát môi trường (Đất, Nước, Không khí)
- Giám sát khí tượng thủy văn và hệ thống cảnh báo sớm
- Giải pháp hỗ trợ vận hành an toàn cho ngành hàng hải và ngoài khơi

Bạn có muốn tìm hiểu về bất kỳ chủ đề nào trong số này không? 🏢"""

            return response

        except Exception as e:
            self.logger.error(f"Error generating polite response: {str(e)}")
            return """Xin chào! Tôi là trợ lý AI của REECOTECH, Cung cấp các giải pháp quan trắc môi trường, khí tượng thủy hải văn, lắp đặt và chuyển giao hệ thống trang thiết bị khảo sát, nghiên cứu. 
Có vẻ như câu hỏi của bạn nằm ngoài lĩnh vực chuyên môn của tôi. Bạn có câu hỏi nào về các lĩnh vực này không? Chúng tôi rất sẳn lòng hỗ trợ 😊"""

    def set_qa_config(
        self,
        qa_source_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Set Q&A configuration for database search

        Args:
            qa_source_name: Q&A source name for database search (e.g., 'Bo_cau_hoi_BAS_embeddings')
            similarity_threshold: Similarity threshold for Q&A matching
        """
        if qa_source_name is not None:
            self.config["qa_source_name"] = qa_source_name
            self.logger.info(f"Q&A source name set to: {qa_source_name}")

        if similarity_threshold is not None:
            self.config["similarity_threshold"] = similarity_threshold
            self.logger.info(f"Q&A similarity threshold set to: {similarity_threshold}")

        # Clear cached Q&A data if source name changed
        if qa_source_name is not None and qa_source_name != self.qa_file_loaded:
            self.qa_items = None
            self.qa_file_loaded = None
            self.logger.info("Cleared cached Q&A data due to source name change")

    def auto_detect_qa_source(self) -> bool:
        """
        Auto detect Q&A source from database

        Returns:
            True if found and configured successfully, False otherwise
        """
        try:
            # Import Flask for app context check
            from flask import has_app_context

            if not has_app_context():
                self.logger.warning(
                    "No Flask app context available for Q&A auto-detection"
                )
                return False

            # Query database for Q&A documents
            from ragbot.models.document import DocumentChunk

            # Look for document chunks that might be Q&A data
            # Use proper string comparison instead of LIKE on JSON columns
            qa_chunks = (
                DocumentChunk.query.filter(DocumentChunk.clean_pdf_name.isnot(None))
                .filter(
                    DocumentChunk.clean_pdf_name.ilike("%embeddings%")
                    | DocumentChunk.clean_pdf_name.ilike("%qa%")
                    | DocumentChunk.clean_pdf_name.ilike("%Q&A%")
                )
                .distinct(DocumentChunk.clean_pdf_name)
                .limit(10)
                .all()
            )

            if not qa_chunks:
                self.logger.warning("No Q&A documents found in database")
                return False

            # Get unique clean_pdf_name values
            unique_qa_sources = list(
                set(chunk.clean_pdf_name for chunk in qa_chunks if chunk.clean_pdf_name)
            )

            if not unique_qa_sources:
                self.logger.warning("No valid Q&A source names found")
                return False

            # Use the first Q&A source name found
            qa_source_name = unique_qa_sources[0]
            self.logger.info(f"Auto-detected Q&A source: {qa_source_name}")

            # Set the configuration
            self.set_qa_config(qa_source_name=qa_source_name)

            self.logger.info(f"Successfully configured Q&A source: {qa_source_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error in auto-detecting Q&A source: {e}")
            return False

    def _vector_search(
        self,
        question: str,
        session: Dict[str, Any],
        active_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform search for relevant documents using vector search

        Args:
            question: User question
            session: Session data
            active_headings: Filter results by specific headings (optional)

        Returns:
            List of relevant document chunks
        """
        try:
            # Validate session parameter
            if session is None:
                self._log_error_concise(
                    "Vector search", Exception("Called with None session")
                )
                return []

            # Use vector search only
            self._log_process_step("Using vector search", "vector similarity search")

            question_embedding = self._get_question_embedding(question)
            if not question_embedding:
                self._log_error_concise(
                    "Embedding generation",
                    Exception("Could not generate embedding"),
                )
                return []

            # Perform ensemble search (includes vector + BM25 + heading expansion)
            ensemble_service = EnsembleRetrieverService()
            ensemble_service.initialize()
            results = ensemble_service.search(
                query=question,
                embedding=question_embedding,
                limit=session["config"]["top_k"],
                min_score=0.01,  # Filter out very low quality results
                active_headings=active_headings,
            )

            self._log_search_result("Vector search", len(results))

            # Format results for frontend compatibility
            formatted_sources = []

            for i, result in enumerate(results):
                # Handle vector search result format
                if isinstance(result, dict):

                    # Vector search results
                    formatted_source = {
                        "score": result.get("score", 0.0),
                        "content": result.get("content", ""),
                        "pdf_name": result.get("file_name")
                        or result.get("pdf_name", "Unknown"),
                        "page": result.get("page", "?"),
                        "clean_pdf_name": result.get("clean_pdf_name", "Unknown"),
                        "meta": {
                            "page": result.get("page"),
                            "block_index": result.get("block_index"),
                            "bbox": result.get("bbox"),
                            "file_name": result.get("file_name")
                            or result.get("pdf_name"),
                            # Heading metadata
                            "heading_id": result.get("heading_id"),
                            "heading_title": result.get("heading_title"),
                            "heading_parent_id": result.get("heading_parent_id"),
                            "heading_level": result.get("heading_level"),
                            "search_method": "vector_only",
                        },
                    }

                    formatted_sources.append(formatted_source)

            return formatted_sources

        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return []

    def _query_graph_rag_service(
        self, question: str, session: Dict[str, Any]
    ) -> Optional[str]:
        """
        As a fallback, query the Graph RAG (Neo4j + LLM) service. Uses session metadata active_entity
        to help resolve ambiguous follow-up questions (e.g., pronouns like 'nó').

        Returns the raw result string from the GraphCypherQAChain or None on error.
        """
        try:
            from ragbot.retrieval.graph_rag_service import OptimizedGraphRAGService

            # Validate session parameter
            if session is None:
                self.logger.error("_query_graph_rag_service called with None session")
                return None

            active_entity = (session.get("metadata") or {}).get("active_entity")
            rag = OptimizedGraphRAGService()
            result = rag.process_query(question, active_entity=active_entity)
            return result
        except Exception as e:
            self.logger.error(f"Error querying GraphRAGService: {e}")
            return None

    def _get_question_embedding(self, question: str) -> Optional[List[float]]:
        """
        Generate embedding for the question using Google AI (same as document embeddings)

        Args:
            question: User question

        Returns:
            Embedding vector or None
        """
        try:
            # Use the same embedding service as documents to ensure compatibility
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from ragbot.ingestion.embedder import embed_single_text, get_embedding_vector

            # Initialize Google AI client and generate embedding
            client = self._get_genai_client()
            if client is None:
                self._log_error_concise(
                    "Embedding generation",
                    Exception("Failed to initialize genai client"),
                )
                return None

            response = embed_single_text(
                client=client,
                text=question,
                task_type="RETRIEVAL_QUERY",  # Use QUERY for questions, DOCUMENT for documents
            )

            # Extract embedding vector from response
            embedding = get_embedding_vector(response, index=0)

            if embedding and len(embedding) == 1536:
                return embedding
            else:
                self.logger.warning(
                    f"Invalid embedding generated: {len(embedding) if embedding else 0} dimensions"
                )
                return None

        except Exception as e:
            self._log_error_concise("Question embedding generation", e)
            # Fallback to mock embedding for testing
            try:
                import numpy as np

                np.random.seed(hash(question) % (2**32))
                embedding = np.random.normal(0, 1, 1536).tolist()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (embedding / norm).tolist()
                return embedding
            except Exception:
                return None

    def _generate_answer_with_sources(
        self, question: str, sources: List[Dict], session: Dict
    ) -> str:
        """
        Generate AI answer using retrieved sources

        Args:
            question: User question
            sources: Retrieved sources
            session: Session data

        Returns:
            Generated answer
        """
        try:
            # Validate session parameter
            if session is None:
                self._log_error_concise(
                    "Answer generation", Exception("Called with None session")
                )
                return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."

            # Get recent context
            context = self._get_recent_context(session.get("history", []))

            # Build prompt with sources using helper method
            prompt = self._build_prompt(question, sources, context)
            # Integrate with Google AI to generate real answer
            try:
                # Validate required parameters
                if not question or not isinstance(question, str):
                    raise ValueError("Invalid question parameter")
                if not sources or not isinstance(sources, list):
                    raise ValueError("Invalid sources parameter")
                if not prompt or not isinstance(prompt, str):
                    raise ValueError("Invalid prompt parameter")

                self._log_process_step(
                    "Generating AI answer",
                    f"{len(sources)} sources, {len(prompt)} chars prompt",
                )

                # Use the same Google AI client as in embedding generation
                client = self._get_genai_client()
                if not client:
                    raise ValueError("Failed to initialize AI client")

                # Generate answer using Google AI
                answer = generate_answer(client, prompt)

                # Extract text from response if it's a response object
                answer_text = ""
                if isinstance(answer, str):
                    answer_text = answer.strip()
                else:
                    answer_text = str(answer).strip()

                if answer_text and len(answer_text) > 10:
                    self._log_process_step(
                        "Answer generated", f"{len(answer_text)} characters"
                    )
                    return answer_text
                else:
                    self.logger.warning(
                        f"Generated answer too short or empty: '{answer_text}'"
                    )
                    return f"Dựa trên {len(sources)} nguồn tài liệu, tôi không thể tạo câu trả lời phù hợp cho câu hỏi này."

            except ImportError as import_error:
                self.logger.error(f"Import error in AI generation: {str(import_error)}")
                import traceback

                self.logger.error(f"Import traceback: {traceback.format_exc()}")
                if sources:
                    summary = f"Dựa trên {len(sources)} nguồn tài liệu tìm thấy về '{question}':\n\n"
                    for i, source in enumerate(sources[:3], 1):
                        content = (
                            source.get("content", "")
                            if isinstance(source, dict)
                            else str(source)
                        )
                        summary += f"{i}. {content[:200]}...\n\n"
                    return summary
                else:
                    return "Xin lỗi, có lỗi hệ thống khi tạo câu trả lời."
            except ValueError as val_error:
                self.logger.error(f"Parameter validation error: {str(val_error)}")
                return f"Lỗi tham số: {str(val_error)}"
            except Exception as ai_error:
                self.logger.error(
                    f"Unexpected error with AI generation: {str(ai_error)}"
                )
                import traceback

                self.logger.error(f"AI generation traceback: {traceback.format_exc()}")
                # Fallback to summarized response based on sources
                if sources:
                    summary = f"Dựa trên {len(sources)} nguồn tài liệu tìm thấy về '{question}':\n\n"
                    for i, source in enumerate(sources[:3], 1):
                        try:
                            content = (
                                source.get("content", "")
                                if isinstance(source, dict)
                                else str(source)
                            )
                            summary += f"{i}. {content[:200]}...\n\n"
                        except Exception as source_error:
                            self.logger.error(
                                f"Error processing source {i}: {str(source_error)}"
                            )
                            summary += f"{i}. [Lỗi xử lý nguồn]\n\n"
                    return summary
                else:
                    return "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn."

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            if sources:
                summary = f"Dựa trên {len(sources)} tài liệu tìm thấy:\n\n"
                for i, source in enumerate(sources[:3], 1):
                    summary += f"📄 Nguồn {i} (Score: {source.get('score', 0):.2f}):\n{source['content'][:300]}...\n\n"

                summary += "Tóm tắt: Các tài liệu chứa thông tin phù hợp. Hãy xem chi tiết các nguồn trên để biết thêm thông tin."
                return summary
            else:
                return "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn."

    def _add_to_history_db(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        return self._sessions.add_to_history(
            session_id, question, answer, sources, extra_metadata
        )

    def _get_or_create_chatbot(
        self, embedded_file: Optional[str] = None
    ) -> Optional[EnhancedRAGChatbot]:
        """
        Get or create a chatbot instance

        Args:
            embedded_file: Path to embedded file (optional - if None, searches all files)

        Returns:
            EnhancedRAGChatbot instance or None
        """
        # If user passed a specific embedded_file, use it. Otherwise find available files
        if embedded_file:
            ef_param = embedded_file
            key = str(embedded_file)
        else:
            embedded_files = self._find_embedded_files()
            if not embedded_files:
                return None

            # Always use multi-file search when no specific file is requested
            ef_param = embedded_files
            # Create a stable cache key for this set of files
            key = "multi:" + "|".join(sorted([str(p) for p in embedded_files]))

        if key not in self.active_chatbots:
            try:
                self.logger.info(f"Creating chatbot with files: {ef_param}")
                self.active_chatbots[key] = EnhancedRAGChatbot(ef_param)
            except Exception as e:
                self.logger.error(f"Failed to create chatbot: {str(e)}")
                return None

        return self.active_chatbots[key]

    def _find_embedded_files(self) -> List[str]:
        """
        Find available embedded files

        Returns:
            List of embedded file paths
        """
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            return []

        embedded_files = []
        patterns = ["*_embedded.jsonl", "*embedded*.jsonl"]

        for pattern in patterns:
            embedded_files.extend([str(f) for f in processed_dir.glob(pattern)])

        return sorted(
            embedded_files, key=lambda x: Path(x).stat().st_mtime, reverse=True
        )

    def _process_question_file_based(
        self, chatbot: EnhancedRAGChatbot, question: str, session: Dict[str, Any]
    ) -> tuple:
        """
        Process a question using file-based search with the chatbot

        Args:
            chatbot: EnhancedRAGChatbot instance
            question: User question
            session: Session data

        Returns:
            Tuple of (answer, sources)
        """
        try:
            # Validate session parameter
            if session is None:
                self.logger.error(
                    "_process_question_file_based called with None session"
                )
                return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn.", []

            # Get sources using retrieval
            self.logger.info(
                f"Starting file-based retrieval for question: {question[:50]}..."
            )
            sources = retrieve(
                chatbot.store,
                chatbot.client,
                question,
                top_k=session["config"]["top_k"],
                min_score=session["config"]["min_score"],
                use_metadata_ranking=session["config"].get("use_smart_ranking", True),
            )

            if not sources:
                self.logger.info("No sources found")
                return "Không tìm thấy tài liệu liên quan đến câu hỏi của bạn", []

            # Convert sources to frontend-friendly format
            frontend_sources = self._convert_sources_to_dict(sources)

            # Get recent context for better responses
            context = self._get_recent_context(session["history"])

            # Build prompt and generate answer
            prompt = self._build_prompt(question, sources, context)

            if chatbot.client:
                try:
                    # Use the shared generator to ensure compatibility with the google-genai client
                    answer = generate_answer(chatbot.client, prompt)
                except Exception as e:
                    self.logger.error(f"Error generating response: {str(e)}")
                    answer = f"Error generating AI response: {str(e)}"
            else:
                self.logger.warning("No AI client available, using search results only")
                answer = "AI client not available. Only search results shown."

            return answer, frontend_sources

        except Exception as e:
            self.logger.error(f"Error in file-based question processing: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error processing question: {str(e)}", []

    def _build_prompt(self, question: str, sources: List, context: str) -> str:
        return build_prompt(question, sources, context, logger=self.logger)

    def _get_recent_context(self, history: List[Dict], last_n: int = 3) -> str:
        return get_recent_context(history, last_n, logger=self.logger)

    def search_documents(
        self, query: str, embedded_file: str, top_k: int = 10, min_score: float = 0.5
    ) -> List[Dict]:
        """
        Search documents without generating full answer

        Args:
            query: Search query
            embedded_file: Path to embedded file
            top_k: Number of top results
            min_score: Minimum score threshold

        Returns:
            List of search results
        """
        try:
            chatbot = self._get_or_create_chatbot(embedded_file)
            if not chatbot:
                return []

            sources = retrieve(
                chatbot.store,
                chatbot.client,
                query,
                top_k=top_k,
                min_score=min_score,
                use_metadata_ranking=True,
            )

            results = []
            for i, source in enumerate(sources, 1):
                try:
                    processed_source = self._process_single_source(
                        source, include_rank=True, rank=i
                    )
                    if processed_source:
                        results.append(processed_source)
                    else:
                        self.logger.warning(
                            f"Unknown search source type: {type(source)}"
                        )
                        continue
                except Exception as e:
                    self.logger.error(
                        f"Error processing search source {source}: {str(e)}"
                    )
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Error in document search: {str(e)}")
            return []

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration

        Returns:
            Current config
        """
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update configuration

        Args:
            new_config: New configuration values

        Returns:
            True if updated successfully
        """
        try:
            self.config.update(new_config)
            return True
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chatbot usage statistics

        Returns:
            Statistics dictionary
        """
        try:
            # Get session statistics from database
            all_sessions = DatabaseService.get_all_chat_sessions()
            total_sessions = len(all_sessions)
            total_interactions = sum(len(session.messages) for session in all_sessions)

            # Calculate sources statistics
            total_sources = 0
            for session in all_sessions:
                for message in session.messages:
                    # Count sources in message metadata if available
                    if hasattr(message, "metadata") and message.metadata:
                        sources = message.metadata.get("sources", [])
                        total_sources += len(sources)

            return {
                "total_sessions": total_sessions,
                "total_interactions": total_interactions,
                "total_sources_retrieved": total_sources,
                "average_sources_per_interaction": (
                    total_sources / total_interactions if total_interactions > 0 else 0
                ),
                "active_chatbots": len(self.active_chatbots),
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {
                "total_sessions": 0,
                "total_interactions": 0,
                "total_sources_retrieved": 0,
                "average_sources_per_interaction": 0,
                "active_chatbots": len(self.active_chatbots),
                "error": str(e),
            }
