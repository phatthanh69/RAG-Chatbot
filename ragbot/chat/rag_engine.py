from __future__ import annotations

import datetime
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional

from ragbot.llm.client import init_genai_client
from ragbot.retrieval.simple_vector_store import SimpleVectorStore
from ragbot.utils.calculations import cosine_similarity


# Timezone configuration for Vietnam (UTC+7)
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


@dataclass
class RetrievalResult:
    score: float
    content: str
    meta: dict

    def get_pdf_name(self) -> str:
        """Get PDF name from metadata with fallback"""
        if not self.meta:
            return "Unknown"
        # Try multiple field names that might contain the file name
        return (
            self.meta.get("pdf_name")
            or self.meta.get("file_name")
            or self.meta.get("source")
            or "Unknown"
        )

    def get_page(self) -> str:
        """Get page number from metadata with fallback"""
        if not self.meta:
            return "?"
        page = self.meta.get("page")
        return str(page) if page is not None else "?"

    def get_clean_pdf_name(self) -> str:
        """Get cleaned PDF name for display"""
        pdf_name = self.get_pdf_name()
        if pdf_name == "Unknown":
            return pdf_name

        filename = os.path.basename(pdf_name)

        # Remove file extension if present
        filename = re.sub(r"\.(pdf|docx|jsonl)$", "", filename, flags=re.IGNORECASE)

        # Clean up the name for better display
        clean_name = filename.replace("_", " ").replace("-", " ")

        # Remove common prefixes and clean up
        if clean_name.startswith("(") and clean_name.endswith(")"):
            clean_name = clean_name[1:-1]

        return clean_name

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization"""
        return {
            "score": self.score,
            "content": self.content,
            "pdf_name": self.get_pdf_name(),
            "page": self.get_page(),
            "clean_pdf_name": self.get_clean_pdf_name(),
            "meta": self.meta,
        }


def load_store(embedded_jsonl_path: str) -> SimpleVectorStore:
    return SimpleVectorStore.from_jsonl(embedded_jsonl_path)


def embed_query(client: Any, text: str) -> List[float]:
    # Import configuration (try package path first, then fallback)
    from ragbot.config import config

    # Use RETRIEVAL_QUERY for query embeddings to match with RETRIEVAL_DOCUMENT
    query_config = {
        "task_type": "RETRIEVAL_QUERY",
        "output_dimensionality": config.EMBEDDING_DIMENSIONS,
    }

    resp = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=[text],
        config=query_config,  # type: ignore
    )
    if resp.embeddings and len(resp.embeddings) > 0 and resp.embeddings[0].values:
        return resp.embeddings[0].values
    else:
        raise ValueError("Failed to get embeddings from response")


@dataclass
class QAItem:
    question: str
    answer: str
    embedding: List[float]


def load_qa_data_with_embeddings(qa_file_path: str) -> Optional[List["QAItem"]]:
    """
    Load Q&A data with pre-computed embeddings from JSONL file
    OPTIMIZED VERSION - loads pre-computed embeddings
    """
    try:
        if not qa_file_path.lower().endswith(".jsonl"):
            logging.warning(f"Expected .jsonl file, got: {qa_file_path}")
            return None

        if not os.path.exists(qa_file_path):
            logging.warning(f"Q&A embeddings file not found: {qa_file_path}")
            return None

        qa_items: List[QAItem] = []

        with open(qa_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)

                    # Validate required fields
                    if not all(
                        key in obj for key in ["question", "answer", "embedding"]
                    ):
                        logging.warning(f"Missing required fields on line {line_num}")
                        continue

                    qa_items.append(
                        QAItem(
                            question=obj["question"],
                            answer=obj["answer"],
                            embedding=obj["embedding"],
                        )
                    )

                except json.JSONDecodeError as e:
                    logging.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

        if not qa_items:
            logging.warning(f"No valid Q&A items loaded from: {qa_file_path}")
            return None

        logging.info(
            f"✅ Loaded {len(qa_items)} Q&A items with pre-computed embeddings"
        )
        return qa_items

    except Exception as e:
        logging.error(f"Error loading Q&A embeddings file {qa_file_path}: {e}")
        return None


def find_similar_question_optimized(
    client: Any,
    query: str,
    qa_items: List[QAItem],
    similarity_threshold: float = 0.75,
) -> Optional[str]:
    """
    OPTIMIZED: Find similar question using pre-computed embeddings
    Only needs to compute embedding for the query, not for Q&A items
    """
    try:
        # Get embedding for the query only
        query_embedding = embed_query(client, query)

        best_similarity = 0.0
        best_answer = None

        # Compare with pre-computed embeddings (NO API CALLS)
        for qa_item in qa_items:
            similarity = cosine_similarity(query_embedding, qa_item.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = qa_item.answer

        logging.info(f"Best similarity score: {best_similarity:.3f}")

        if best_similarity > similarity_threshold:
            logging.info(
                f"Found similar question with similarity {best_similarity:.3f} > {similarity_threshold}"
            )
            return best_answer
        else:
            logging.info(
                f"No similar question found (best: {best_similarity:.3f} <= {similarity_threshold})"
            )
            return None

    except Exception as e:
        logging.error(f"Error in find_similar_question_optimized: {e}")
        return None


def generate_answer(
    client: Any, prompt: str, max_retries: Optional[int] = None
) -> str:
    # logger = logging.getLogger(__name__)
    # logger.info(f"Prompt sent to LLM: {prompt}")

    # Import configuration (try package path first, then fallback)
    from ragbot.config import config

    if max_retries is None:
        max_retries = config.MAX_RETRIES

    # Prefer an available, lightweight model; allow env override.
    primary = config.GENERATION_MODEL
    fallbacks = [primary]

    last_err: Exception | None = None

    for attempt in range(max_retries):
        for model_name in fallbacks:
            try:
                chat = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config.get_generation_config(),  # type: ignore
                )
                try:
                    if chat.text:
                        response = chat.text.strip()
                        if (
                            response and len(response) > 10
                        ):  # Ensure the answer has content
                            return response
                        else:
                            return "Sorry, I could not generate a suitable answer for this question."
                    else:
                        return str(chat)
                except Exception:
                    return str(chat)
            except Exception as e:
                last_err = e
                continue

        if attempt < max_retries - 1:
            print(f"Retrying attempt {attempt + 2}...")

    # If all attempts fail, raise the last error for visibility
    raise (
        last_err
        if last_err
        else RuntimeError("Failed to generate an answer after multiple attempts")
    )


def build_prompt(
    query: str, hits: List[RetrievalResult], conversation_context: str = ""
) -> str:
    context_parts = []
    for i, h in enumerate(hits, 1):
        meta = h.meta or {}
        src = meta.get("pdf_name", meta.get("source", "unknown"))
        page = meta.get("page")
        # Clean file name for readability
        if src != "unknown":
            src = src.replace("_", " ").replace("-", " ")
        context_parts.append(
            f"[Document {i} | Relevance: {h.score:.3f} | Source: {src} | Page: {page}]\n{h.content}\n"
        )

    context_block = "\n---\n".join(context_parts) if context_parts else "(no context)"

    sys_instr = (
        "Bạn là một trợ lý AI chuyên nghiệp và thân thiện,chuyên tư vấn các sản phẩm và giải pháp của REECOTECH. "
        "LƯU Ý QUAN TRỌNG: Chỉ sử dụng thông tin có trong tài liệu. "
        "KHÔNG được bịa đặt hoặc bổ sung kiến thức ngoài tài liệu."
        "Trả lời trực tiếp, KHÔNG viết lời dẫn, KHÔNG nhắc lại quy tắc."
        "Tuyệt đối không được sử dụng các cụm từ: Dựa trên thông tin được cung cấp, Dựa trên các tài liệu, v.v."
    )

    context_section = ""
    if conversation_context:
        context_section = f"\nNgữ cảnh hội thoại trước đó:\n{conversation_context}\n"

    prompt = f"""
    {sys_instr}
    {context_section}
    Câu hỏi của người dùng: {query}

    Ngữ cảnh tài liệu:
    {context_block}

    Nguyên tắc trả lời:
    
    - Trả lời ngắn gọn, có cấu trúc (khoảng 8–10 câu)
    - Ưu tiên nội dung quan trọng nếu có quá nhiều thông tin
    - Sử dụng gạch đầu dòng hoặc đánh số nếu phù hợp
    - Nếu thiếu thông tin, hãy nói: "Rất tiếc, tôi hiện tại không có đủ thông tin để trả lời câu hỏi này.Xin vui lòng liên hệ email:info@reecotech.com.vn;Hotline:Phòng Kinh doanh:0938 696 131 và Phòng Kỹ thuật:0901 880 386 để được hỗ trợ thêm."
    """.strip()
    return textwrap.dedent(prompt)


def retrieve(
    store: SimpleVectorStore,
    client: Any,
    query: str,
    top_k: int = 10,
    min_score: float = 0.5,
    use_metadata_ranking: bool = True,
) -> List[RetrievalResult]:
    """
    Retrieve with metadata ranking and section-aware capability
    """
    logging.debug(f"Starting retrieval with top_k={top_k}, min_score={min_score}")

    # Detect if we need to expand search scope for section queries
    should_expand_search = False

    # If using metadata ranking, check section query first
    if use_metadata_ranking:
        try:
            logging.debug("Attempting to import metadata_ranker...")
            from ragbot.retrieval.metadata_ranker import create_smart_ranker  # type: ignore[import-not-found]

            logging.debug("Successfully imported create_smart_ranker")

            logging.debug("Creating smart ranker...")
            ranker = create_smart_ranker()
            logging.debug(f"Created ranker: {type(ranker)}")

            # Take sample to detect section
            qv = embed_query(client, query)
            sample_results = store.search_by_vector(
                qv, top_k=min(10, len(store.items)), min_score=min_score
            )
            sample_retrieval_results = []
            for score, item in sample_results:
                sample_retrieval_results.append(
                    RetrievalResult(
                        score=score,
                        content=item.get("content", ""),
                        meta=item.get("meta", {}),
                    )
                )

            # Detect section query
            target_section = ranker.detect_section_query(
                query, sample_retrieval_results
            )
            if target_section:
                should_expand_search = True
                logging.info(
                    f"Section query detected: '{target_section}' - Expanding search scope"
                )

        except ImportError as e:
            logging.warning(
                "Metadata ranker not available for section detection, skipping"
            )
            logging.debug(f"ImportError details: {e}")
            pass

    # Adjust search scope based on section detection
    if should_expand_search:
        # Get more results to ensure enough section content
        search_top_k = min(len(store.items), top_k * 4)
    elif use_metadata_ranking:
        search_top_k = top_k * 2  # Normal metadata ranking
    else:
        search_top_k = top_k  # Basic search

    qv = embed_query(client, query)
    results = store.search_by_vector(qv, top_k=search_top_k, min_score=min_score)

    out: List[RetrievalResult] = []
    for score, item in results:
        # Save original cosine score to metadata for later filtering
        meta = item.get("meta", {}).copy()
        meta["original_score"] = score

        out.append(
            RetrievalResult(score=score, content=item.get("content", ""), meta=meta)
        )

    # Apply metadata ranking if requested
    if use_metadata_ranking and out:
        try:
            logging.debug("Attempting to import metadata_ranker...")
            from ragbot.retrieval.metadata_ranker import create_smart_ranker  # type: ignore[import-not-found]

            logging.debug("Successfully imported create_smart_ranker")

            logging.debug("Creating smart ranker...")
            ranker = create_smart_ranker()
            logging.debug(f"Created ranker: {type(ranker)}")

            # Detect section query to adjust final selection
            target_section = (
                ranker.detect_section_query(query, out)
                if should_expand_search
                else None
            )
            logging.debug(f"Target section detected: {target_section}")

            logging.debug("Calling ranker.rerank_results...")
            out = ranker.rerank_results(out, query, min_score)
            logging.debug(f"Rerank completed. Result type: {type(out)}")
            logging.debug(f"Result length: {len(out) if out else 0}")
            if out:
                print(f"First reranked result type: {type(out[0])}")
                print(f"First reranked result has score: {hasattr(out[0], 'score')}")

            # Intelligent truncation cho section queries
            if target_section:
                final_results = []
                section_results = []
                other_results = []

                # Categorize results
                for result in out:
                    result_section = result.meta.get("section") or ""
                    if (
                        result_section
                        and result_section.upper() == target_section.upper()
                    ):
                        section_results.append(result)
                    else:
                        other_results.append(result)

                # Prioritize section content - take all if direct section query
                section_count = len(section_results)
                if section_count > 0:
                    # Check if we need to take all section content
                    if section_count > top_k:
                        # Take all section content if more than top_k
                        final_results.extend(section_results)
                        logging.info(
                            f"Section expansion: returning {section_count} items (exceeds top_k={top_k})"
                        )
                    else:
                        # Take section content and add other results if space remains
                        final_results.extend(section_results)
                        remaining_slots = top_k - len(final_results)
                        if remaining_slots > 0:
                            final_results.extend(other_results[:remaining_slots])
                else:
                    # Fallback if no section content found
                    final_results = out[:top_k]

                out = final_results
                logging.debug(
                    f"Section-aware selection: {len(section_results)} section items, {len(final_results)} total"
                )
                logging.debug(f"Final results type: {type(final_results)}")
                if final_results:
                    logging.debug(f"First final result type: {type(final_results[0])}")
                    logging.debug(
                        f"First final result has score: {hasattr(final_results[0], 'score')}"
                    )
                    logging.debug(
                        f"First final result has content: {hasattr(final_results[0], 'content')}"
                    )
                    logging.debug(
                        f"First final result has meta: {hasattr(final_results[0], 'meta')}"
                    )
            else:
                # Normal truncation
                out = out[:top_k]
                logging.debug(f"Normal truncation applied. Result type: {type(out)}")
                if out:
                    logging.debug(f"First truncated result type: {type(out[0])}")
                    logging.debug(
                        f"First truncated result has score: {hasattr(out[0], 'score')}"
                    )

        except ImportError as e:
            logging.warning("Metadata ranker not available, using basic ranking")
            logging.debug(f"ImportError details: {e}")
            logging.debug(f"Current out type: {type(out)}")
            if out:
                logging.debug(f"First result type: {type(out[0])}")
                logging.debug(f"First result has score: {hasattr(out[0], 'score')}")
            out = out[:top_k]
    else:
        out = out[:top_k]

    logging.debug(f"Retrieval completed. Returning {len(out)} results")
    logging.debug(f"First result type: {type(out[0]) if out else 'No results'}")
    if out:
        logging.debug(f"First result class: {out[0].__class__}")
        logging.debug(f"First result attributes: {dir(out[0])}")
        logging.debug(f"First result has score: {hasattr(out[0], 'score')}")
        logging.debug(f"First result has content: {hasattr(out[0], 'content')}")
        logging.debug(f"First result has meta: {hasattr(out[0], 'meta')}")

        # Check if it's actually a RetrievalResult
        if (
            hasattr(out[0], "score")
            and hasattr(out[0], "content")
            and hasattr(out[0], "meta")
        ):
            print("First result appears to be RetrievalResult-like")
            print(f"First result score: {out[0].score}")
            print(f"First result content length: {len(out[0].content)}")
            print(
                f"First result meta keys: {list(out[0].meta.keys()) if isinstance(out[0].meta, dict) else 'Not a dict'}"
            )
        else:
            logging.warning(
                "First result does NOT have expected RetrievalResult attributes!"
            )
            logging.debug(
                f"First result actual attributes: {[attr for attr in dir(out[0]) if not attr.startswith('_')]}"
            )

    return out


class ChatSession:
    """Quản lý phiên trò chuyện với lịch sử"""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.session_id = datetime.datetime.now(VIETNAM_TIMEZONE).strftime(
            "%Y%m%d_%H%M%S"
        )

    def add_interaction(
        self, question: str, answer: str, sources: List[RetrievalResult]
    ):
        """Thêm một tương tác vào lịch sử"""
        interaction = {
            "timestamp": datetime.datetime.now(VIETNAM_TIMEZONE).isoformat(),
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "score": src.score,
                    "content": (
                        src.content[:200] + "..."
                        if len(src.content) > 200
                        else src.content
                    ),
                    "meta": src.meta,
                }
                for src in sources
            ],
        }
        self.history.append(interaction)

    def save_session(self, save_dir: Optional[str] = None):
        """Lưu phiên trò chuyện"""
        # Import configuration
        from ragbot.config import paths

        if save_dir is None:
            save_dir = str(paths.CHAT_SESSIONS_DIR)

        os.makedirs(save_dir, exist_ok=True)
        filename = f"chat_session_{self.session_id}.json"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "created_at": datetime.datetime.now(VIETNAM_TIMEZONE).isoformat(),
                    "total_interactions": len(self.history),
                    "history": self.history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return filepath

    def get_recent_context(self, last_n: int = 3) -> str:
        """Lấy ngữ cảnh từ các câu hỏi gần đây"""
        if not self.history or last_n <= 0:
            return ""

        recent = self.history[-last_n:]
        context_parts = []

        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Q{i}: {interaction['question']}")
            context_parts.append(f"A{i}: {interaction['answer'][:100]}...")

        return "\n".join(context_parts)


class EnhancedRAGChatbot:
    """Chatbot RAG được cải tiến với nhiều tính năng mới"""

    def __init__(
        self,
        embedded_jsonl_path: Optional[str | List[str]] = None,
        search_all_jsonl: bool = False,
        jsonl_dir: Optional[str] = None,
        qa_source_name: Optional[
            str
        ] = None,  # Thay đổi từ qa_file_path thành qa_source_name
        similarity_threshold: float = 0.75,
    ):
        # embedded_jsonl_path can be a single path (str) or list of paths
        # If search_all_jsonl is True, load all .jsonl files in the directory
        if search_all_jsonl:
            if jsonl_dir is None:
                # Default to data/processed if not provided
                jsonl_dir = os.path.join(os.getcwd(), "data", "processed")
            self.embedded_path = self._find_all_jsonl_files(jsonl_dir)
        else:
            if embedded_jsonl_path is None:
                raise ValueError(
                    "embedded_jsonl_path must be provided if search_all_jsonl is False"
                )
            self.embedded_path = embedded_jsonl_path
        self.store = self._load_store()
        self.client = init_genai_client()
        self.session = ChatSession()

        # Load Q&A embeddings from database instead of file
        self.qa_source_name = qa_source_name
        self.similarity_threshold = similarity_threshold

    # Caching is no longer needed because we query the database directly
        self.qa_embeddings_cache = None
        self.qa_data_cache = None

    # Calling _load_qa_embeddings() is no longer necessary

        # Import configuration
        from ragbot.config import config

        self.config = {
            "top_k": config.DEFAULT_TOP_K,
            "min_score": config.DEFAULT_MIN_SCORE,  # Giảm xuống để tìm được nhiều tài liệu hơn
            "max_answer_length": config.MAX_ANSWER_LENGTH,
            "show_sources": True,
            "use_context": True,
            "debug_mode": False,  # Thêm chế độ debug
        }

    @staticmethod
    def _find_all_jsonl_files(directory: str) -> list:
        """Tìm tất cả file .jsonl trong thư mục (không lấy file _embedded nếu đã có file gốc)"""
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".jsonl") and not file.endswith("_embedded.jsonl"):
                    all_files.append(os.path.join(root, file))
        # Also include _embedded.jsonl files if present
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("_embedded.jsonl"):
                    all_files.append(os.path.join(root, file))
        return all_files

    def _load_store(self) -> SimpleVectorStore:
        """Tải vector store từ file"""
        try:
            # If user provided a list of paths, load from multiple files
            if isinstance(self.embedded_path, (list, tuple)):
                store = SimpleVectorStore.from_jsonl_paths(self.embedded_path)
            else:
                store = SimpleVectorStore.from_jsonl(self.embedded_path)
            if len(store) == 0:
                raise ValueError("No embedding data found in the file")
            return store
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải vector store: {e}")

    def _find_similar_question_optimized(self, question: str) -> Optional[str]:
        """Tìm câu hỏi tương tự sử dụng database search (thay vì cache trong memory)"""
        if not self.qa_source_name:
            return None

        try:
            # Import VectorSearchService
            from ragbot.retrieval.vector_search import VectorSearchService

            # Tạo embedding cho câu hỏi
            query_embedding = embed_query(self.client, question)

            # Tìm kiếm Q&A chunks từ database
            qa_results = VectorSearchService.search_qa_chunks(
                embedding=query_embedding,
                limit=5,  # Lấy top 5 kết quả
                min_score=self.similarity_threshold,
                qa_source_name=self.qa_source_name,
            )

            if not qa_results:
                print(
                    f"🔍 No similar Q&A found (threshold: {self.similarity_threshold})"
                )
                return None

            # Lấy kết quả tốt nhất
            best_result = qa_results[0]
            best_similarity = best_result["score"]

            print(
                f"🔍 Best Q&A similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})"
            )

            if best_similarity > self.similarity_threshold:
                # Trích xuất answer từ content
                content = best_result["content"]
                # Content format: "Question: {question}\nAnswer: {answer}"
                if "Answer:" in content:
                    answer = content.split("Answer:", 1)[1].strip()
                    return answer
                else:
                    # Nếu không có format chuẩn, trả về toàn bộ content
                    return content
            else:
                return None

        except Exception as e:
            print(f"⚠️  Error in database Q&A search: {e}")
            return None

    def ask_question(self, question: str) -> None:
        """Process a question, ưu tiên trả lời từ bộ Q&A nếu có"""
        if not self.client:
            print("❌ Cannot answer: AI not connected")
            print("   Please check GOOGLE_API_KEY or GOOGLE_CLOUD_* configuration")
            print("   Or use !search for document search only")
            return

        try:
            # Kiểm tra độ dài câu hỏi
            if len(question.strip()) < 3:
                print("❌ Question is too short. Please ask a more detailed question.")
                return

            # --- Q&A similarity check ---
            # Nếu có Q&A source, ưu tiên trả lời nếu query giống câu hỏi mẫu
            qa_source_name = getattr(self, "qa_source_name", None)
            found_qa_answer = False

            if qa_source_name:
                try:
                    # Sử dụng database search thay vì cache
                    similar_answer = self._find_similar_question_optimized(question)

                    if similar_answer:
                        print(
                            "✅ Found similar question in Q&A data. Returning direct answer:"
                        )
                        self._format_answer(similar_answer, [])
                        self.session.add_interaction(question, similar_answer, [])
                        found_qa_answer = True
                except Exception as e:
                    print(f"⚠️  Q&A similarity check error: {e}")

            if found_qa_answer:
                return

            # Tìm kiếm tài liệu liên quan
            print("Searching for relevant documents...")
            # build_prompt, generate_answer, retrieve are defined in this file

            sources = retrieve(
                self.store,
                self.client,
                question,
                top_k=self.config["top_k"],
                min_score=self.config["min_score"],
                use_metadata_ranking=self.config.get("use_smart_ranking", True),
            )

            if not sources:
                print("❌ No documents found related to your question")
                print(
                    "Tip: Try a different question or lower min_score using !set min_score 0.05"
                )

                # Thử tìm với min_score thấp hơn nếu chế độ debug bật
                if self.config.get("debug_mode", False):
                    print("\n🐛 DEBUG: Trying search with min_score=0.01...")
                    debug_sources = retrieve(
                        self.store,
                        self.client,
                        question,
                        top_k=self.config["top_k"],
                        min_score=0.01,
                        use_metadata_ranking=self.config.get("use_smart_ranking", True),
                    )
                    if debug_sources:
                        print(f"🐛 Found {len(debug_sources)} results with low score:")
                        for i, src in enumerate(debug_sources[:3], 1):
                            meta = src.meta or {}
                            print(
                                f"   [{i}] Score: {src.score:.4f} | {meta.get('pdf_name', 'Unknown')} (p.{meta.get('page', '?')})"
                            )

                return

            print(f"Found {len(sources)} related documents")

            # Hiển thị thông tin debug nếu bật
            if self.config.get("debug_mode", False):
                print("\n🐛 DEBUG - SEARCH RESULTS:")
                for i, src in enumerate(sources, 1):
                    meta = src.meta or {}
                    pdf_name = meta.get("pdf_name", "Unknown")
                    page = meta.get("page", "?")
                    content_preview = (
                        src.content[:100] + "..."
                        if len(src.content) > 100
                        else src.content
                    )
                    print(f"   [{i}] Score: {src.score:.4f} | {pdf_name} (p.{page})")
                    print(f"       Content: {content_preview}")
                print("🐛 ----")

            # Tạo prompt với ngữ cảnh
            context = ""
            if self.config["use_context"] and self.session.history:
                context = self.session.get_recent_context(2)

            prompt = build_prompt(question, sources, context)

            # Sinh câu trả lời
            print("🧠 Generating answer...")
            answer = generate_answer(self.client, prompt)

            if not answer or answer.strip() == "":
                print("❌ Could not generate an answer. Please try again.")
                return

            # Hiển thị kết quả
            self._format_answer(answer, sources)

            # Lưu vào lịch sử
            self.session.add_interaction(question, answer, sources)

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            print("Please try again or contact an administrator")
            self._save_session()
            return

    def _save_session(self):
        """Save chat session"""
        try:
            filepath = self.session.save_session()
            print(f"💾 Session saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")

    def _format_answer(self, answer: str, sources: List[RetrievalResult]):
        """Format and display the answer with sources"""
        print(f"\n🤖 Answer: {answer}")

        if sources and self.config["show_sources"]:
            print(f"\n📚 Sources ({len(sources)}):")
            for i, src in enumerate(sources, 1):
                meta = src.meta or {}
                page = meta.get("page", "?")
                print(
                    f"   [{i}] {src.get_clean_pdf_name()} (p.{page}) - Score: {src.score:.3f}"
                )
                if self.config.get("debug_mode", False):
                    content_preview = (
                        src.content[:100] + "..."
                        if len(src.content) > 100
                        else src.content
                    )
                    print(f"       Content: {content_preview}")
        print()
