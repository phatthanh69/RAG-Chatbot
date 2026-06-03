"""Question understanding for the BAS chatbot.

`QuestionClassifier` owns the LLM-backed question taxonomy classifier, the
heading-first analysis/rewrite step, and the rule-based fallback. The bodies are
moved verbatim from `ChatbotService`; collaborators that used to be reached via
`self.<x>` are injected so the class is independently testable:

  - genai_client_provider() -> client   (was self._get_genai_client)
  - heading_provider(limit=...) -> dict  (was self._get_heading_context)
  - logger                                (was self.logger)
  - log_classification_result/log_heading_analysis/log_error_concise callbacks
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from ragbot.chat.rag_engine import generate_answer
from ragbot.utils.text import dedupe_preserve_order


class QuestionClassifier:
    """Classifies questions and performs heading-first analysis/rewrite."""

    def __init__(
        self,
        genai_client_provider=None,
        heading_provider=None,
        logger=None,
        log_classification_result=None,
        log_heading_analysis=None,
        log_error_concise=None,
    ):
        self._get_genai_client = genai_client_provider or (lambda: None)
        self._get_heading_context = heading_provider or (lambda **k: {})
        self.logger = logger or logging.getLogger(__name__)
        self._log_classification_result = log_classification_result or (
            lambda *a, **k: None
        )
        self._log_heading_analysis = log_heading_analysis or (lambda *a, **k: None)
        self._log_error_concise = log_error_concise or (lambda *a, **k: None)

    def classify(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Bước 1: Sử dụng LLM để phân loại câu hỏi theo taxonomy

        Args:
            question: Câu hỏi của người dùng
            session_data: Dữ liệu session hiện tại (optional)

        Returns:
            Dict chứa loại câu hỏi, confidence và phản hồi (nếu có)
        """
        try:
            client = self._get_genai_client()
            if not client:
                # Fallback classification
                self.logger.warning(
                    "No genai client available, using fallback classification"
                )
                return {
                    "question_type": "BlueEco_BAS",
                    "confidence": 0.5,
                    "response": "Đang tìm kiếm thông tin...",
                    "needs_vector_search": True,
                    "needs_qa_check": True,
                }

            # Heading-first: lấy danh sách heading để cung cấp ngữ cảnh nhẹ
            heading_ctx = self._get_heading_context(limit=200)
            titles_preview = ", ".join(heading_ctx.get("titles", [])[:50]) or "None"

            # Tạo prompt ngắn gọn cho phân loại câu hỏi (heading-first)
            classification_prompt = f"""
            Bạn là một chatbot thông minh chuyên tư vấn về giải pháp BAS (Berthing Aid System) của REECOTECH.
Hãy phân loại câu hỏi một cách khách quan
HEADINGS CÓ SẴN (chỉ để tham khảo):
{titles_preview}

CÂU HỎI: "{question}"

QUY TẮC PHÂN LOẠI:
- greeting: Lời chào, hỏi thăm, xã giao (ví dụ: "chào", "hello", "bạn là ai")
- BlueEco_BAS: Câu hỏi trực tiếp về sản phẩm BAS, cảm biến, thiết bị của REECOTECH (ví dụ: "cảm biến laser là gì", "giá LS-BE-001")
- off_topic: Câu hỏi không liên quan đến BAS hoặc công ty (ví dụ: "thời tiết hôm nay", "bóng đá")
- quotation: Yêu cầu báo giá, mua hàng, liên hệ kinh doanh (ví dụ: "giá bao nhiêu", "muốn mua")
- capability: Hỏi về khả năng của chatbot (ví dụ: "bạn có thể làm gì", "bạn biết gì")
- company_info: Thông tin về công ty REECOTECH (ví dụ: "công ty ở đâu", "liên hệ")

PHẢN HỒI TƯƠNG ỨNG:
** ĐẢM BẢO PHẢN HỒI PHÙ HỢP CÓ CHỦ NGỮ,VỊ NGỮ ĐẦY ĐỦ, KHÔNG CẮT NGẮT GIỮA CHỪNG**
- greeting: Chào hỏi thân thiện phù hợp với lời chào, đồng thời giới thiệu REECOTECH
- BlueEco_BAS: "" (để trống)
- off_topic: Lịch sự chuyển hướng về BAS
- quotation: "Để nhận được báo giá chi tiết xin vui lòng liên hệ: info@reecotech.com.vn. Phòng Kinh doanh:0938 696 131. Phòng Kỹ thuật:0901 880 386". Đồng thời gợi mở khách hàng hỏi thêm về sản phẩm.
- capability: Mô tả khả năng tư vấn BAS
- company_info: "" (để trống)
TRẢ VỀ JSON:
{{
    "question_type": "greeting|BlueEco_BAS|off_topic|quotation|capability|company_info",
    "confidence": 0.0-1.0,
    "response": "phản hồi tương ứng hoặc để trống",
    "needs_vector_search": true|false,
    "needs_qa_check": true|false
}}
CHỈ TRẢ VỀ JSON, KHÔNG THÊM VĂN BẢN KHÁC.
"""

            # Gọi LLM để phân loại câu hỏi
            response = generate_answer(client, classification_prompt)

            # Parse JSON response
            response_text = ""
            if isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()

            # Extract JSON từ response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                classification_result = json.loads(json_str)

                # Validate required fields for question type classification only
                required_fields = [
                    "question_type",
                    "confidence",
                    "response",
                    "needs_vector_search",
                    "needs_qa_check",
                ]

                if all(field in classification_result for field in required_fields):
                    # Log classification result using helper method
                    self._log_classification_result(classification_result)
                    return classification_result
                else:
                    self.logger.warning(
                        f"Missing required fields in classification: {classification_result}"
                    )
                    raise ValueError("Invalid classification response format")
            else:
                self.logger.warning(f"No JSON found in LLM response: {response_text}")
                raise ValueError("No valid JSON in classification response")
        except Exception as e:
            self._log_error_concise("Question type classification", e)
            # Fallback response
            return {
                "question_type": "BlueEco_BAS",
                "confidence": 0.5,
                "response": "Đang tìm kiếm thông tin...",
                "needs_vector_search": True,
                "needs_qa_check": True,
            }

    def analyze_and_rewrite(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heading-first analysis: choose active heading_title and optionally rewrite question."""
        try:
            client = self._get_genai_client()
            # Check if API key is available (force fallback if not)

            if not client or not os.getenv("GOOGLE_API_KEY"):
                self.logger.warning(
                    "No Google API key available, using fallback heading analysis"
                )
                current_active_headings = []
                if session_data and isinstance(session_data, dict):
                    session_metadata = session_data.get("metadata", {})
                    current_headings = session_metadata.get("active_headings") or []
                    if isinstance(current_headings, list):
                        current_active_headings = current_headings
                    elif current_headings:
                        current_active_headings = [current_headings]
                return {
                    "heading_info": {
                        "active_headings": current_active_headings,
                        "context_maintained": True,
                        "confidence": 0.3,
                    },
                    "rewritten_question": None,
                }

            # Collect session context
            current_active_headings: List[str] = []
            history_context = ""
            recent_questions: List[str] = []
            recent_headings: List[str] = []

            if session_data and isinstance(session_data, dict):
                session_metadata = session_data.get("metadata", {})
                current_active_headings = session_metadata.get("active_headings") or []

                session_history = session_metadata.get("history", [])
                for i in range(min(3, len(session_history))):
                    hist_item = session_history[-(i + 1)]
                    if isinstance(hist_item, dict):
                        q = hist_item.get("question", "")
                        h = hist_item.get("active_headings") or hist_item.get(
                            "active_heading_title"
                        )
                        if q:
                            recent_questions.append(q)
                        if h:
                            if isinstance(h, list):
                                recent_headings.extend(h)
                            else:
                                recent_headings.append(h)

                if "history" in session_data and session_data["history"]:
                    recent_history = session_data["history"][-3:]
                    history_context = "\n".join(
                        [
                            f"- {msg.get('message_type', 'user')}: {msg.get('content', '')[:100]}"
                            for msg in recent_history
                        ]
                    )

            # DB heading context
            heading_ctx = self._get_heading_context(limit=300)
            raw_headings = heading_ctx.get("titles", []) or []

            valid_headings = dedupe_preserve_order(raw_headings)
            valid_headings_str = ", ".join(valid_headings) if valid_headings else "None"

            heading_analysis_prompt = f"""
Phân tích heading phù hợp và viết lại câu hỏi nếu cần (heading-first).

CÂU HỎI HIỆN TẠI: "{question}"
ACTIVE HEADINGS HIỆN TẠI: {', '.join(current_active_headings) if current_active_headings else "None"}

=== LỊCH SỬ SESSION ===
RECENT QUESTIONS: {', '.join(recent_questions[-3:]) if recent_questions else "None"}
RECENT HEADINGS: {', '.join(set(recent_headings)) if recent_headings else "None"}

LỊCH SỬ: {history_context or "None"}

=== THÔNG TIN TỪ DATABASE (CHỈ ĐỊNH) ===
VALID HEADINGS: {valid_headings_str}

QUY TẮC (heading-first):
    - Ưu tiên exact match theo tiêu đề (heading_title) có trong VALID HEADINGS.
- Nếu câu hỏi liên quan nhiều headings, trả về list các headings phù hợp TỪ VALID HEADINGS.
- **ĐẶC BIỆT CHO CÂU HỎI VỀ CẢM BIẾN**: Nếu câu hỏi hỏi về "cảm biến" hoặc "loại cảm biến" mà không chỉ định loại cụ thể, chọn TẤT CẢ headings liên quan đến cảm biến từ VALID HEADINGS.
- Nếu câu hỏi không nêu rõ heading mới, GIỮ NGUYÊN active_headings hiện tại (context_maintained=true).
- KHÔNG được tự tạo heading mới - chỉ chọn từ danh sách VALID HEADINGS phía trên.
- Không tự động chuyển về heading tổng quát nếu không nêu rõ.

QUY TẮC CHỌN TỪ DB (BẮT BUỘC):
- active_headings phải thuộc VALID HEADINGS phía trên
- Nếu câu hỏi KHÔNG nêu rõ heading mới và active_headings hiện tại là một tiêu đề cụ thể (ví dụ LS-BE-001), GIỮ NGUYÊN heading hiện tại (context_maintained=true).
- KHÔNG được chuyển từ heading cụ thể ví dụ: LS-BE-001, WTX536... sang heading tổng quát (BAS) trừ khi câu hỏi nêu rõ "BAS" hoặc "bas".

QUY TẮC ƯU TIÊN (theo thứ tự):

1. HEADING TITLE MATCHING - ƯU TIÊN CAO NHẤT:
    - Nếu câu hỏi chứa TIÊU ĐỀ HEADING CHÍNH XÁC có trong VALID HEADINGS: chọn đúng tiêu đề đó.
    - Chuẩn hoá mã LS-BE: "LS-BE-001", "LS-Be-001", "ls be 001" → "LS-BE-001" (chỉ nếu có trong VALID HEADINGS).
    - "BAS", "bas" → chỉ chọn nếu câu hỏi nêu rõ.
    - KHÔNG match chéo: "LS-BE-001" ≠ "BAS", "BAS" ≠ "LS-BE-001".
    - Ưu tiên exact match, hạn chế fuzzy.

2. CONTEXT REFERENCES - ƯU TIÊN CAO:
    - "cảm biến vừa rồi", "thiết bị này", "nó", "này", "đó" → GIỮ NGUYÊN active_headings
    - "tiêu chuẩn an toàn của [heading hiện tại]" → GIỮ NGUYÊN active_headings
    - "thông số kỹ thuật", "cách lắp đặt", "giá cả" → thuộc tính của active_headings

4. AMBIGUOUS CASES:
    - Nếu không chắc chắn → GIỮ NGUYÊN active_headings, context_maintained=true.
    - Nếu câu hỏi quá chung chung (ví dụ: "cảm biến,thiết bị...") KHÔNG ĐƯỢC chọn heading mới.Trả về None hoặc giữ nguyên active_headings.
    - Ưu tiên ngữ cảnh cuộc hội thoại hơn heading mới.
    - Không đẩy về BAS nếu câu hỏi không nêu rõ.

VÍ DỤ CHÍNH XÁC:
✅ "LS-BE-001" → active_headings=["LS-BE-001"], context_maintained=false  
✅ "tiêu chuẩn an toàn của cảm biến vừa rồi" → active_headings=["LS-BE-001"], context_maintained=true  
✅ "thông số kỹ thuật của nó" → active_headings=["LS-BE-001"], context_maintained=true  
✅ "có BAS không?" → active_headings=["BAS"], context_maintained=false  
✅ "so sánh LS-BE-001 và WTX536" → active_headings=["LS-BE-001", "WTX536"], context_maintained=false
✅ "VnEmisoft (Cloud)" → active_headings=["VnEmisoft (Cloud)"], context_maintained=false
✅ "BAS xài cảm biến gì" → active_headings=["Cảm biến laser (Laser Sensors)", "Cảm biến khí tượng – thủy văn (Meteorological & Hydrological Sensors)"], context_maintained=false

VÍ DỤ SAI (KHÔNG CHỌN):
❌ "thiết bị đo khoảng cách" → KHÔNG có trong VALID HEADINGS
❌ "LS-BE-001" → KHÔNG được match thành "BAS"

VÍ DỤ SAI:
❌ "LS-Be-001" → active_heading="BE-BAS-01" (SAI - phải là LS-BE-001)

TRẢ VỀ JSON:
{{
  "heading_info": {{
    "active_headings": ["heading1", "heading2"] | null,
    "context_maintained": true|false,
    "confidence": 0.0-1.0
  }},
  "rewritten_question": "câu hỏi đã viết lại" | null
}}
CHỈ TRẢ VỀ JSON, KHÔNG THÊM VĂN BẢN KHÁC.
"""

            response = generate_answer(client, heading_analysis_prompt)

            response_text = (
                response.strip() if isinstance(response, str) else str(response).strip()
            )
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                required_fields = ["heading_info", "rewritten_question"]
                if all(field in result for field in required_fields):
                    heading_info = result.get("heading_info", {}) or {}
                    # Log heading analysis result using helper method
                    self._log_heading_analysis(
                        heading_info, heading_info.get("context_maintained", True)
                    )
                    return {
                        "heading_info": heading_info,
                        "rewritten_question": result.get("rewritten_question"),
                    }
                else:
                    self.logger.warning(
                        f"Missing required fields in heading analysis: {result}"
                    )
                    raise ValueError("Invalid heading analysis response format")
            else:
                self.logger.warning(
                    f"No JSON found in heading response: {response_text}"
                )
                raise ValueError("No valid JSON in heading analysis response")

        except Exception as e:
            self._log_error_concise("Heading analysis", e)
            current_active_headings = []
            if session_data and isinstance(session_data, dict):
                session_metadata = session_data.get("metadata", {})
                current_headings = session_metadata.get("active_headings") or []
                if isinstance(current_headings, list):
                    current_active_headings = current_headings
                elif current_headings:
                    current_active_headings = [current_headings]
            return {
                "heading_info": {
                    "active_headings": current_active_headings,
                    "context_maintained": True,
                    "confidence": 0.3,
                },
                "rewritten_question": None,
            }

    def fallback_classify(
        self, question: str, session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback classification khi LLM không khả dụng (heading-first)

            Args:
                question: Câu hỏi của người dùng
                session_data: Dữ liệu session hiện tại

            Returns:
                Dict chứa phân loại, heading info và phản hồi
        """
        try:
            # Get current active headings from session
            current_active_headings = []
            if session_data and isinstance(session_data, dict):
                session_metadata = session_data.get("metadata", {})
                current_headings = session_metadata.get("active_headings") or []
                if isinstance(current_headings, list):
                    current_active_headings = current_headings
                elif current_headings:
                    current_active_headings = [current_headings]

            # Simple fallback: assume BlueEco_BAS question if LLM fails
            result = {
                "question_type": "BlueEco_BAS",
                "confidence": 0.5,
                "response": "Đang tìm kiếm thông tin...",  # Short response for DB query cases
                "needs_vector_search": True,
                "needs_qa_check": True,
                "rewritten_question": question,  # Giữ nguyên câu hỏi gốc trong fallback
                "heading_info": {
                    "active_headings": current_active_headings,
                    "context_maintained": True,
                    "confidence": 0.3,
                },
            }
            return result

        except Exception as e:
            self.logger.error(f"Error in fallback classification: {str(e)}")
            result = {
                "question_type": "BlueEco_BAS",
                "confidence": 0.5,
                "response": "Đang tìm kiếm thông tin...",  # Short response for DB query cases
                "needs_vector_search": True,
                "needs_qa_check": True,
                "rewritten_question": question,  # Giữ nguyên câu hỏi gốc trong fallback error
                "heading_info": {
                    "active_headings": [],
                    "context_maintained": False,
                    "confidence": 0.0,
                },
            }
            self.logger.info(f"Fallback classification result (error): {result}")
            return result
