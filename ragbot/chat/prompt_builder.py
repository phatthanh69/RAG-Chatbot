"""Prompt construction helpers for chat orchestration.

These are near-pure: they read their inputs and emit strings. The only side
effect is logging, which is injected (defaults to this module's logger) so the
functions can be used standalone or delegated to from ChatbotService with its
own logger to preserve behavior.
"""

import logging
from typing import Dict, List

_LOGGER = logging.getLogger(__name__)


def build_prompt(question: str, sources: List, context: str, logger=None) -> str:
    """
    Build prompt for the AI model

    Args:
        question: User question
        sources: Retrieved sources
        context: Recent conversation context

    Returns:
        Formatted prompt
    """
    log = logger or _LOGGER
    # Combine source content - handle both RetrievalResult objects and dictionaries
    source_content_parts = []
    for i, src in enumerate(sources):
        try:
            if hasattr(src, "content"):
                # RetrievalResult object
                content = src.content
            elif isinstance(src, dict):
                # Dictionary
                content = src.get("content", "")
            else:
                content = str(src)
                log.debug(
                    f"Source {i} content from str conversion: {len(content)} chars"
                )

            source_content_parts.append(f"Source {i+1}: {content}")
        except Exception as e:
            log.error(f"Error processing source {i} for prompt: {str(e)}")
            log.error(f"Source {i} details: {src}")
            import traceback

            log.error(f"Traceback: {traceback.format_exc()}")
            source_content_parts.append(f"Source {i+1}: [Error processing content]")

    source_content = "\n\n".join(source_content_parts)
    sys_instr = (
        "Bạn là một trợ lý AI chuyên nghiệp và thân thiện,chuyên tư vấn các sản phẩm và giải pháp của REECOTECH. "
        "LƯU Ý QUAN TRỌNG: Chỉ sử dụng thông tin có trong tài liệu. "
        "KHÔNG được bịa đặt hoặc bổ sung kiến thức ngoài tài liệu."
        "Trả lời trực tiếp, KHÔNG viết lời dẫn, KHÔNG nhắc lại quy tắc."
        "Tuyệt đối không được sử dụng các cụm từ: Dựa trên thông tin được cung cấp, Dựa trên các tài liệu, v.v."
    )

    context_section = ""
    if context:
        context_section = f"\nNgữ cảnh hội thoại trước đó:\n{context}\n"

    # NOTE: content lines are intentionally indented 12 spaces to preserve the
    # exact prompt string (including leading/trailing whitespace) emitted by the
    # original ChatbotService._build_prompt.
    prompt = f"""
            {sys_instr}
            {context_section}
            Câu hỏi của người dùng: {question}

            Ngữ cảnh tài liệu:
            {source_content}

            Nguyên tắc trả lời:
            - Trả lời ngắn gọn, có cấu trúc (khoảng 10-15 câu nếu đủ thông tin hữu ích)
            - Ưu tiên nội dung quan trọng nếu có quá nhiều thông tin
            - Sử dụng gạch đầu dòng hoặc đánh số nếu phù hợp
            - Nếu thiếu thông tin, hãy nói: "Rất tiếc, tôi hiện tại không thể trả lời câu hỏi này.Xin vui lòng liên hệ email:info@reecotech.com.vn;Hotline:Phòng Kinh doanh:0938 696 131 và Phòng Kỹ thuật:0901 880 386 để được hỗ trợ thêm."
            - Đảm bảo tính chính xác và trung thực trong câu trả lời không có thông tin thừa(không cần thiết)
            """
    return prompt


def get_recent_context(history: List[Dict], last_n: int = 3, logger=None) -> str:
    """
    Get recent conversation context

    Args:
        history: Session history (list of ChatMessage.to_dict())
        last_n: Number of recent interactions

    Returns:
        Formatted context string
    """
    log = logger or _LOGGER
    if not history or last_n <= 0:
        return ""

    # Take the last N*2 messages (since we have separate user/bot messages)
    recent = history[-(last_n * 2) :]
    context_parts = []

    # Group messages by pairs (user question + bot answer)
    for i in range(0, len(recent), 2):
        try:
            if i < len(recent):
                user_msg = recent[i]
                # Check if this is a user message
                if user_msg.get("message_type") == "user":
                    question = user_msg.get("content", "")
                    context_parts.append(f"Q: {question}")

                    # Check if there's a corresponding bot response
                    if i + 1 < len(recent):
                        bot_msg = recent[i + 1]
                        if bot_msg.get("message_type") == "bot":
                            answer = bot_msg.get("content", "")
                            context_parts.append(f"A: {answer[:100]}...")
                elif user_msg.get("message_type") == "bot":
                    # If we start with a bot message, just add it as an answer
                    answer = user_msg.get("content", "")
                    context_parts.append(f"A: {answer[:100]}...")
        except Exception as e:
            log.warning(f"Error processing context message {i}: {str(e)}")
            continue

    return "\n".join(context_parts)
