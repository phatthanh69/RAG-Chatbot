import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Timezone configuration for Vietnam (UTC+7)
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def normalize_text(text):
    """Chuẩn hóa văn bản: lowercase, clean whitespace, normalize units"""
    if not text or not text.strip():
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFC", text)

    # Chuyển chữ hoa → chữ thường
    text = text.lower()

    # Thay ký tự thừa, newline, tab
    text = re.sub(r"\s+", " ", text)

    # Chuẩn hóa các đơn vị kỹ thuật
    patterns = {
        r"(\d+)\s*k\s*w": r"\1 kW",
        r"(\d+)\s*r\s*p\s*m": r"\1 rpm",
        r"(\d+)\s*v(?:\s|$)": r"\1 V ",
        r"(\d+)\s*a(?:\s|$)": r"\1 A ",
        r"(\d+)\s*hz": r"\1 Hz",
        r"(\d+)\s*k\s*va": r"\1 kVA",
        r"(\d+)\s*Ω": r"\1 Ω",
        r"(\d+)\s*°\s*c": r"\1°C",
        r"(\d+)\s*m\s*m": r"\1 mm",
        r"(\d+)\s*k\s*g": r"\1 kg",
        r"(\d+)\s*l(?:\s|$)": r"\1 L ",
        r"(\d+)\s*m(?:\s|$)": r"\1 m ",
    }

    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_jsonl_file(file_path: Path) -> List[str]:
    """Đọc file JSONL và trích xuất nội dung"""
    documents = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse JSON từ mỗi dòng
                    data = json.loads(line)

                    # Trích xuất content từ JSON
                    if isinstance(data, dict) and "content" in data:
                        content = data["content"]
                        if content and content.strip():
                            normalized_content = normalize_text(content)
                            documents.append(normalized_content)

                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing line {line_num}: {e}")
                    continue

        logging.info(f"Loaded {len(documents)} documents from JSONL")
        return documents

    except Exception as e:
        logging.error(f"Error reading JSONL file: {e}")
        raise


def save_processed_data(documents: List[str], output_dir: Optional[str] = None):
    """Lưu trữ kết quả đã xử lý"""
    if output_dir is None:
        from src.config import paths

        output_dir = str(paths.OUTPUT_DIR)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Lưu dưới dạng JSON
    json_file = output_path / f"cleaned_documents_{timestamp}.json"
    data_to_save = {
        "timestamp": datetime.now(VIETNAM_TIMEZONE).isoformat(),
        "total_documents": len(documents),
        "documents": documents,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    # Lưu dưới dạng text thuần
    txt_file = output_path / f"cleaned_documents_{timestamp}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents, 1):
            f.write(f"=== Document {i} ===\n")
            f.write(doc)
            f.write("\n\n")

    logging.info(f"Saved {len(documents)} documents")
    logging.info(f"JSON: {json_file}")
    logging.info(f"TXT: {txt_file}")

    return json_file, txt_file
