import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.ingestion.cleaner import normalize_text
from ragbot.config import config, paths
from ragbot.ingestion.extractor import (
    extract_document_blocks,
    filter_layout_noise,
    get_document_files,
)


def detect_qa_pairs(text):
    """
    Phát hiện các bộ câu hỏi và trả lời trong văn bản.
    Xử lý trường hợp câu hỏi có thể bị xuống dòng.
    """
    qa_pairs = []

    # Chuẩn hóa văn bản: thay thế tab bằng space, loại bỏ khoảng trắng dư thừa
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" +", " ", text)

    # Tách văn bản thành các dòng để xử lý
    lines = text.split("\n")

    # Tìm các dòng bắt đầu bằng số thứ tự (1., 2., 3., ...)
    question_starts = []
    for i, line in enumerate(lines):
        line = line.strip()
        # Tìm dòng bắt đầu bằng số và dấu chấm
        if re.match(r"^\d+\.\s+", line):
            question_starts.append(i)

    if not question_starts:
        return qa_pairs

    # Xử lý từng cặp Q&A
    for idx, start_line in enumerate(question_starts):
        # Xác định dòng kết thúc của Q&A này (trước Q&A tiếp theo hoặc cuối file)
        if idx + 1 < len(question_starts):
            end_line = question_starts[idx + 1]
        else:
            end_line = len(lines)

        # Lấy toàn bộ nội dung của Q&A pair này
        qa_lines = lines[start_line:end_line]
        qa_text = "\n".join(qa_lines).strip()

        if not qa_text:
            continue

        # Tìm dấu hỏi đầu tiên để phân tách question và answer
        question_end = qa_text.find("?")
        if question_end == -1:
            continue  # No question mark found

        # Phân tách question và answer
        question = qa_text[: question_end + 1].strip()
        answer = qa_text[question_end + 1 :].strip()

        # Chuẩn hóa question và answer
        question = re.sub(r"\s+", " ", question)
        answer = re.sub(r"\s+", " ", answer)

        # Kiểm tra tính hợp lệ
        if len(question) > 10 and len(answer) > 5:
            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "full_text": question + "\n" + answer,
                }
            )

    return qa_pairs


def split_bullet_points(text):
    """
    Tách các bullet points thành chunks riêng biệt.
    Phát hiện các dòng bắt đầu bằng ký tự bullet (•, -, etc.) và tách thành chunks riêng.
    """
    import re

    # First, split by newlines to handle multi-line bullet points
    lines = text.split("\n")
    chunks = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if line contains multiple bullet points separated by " . - "
        # Pattern: " . - " indicates bullet points on the same line
        if " . - " in line_stripped or line_stripped.count(" - ") > 1:
            # Split by " . - " pattern to separate bullet points
            parts = re.split(r"\.\s*-\s*", line_stripped)
            for part in parts:
                part = part.strip()
                if part and (
                    part.startswith(("-", "•"))
                    or part.startswith(
                        (
                            "dải",
                            "độ",
                            "tần",
                            "bước",
                            "cấp",
                            "ngõ",
                            "nhiệt",
                            "giao",
                            "nguồn",
                            "nguyên",
                            "tiêu",
                            "kích",
                        )
                    )
                ):
                    # Add back the bullet if it was removed by split
                    if not part.startswith(("-", "•")):
                        part = "- " + part
                    chunks.append(part)
        elif line_stripped.startswith(("•", "-")) and len(line_stripped) > 1:
            chunks.append(line_stripped)
        elif line_stripped and not line_stripped.startswith(("•", "-")):
            # Dòng không phải bullet nhưng có nội dung
            chunks.append(line_stripped)

    return chunks


def split_text_by_qa_priority(text, chunk_size=1024, chunk_overlap=128):
    """
    Chia văn bản ưu tiên theo bộ câu hỏi-trả lời, sau đó mới dùng RecursiveCharacterTextSplitter.
    """
    # Phát hiện bộ Q&A
    qa_pairs = detect_qa_pairs(text)

    if qa_pairs:
        # Nếu tìm thấy Q&A pairs, ưu tiên cắt theo từng cặp
        chunks = []
        for qa in qa_pairs:
            qa_text = qa["full_text"]

            # Nếu Q&A quá dài, chia nhỏ thêm
            if len(qa_text) > chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", ".\n", " ", ""],
                )
                sub_chunks = splitter.split_text(qa_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(qa_text)

        return chunks, True  # True indicates QA pairs were found
    else:
        # Tách bullet points trước khi chunking thông thường
        bullet_chunks = split_bullet_points(text)

        # Với mỗi bullet chunk, kiểm tra độ dài và chunk nếu cần
        final_chunks = []
        for chunk in bullet_chunks:
            if len(chunk) > chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", ".\n", " ", ""],
                )
                sub_chunks = splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks, False  # False indicates no QA pairs found


def split_blocks_with_recursive(blocks, chunk_size=1024, chunk_overlap=128):
    """
    Chia từng block ưu tiên theo Q&A pairs, sau đó dùng RecursiveCharacterTextSplitter.
    Trả về danh sách chunk có metadata (pdf_name/page/bbox/block_index).
    """
    out_chunks = []
    for b in blocks:
        # Ưu tiên cắt theo Q&A pairs trước
        parts, has_qa = split_text_by_qa_priority(
            b["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for j, part in enumerate(parts):
            if not part.strip():
                continue
            text_part = part.strip()

            # Thêm metadata về việc có Q&A pairs hay không
            chunk_meta = {
                "page": b["page"],
                "block_index": b["block_index"],
                "bbox": b["bbox"],
                "page_width": b["w"],
                "page_height": b["h"],
                "chunk_in_block": j,
                # Hiển thị: lấy kích thước chữ từ block (nếu có)
                "font_size": b.get("font_size"),
                "font_sizes": b.get("font_sizes", []),
                # Section info: block heading detected by extractor
                "is_heading": b.get("is_heading", False),
                # New hierarchical heading fields
                "heading_id": b.get("heading_id"),
                "heading_title": b.get("heading_title"),
                "heading_parent_id": b.get("heading_parent_id"),
                "heading_level": b.get("heading_level"),
                # Thông tin về Q&A
                "has_qa_pairs": has_qa,
                "chunk_type": "qa_pair" if has_qa else "regular",
            }

            out_chunks.append(
                {
                    "content": text_part,
                    "meta": chunk_meta,
                }
            )
    return out_chunks


def pdf_to_chunks_jsonl(
    input_path_str,
    output_dir_str=None,
    chunk_size=None,
    chunk_overlap=None,
):
    """
    - Đọc PDF (file hoặc thư mục).
    - Trích blocks (PyMuPDF) -> lọc noise -> chia chunk (RecursiveCharacterTextSplitter).
    - Ghi JSONL: mỗi dòng {content, meta:{...}}
    """
    # Use configuration values if not provided
    if output_dir_str is None:
        output_dir_str = str(paths.PROCESSED_DATA_DIR)
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)

    # Sử dụng hàm get_document_files từ extractor để lấy cả PDF và DOCX
    file_result = get_document_files(input_path_str)
    pdf_files = file_result["pdf_files"]
    docx_files = file_result["docx_files"]
    all_files = pdf_files + docx_files

    if file_result["errors"]:
        return {"success": [], "errors": file_result["errors"]}

    results = {"success": [], "errors": []}

    for file_path in all_files:
        try:
            blocks = extract_document_blocks(file_path)

            # Áp dụng filter khác nhau cho PDF và DOCX
            if file_path.suffix.lower() == ".pdf":
                blocks = filter_layout_noise(blocks)
            else:
                # Với DOCX, chỉ lọc những block trống hoặc quá ngắn
                blocks = [
                    b
                    for b in blocks
                    if b["text"].strip() and len(b["text"].strip()) > 3
                ]

            # Normalize text trong blocks trước khi chunking
            for block in blocks:
                block["text"] = normalize_text(block["text"])

            # Thử detect Q&A trên toàn bộ văn bản trước
            full_text = "\n".join([b["text"] for b in blocks])
            qa_pairs = detect_qa_pairs(full_text)

            if qa_pairs:
                # Nếu tìm thấy Q&A pairs, ưu tiên chunking theo Q&A
                print(f"Tìm thấy {len(qa_pairs)} Q&A pairs trong {file_path.name}")
                out_chunks = []

                for i, qa in enumerate(qa_pairs):
                    qa_text = qa["full_text"]

                    # Nếu Q&A quá dài, chia nhỏ thêm nhưng cố gắng giữ nguyên cấu trúc Q&A
                    if len(qa_text) > chunk_size:
                        # Thử chia tại các điểm ngắt tự nhiên trong câu trả lời
                        # nhưng vẫn giữ câu hỏi ở đầu chunk đầu tiên
                        question_part = qa["question"]
                        answer_part = qa["answer"]

                        # Chunk đầu tiên luôn chứa câu hỏi
                        if (
                            len(
                                question_part
                                + " "
                                + answer_part[: chunk_size - len(question_part) - 50]
                            )
                            <= chunk_size
                        ):
                            # Nếu có thể fit vào một chunk
                            chunk_meta = {
                                "page": 1,
                                "block_index": 0,
                                "bbox": [0, 0, 0, 0],
                                "page_width": 612.0,
                                "page_height": 792.0,
                                "chunk_in_block": i,
                                "font_size": None,
                                "font_sizes": [],
                                "is_heading": False,
                                # Hierarchy unknown in DOCX QA shortcut
                                "heading_id": None,
                                "heading_title": None,
                                "heading_parent_id": None,
                                "heading_level": None,
                                "has_qa_pairs": True,
                                "chunk_type": "qa_pair",
                            }
                            out_chunks.append({"content": qa_text, "meta": chunk_meta})
                        else:
                            # Chia nhỏ answer nhưng giữ question ở chunk đầu
                            first_chunk = (
                                question_part
                                + "\n"
                                + answer_part[: chunk_size - len(question_part) - 50]
                            )
                            chunk_meta = {
                                "page": 1,
                                "block_index": 0,
                                "bbox": [0, 0, 0, 0],
                                "page_width": 612.0,
                                "page_height": 792.0,
                                "chunk_in_block": i * 10,  # Avoid duplicates
                                "font_size": None,
                                "font_sizes": [],
                                "is_heading": False,
                                "heading_id": None,
                                "heading_title": None,
                                "heading_parent_id": None,
                                "heading_level": None,
                                "has_qa_pairs": True,
                                "chunk_type": "qa_pair",
                            }
                            out_chunks.append(
                                {"content": first_chunk, "meta": chunk_meta}
                            )

                            # Các chunk tiếp theo chỉ chứa phần answer còn lại
                            remaining_answer = answer_part[
                                chunk_size - len(question_part) - 50 :
                            ]
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                separators=["\n\n", "\n", ". ", ".\n", " ", ""],
                            )
                            sub_chunks = splitter.split_text(remaining_answer)

                            for j, sub_chunk in enumerate(sub_chunks):
                                sub_chunk_meta = {
                                    "page": 1,
                                    "block_index": 0,
                                    "bbox": [0, 0, 0, 0],
                                    "page_width": 612.0,
                                    "page_height": 792.0,
                                    "chunk_in_block": i * 10 + j + 1,
                                    "font_size": None,
                                    "font_sizes": [],
                                    "is_heading": False,
                                    "heading_id": None,
                                    "heading_title": None,
                                    "heading_parent_id": None,
                                    "heading_level": None,
                                    "has_qa_pairs": True,
                                    "chunk_type": "qa_pair_continuation",
                                }
                                out_chunks.append(
                                    {"content": sub_chunk, "meta": sub_chunk_meta}
                                )
                    else:
                        chunk_meta = {
                            "page": 1,
                            "block_index": 0,
                            "bbox": [0, 0, 0, 0],
                            "page_width": 612.0,
                            "page_height": 792.0,
                            "chunk_in_block": i,
                            "font_size": None,
                            "font_sizes": [],
                            "is_heading": False,
                            "heading_id": None,
                            "heading_title": None,
                            "heading_parent_id": None,
                            "heading_level": None,
                            "has_qa_pairs": True,
                            "chunk_type": "qa_pair",
                        }
                        out_chunks.append({"content": qa_text, "meta": chunk_meta})
            else:
                # Fallback về logic cũ nếu không tìm thấy Q&A
                chunks = split_blocks_with_recursive(
                    blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                out_chunks = chunks

            out_path = output_dir / file_path.with_suffix(".jsonl").name
            with out_path.open("w", encoding="utf-8") as fw:
                for ch in out_chunks:
                    # Gắn thêm tên file ở metadata để truy vết
                    ch["meta"]["file_name"] = file_path.name
                    fw.write(json.dumps(ch, ensure_ascii=False) + "\n")

            results["success"].append(str(out_path))
        except Exception as e:
            results["errors"].append(f"Lỗi {file_path.name}: {e}")

    return results
