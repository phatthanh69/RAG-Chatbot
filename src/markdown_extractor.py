"""
Markdown Extractor - Trích xuất và xử lý nội dung từ file Markdown
Áp dụng logic tương tự như PDF extractor nhưng cho định dạng Markdown
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.cleaner import normalize_text
from src.config import config


@dataclass
class MarkdownBlock:
    """Class đại diện cho một block trong Markdown"""

    content: str
    block_type: str  # 'heading', 'text', 'code', 'list', 'table', etc.
    heading_level: Optional[int] = None
    heading_id: Optional[str] = None
    heading_title: Optional[str] = None
    heading_parent_id: Optional[str] = None
    line_number: int = 0
    is_heading: bool = False


class MarkdownHierarchyManager:
    """Quản lý hierarchy của headings trong Markdown tương tự như PDF"""

    def __init__(self):
        self.stack = []  # Stack để track current path
        self.history = []  # Lịch sử tất cả headings đã xử lý
        self.heading_counter = {}  # Counter cho từng level để tạo ID

    def generate_heading_id(self, level: int, title: str) -> str:
        """Tạo heading ID dựa trên level với format số thuần túy (1, 1.1, 1.1.1, ...)"""
        # Reset counters cho các level cao hơn khi gặp level thấp hơn hoặc bằng
        current_levels = list(self.heading_counter.keys())
        for lvl in current_levels:
            if lvl > level:
                self.heading_counter[lvl] = 0

        # Tăng counter cho level hiện tại
        if level not in self.heading_counter:
            self.heading_counter[level] = 0
        self.heading_counter[level] += 1

        # Tạo ID dựa trên parent hierarchy
        if level == 1:
            # Level 1: 1, 2, 3, ...
            return str(self.heading_counter[level])
        else:
            # Level 2+: tìm parent ID từ stack và thêm số hiện tại
            parent_id = None
            for item in reversed(self.stack):
                if item["level"] < level:
                    parent_id = item["id"]
                    break

            if parent_id:
                return f"{parent_id}.{self.heading_counter[level]}"
            else:
                # Fallback nếu không tìm thấy parent (không nên xảy ra)
                return str(self.heading_counter[level])

    def process_heading(self, level: int, title: str) -> Dict[str, Any]:
        """Xử lý heading và cập nhật hierarchy với numbering system (1, 1.1, 1.1.1, ...)"""
        # Reset counters cho các level cao hơn khi gặp level thấp hơn
        current_levels = list(self.heading_counter.keys())
        for lvl in current_levels:
            if lvl > level:
                self.heading_counter[lvl] = 0

        # Pop stack để tìm parent phù hợp
        while self.stack and self.stack[-1]["level"] >= level:
            self.stack.pop()

        # Tạo ID và tìm parent
        heading_id = self.generate_heading_id(level, title)
        parent_id = self.stack[-1]["id"] if self.stack else None

        # Tạo heading info
        heading_info = {
            "id": heading_id,
            "title": title,
            "level": level,
            "parent_id": parent_id,
        }

        # Thêm vào stack và history
        self.stack.append(heading_info)
        self.history.append(heading_info)

        return heading_info


def get_markdown_files(input_path_str: str) -> Dict[str, Any]:
    """
    Lấy danh sách file Markdown từ đường dẫn đầu vào.
    Trả về dict chứa danh sách file MD và lỗi (nếu có).
    """
    input_path = Path(input_path_str)

    if not input_path.exists():
        return {
            "md_files": [],
            "errors": [f"Đường dẫn không tồn tại: {input_path_str}"],
        }

    if input_path.is_file():
        md_files = (
            [input_path] if input_path.suffix.lower() in [".md", ".markdown"] else []
        )
        if not md_files:
            return {
                "md_files": [],
                "errors": [f"File không phải Markdown: {input_path.name}"],
            }
    else:
        md_files = list(input_path.glob("*.md")) + list(input_path.glob("*.markdown"))
        if not md_files:
            return {
                "md_files": [],
                "errors": [
                    f"Không tìm thấy file Markdown trong thư mục: {input_path_str}"
                ],
            }

    return {"md_files": md_files, "errors": []}


def extract_markdown_blocks(md_path: Path) -> List[MarkdownBlock]:
    """
    Trích xuất và phân tích các block từ file Markdown.
    Trả về danh sách MarkdownBlock với thông tin hierarchy.
    """
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Lỗi khi đọc file Markdown {md_path}: {e}")
        return []

    blocks = []
    hierarchy_manager = MarkdownHierarchyManager()

    lines = content.split("\n")
    current_block_lines = []
    current_block_type = "text"
    line_number = 0

    def flush_current_block():
        """Xử lý block hiện tại và thêm vào danh sách"""
        if current_block_lines:
            block_content = "\n".join(current_block_lines).strip()
            if block_content:
                block = MarkdownBlock(
                    content=block_content,
                    block_type=current_block_type,
                    line_number=line_number - len(current_block_lines),
                )
                blocks.append(block)
            current_block_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        line_number = i + 1

        # Detect headings (# ## ### #### ##### ######)
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading_match:
            # Flush previous block
            flush_current_block()

            # Process heading
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Xử lý dấu : để lấy phần sau làm heading_title
            processed_title = title
            if ":" in title:
                # Lấy phần sau dấu : cuối cùng và loại bỏ markdown formatting
                _, after_colon = title.rsplit(":", 1)
                after_colon = after_colon.strip()
                # Loại bỏ markdown bold formatting (**text**)
                if after_colon.startswith("**") and after_colon.endswith("**"):
                    after_colon = after_colon[2:-2].strip()
                if after_colon:  # Chỉ sử dụng nếu có nội dung sau dấu :
                    processed_title = after_colon

            # Normalize title
            normalized_title = normalize_text(processed_title)

            # Get hierarchy info
            heading_info = hierarchy_manager.process_heading(level, normalized_title)

            # Create heading block
            heading_block = MarkdownBlock(
                content=line.strip(),
                block_type="heading",
                heading_level=level,
                heading_id=heading_info["id"],
                heading_title=normalized_title,
                heading_parent_id=heading_info["parent_id"],
                line_number=line_number,
                is_heading=True,
            )
            blocks.append(heading_block)

            current_block_type = "text"

        # Detect code blocks
        elif line.strip().startswith("```"):
            flush_current_block()

            # Find closing ```
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if lines[i].strip().startswith("```"):
                    break
                i += 1

            # Create code block
            code_block = MarkdownBlock(
                content="\n".join(code_lines),
                block_type="code",
                line_number=line_number,
            )
            blocks.append(code_block)
            current_block_type = "text"

        # Detect tables
        elif "|" in line and line.strip():
            if current_block_type != "table":
                flush_current_block()
                current_block_type = "table"
            current_block_lines.append(line)

        # Detect lists
        elif re.match(r"^[\s]*[-*+]\s+", line) or re.match(r"^[\s]*\d+\.\s+", line):
            if current_block_type != "list":
                flush_current_block()
                current_block_type = "list"
            current_block_lines.append(line)

        # Regular text
        else:
            if line.strip():  # Non-empty line
                if current_block_type not in ["text"]:
                    flush_current_block()
                    current_block_type = "text"
                current_block_lines.append(line)
            else:  # Empty line
                if current_block_lines:  # If we have content, it's end of block
                    flush_current_block()
                    current_block_type = "text"

        i += 1

    # Flush final block
    flush_current_block()

    return blocks


def chunk_markdown_blocks(
    blocks: List[MarkdownBlock], max_chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Chia các block Markdown thành chunks với thông tin heading hierarchy.
    Tương tự như chunker cho PDF.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    current_heading_info = {
        "heading_id": None,
        "heading_title": None,
        "heading_parent_id": None,
        "heading_level": None,
    }

    for block in blocks:
        if block.is_heading:
            # Update current heading context
            current_heading_info = {
                "heading_id": block.heading_id,
                "heading_title": block.heading_title,
                "heading_parent_id": block.heading_parent_id,
                "heading_level": block.heading_level,
            }

            # Add heading as a separate chunk
            chunks.append(
                {
                    "content": block.content,
                    "block_type": block.block_type,
                    "line_number": block.line_number,
                    "is_heading": True,
                    **current_heading_info,
                }
            )
        else:
            # Split non-heading content if too large
            if len(block.content) > max_chunk_size:
                text_chunks = splitter.split_text(block.content)
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(
                        {
                            "content": normalize_text(chunk_text),
                            "block_type": block.block_type,
                            "line_number": block.line_number,
                            "is_heading": False,
                            "chunk_index": i,
                            **current_heading_info,
                        }
                    )
            else:
                chunks.append(
                    {
                        "content": normalize_text(block.content),
                        "block_type": block.block_type,
                        "line_number": block.line_number,
                        "is_heading": False,
                        **current_heading_info,
                    }
                )

    return chunks


def process_markdown_file(
    md_path: Path, output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Xử lý một file Markdown hoàn chỉnh: extract blocks, chunk, và xuất kết quả.
    """
    if output_dir is None:
        output_dir = Path("data/processed")

    print(f"Đang xử lý file: {md_path}")

    # Extract blocks
    blocks = extract_markdown_blocks(md_path)

    if not blocks:
        return {"error": f"Không thể extract blocks từ {md_path}"}

    print(f"Đã extract {len(blocks)} blocks")

    # Create chunks
    chunks = chunk_markdown_blocks(blocks)

    print(f"Đã tạo {len(chunks)} chunks")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save chunks to JSON
    output_file = output_dir / f"{md_path.stem}_chunks.json"

    result = {
        "file_name": md_path.name,
        "file_path": str(md_path),
        "total_blocks": len(blocks),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả vào: {output_file}")

    return result


def process_markdown_directory(
    input_path: str, output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Xử lý tất cả file Markdown trong thư mục.
    """
    # Get markdown files
    file_info = get_markdown_files(input_path)

    if file_info["errors"]:
        return {"errors": file_info["errors"]}

    md_files = file_info["md_files"]
    output_path = Path(output_dir)

    results = []
    errors = []

    for md_file in md_files:
        try:
            result = process_markdown_file(md_file, output_path)
            if "error" in result:
                errors.append(result["error"])
            else:
                results.append(result)
        except Exception as e:
            error_msg = f"Lỗi khi xử lý {md_file}: {e}"
            print(error_msg)
            errors.append(error_msg)

    return {
        "processed_files": len(results),
        "results": results,
        "errors": errors,
        "total_files": len(md_files),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract và chunk file Markdown với heading hierarchy"
    )
    parser.add_argument("input_path", help="Đường dẫn đến file hoặc thư mục Markdown")
    parser.add_argument(
        "-o",
        "--output",
        default="data/processed",
        help="Thư mục output (mặc định: data/processed)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1000,
        help="Kích thước tối đa của chunk (mặc định: 1000)",
    )

    args = parser.parse_args()

    print("=== Markdown Extractor ===")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output}")
    print(f"Max chunk size: {args.max_chunk_size}")
    print()

    # Process files
    result = process_markdown_directory(args.input_path, args.output)

    if result.get("errors"):
        print("Lỗi:")
        for error in result["errors"]:
            print(f"  - {error}")

    print(
        f"\nKết quả: Đã xử lý {result.get('processed_files', 0)}/{result.get('total_files', 0)} files"
    )
