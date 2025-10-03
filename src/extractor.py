import re
from pathlib import Path

import pymupdf  # PyMuPDF
from docx import Document


def get_document_files(input_path_str):
    """
    Lấy danh sách file PDF và DOCX từ đường dẫn đầu vào.
    Trả về dict chứa danh sách file PDF, DOCX và lỗi (nếu có).
    """
    input_path = Path(input_path_str)

    if not input_path.exists():
        return {
            "pdf_files": [],
            "docx_files": [],
            "errors": [f"Đường dẫn không tồn tại: {input_path_str}"],
        }

    if input_path.is_file():
        pdf_files = [input_path] if input_path.suffix.lower() == ".pdf" else []
        docx_files = [input_path] if input_path.suffix.lower() == ".docx" else []
        if not pdf_files and not docx_files:
            return {
                "pdf_files": [],
                "docx_files": [],
                "errors": [f"File không phải PDF hoặc DOCX: {input_path.name}"],
            }
    else:
        pdf_files = list(input_path.glob("*.pdf"))
        docx_files = list(input_path.glob("*.docx"))
        if not pdf_files and not docx_files:
            return {
                "pdf_files": [],
                "docx_files": [],
                "errors": [
                    f"Không tìm thấy file PDF hoặc DOCX trong thư mục: {input_path_str}"
                ],
            }

    return {"pdf_files": pdf_files, "docx_files": docx_files, "errors": []}


def extract_blocks(pdf_path: Path):
    """
    Trả về danh sách block:
    [{"page": int, "block_index": int, "bbox": [x0,y0,x1,y1], "text": str, "w": float, "h": float,
    "font_size": float, "font_sizes": [float, ...], "is_heading": bool, "heading_id": str|None, "heading_title": str|None, "heading_parent_id": str|None, "heading_level": int|None}]
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Lỗi khi mở file PDF {pdf_path}: {e}")
        return []

    blocks = []
    # Maintain state (stack + history) across pages
    hierarchy = {"stack": [], "history": []}

    for pno in range(len(doc)):
        try:
            page = doc[pno]
            W, H = page.rect.width, page.rect.height
            blks = page.get_text("dict").get("blocks", []) or []  # type: ignore
            pdf_name = pdf_path.stem

            images_out_dir = Path("data/processed/images") / pdf_name
            images_out_dir.mkdir(parents=True, exist_ok=True)

            page_blocks = []
            page_image_tuples = page.get_images(full=True) or []
            _page_img_seq = 0

            def _save_image_from_doc(doc_obj, xref, out_path_base: Path):
                try:
                    img_dict = doc_obj.extract_image(xref)
                    img_bytes = img_dict.get("image")
                    ext = img_dict.get("ext", "png")
                    out_path = out_path_base.with_suffix("." + ext)
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    return str(out_path)
                except Exception:
                    try:
                        pix = pymupdf.Pixmap(doc_obj, xref)
                        if pix.n > 4:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        out_path = out_path_base.with_suffix(".png")
                        pix.save(out_path)
                        pix = None
                        return str(out_path)
                    except Exception:
                        return None

            for i, b in enumerate(blks):
                btype = b.get("type", 0)
                bbox = b.get("bbox", [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

                # Handle blocks that are images (type == 1)
                if btype == 1:
                    # try to get xref from block metadata
                    xref = None
                    img_meta = b.get("image") or {}
                    if isinstance(img_meta, dict):
                        xref = img_meta.get("xref")

                    if xref is None:
                        # fallback: use next image tuple from page.get_images()
                        if _page_img_seq < len(page_image_tuples):
                            xref = page_image_tuples[_page_img_seq][0]
                            _page_img_seq += 1

                    image_path = None
                    if xref is not None:
                        base_name = f"{pdf_name}_p{pno+1}_img{i}"
                        out_base = images_out_dir / base_name
                        image_path = _save_image_from_doc(doc, xref, out_base)

                    page_blocks.append(
                        {
                            "page": pno + 1,
                            "block_index": i,
                            "bbox": [x0, y0, x1, y1],
                            "w": W,
                            "h": H,
                            "text": "",
                            "font_size": None,
                            "font_sizes": [],
                            "entity": None,
                            "section": None,
                            "is_heading": False,
                            "is_image": True,
                            "image_path": image_path,
                        }
                    )
                    continue

                # Only process text blocks (type == 0)
                if btype != 0:
                    # other types (e.g., figures) ignored for now
                    continue

                # gom text từ lines->spans
                texts = []
                font_sizes = []
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        s_text = span.get("text", "")
                        if (
                            s_text and s_text.strip()
                        ):  # Ensure s_text is not None and not empty
                            texts.append(s_text)
                        size = span.get("size")
                        if isinstance(size, (int, float)):
                            font_sizes.append(float(size))
                text = " ".join(
                    t.strip() for t in texts if t is not None
                ).strip()  # Ensure t is not None
                if not text:
                    continue
                avg_size = None
                if font_sizes:
                    avg_size = sum(font_sizes) / len(font_sizes)
                page_blocks.append(
                    {
                        "page": pno + 1,
                        "block_index": i,
                        "bbox": [x0, y0, x1, y1],
                        "w": W,
                        "h": H,
                        "text": text,
                        "font_size": (
                            round(avg_size, 2) if avg_size is not None else None
                        ),
                        "font_sizes": [round(s, 2) for s in font_sizes],
                        "entity": None,
                        "section": None,
                        "is_heading": False,
                    }
                )

            # Sort blocks by top-to-bottom position (increasing y0)
            page_blocks.sort(key=lambda bb: bb["bbox"][1])

            # Assign sections to blocks, using the state to continue across pages
            page_blocks, hierarchy = assign_sections_to_blocks(page_blocks, hierarchy)

            # Add processed page_blocks to the aggregated result
            blocks.extend(page_blocks)

        except Exception as e:
            print(f"Lỗi khi xử lý trang {pno + 1} của file PDF {pdf_path}: {e}")
            continue

    doc.close()
    return blocks


def assign_sections_to_blocks(page_blocks, hierarchy=None):
    """
    Convert from entity/section logic to a hierarchical heading structure with fields:
    - heading_id: Structured identifier (e.g., I, I.1, I.1.a)
    - heading_title: Title after the prefix (e.g., "Cảm biến laser (Laser Sensors)")
    - heading_parent_id: Parent identifier (e.g., the parent of I.1 is I; the parent of I.1.a is I.1)
    - heading_level: Level 1/2/3 corresponding to Roman numerals, numbers, or letters

    """
    # Convert the 'hierarchy' parameter into a state that can persist the stack across pages
    # For backward compatibility, allow passing a list; internally wrap it into a dict-based state
    if hierarchy is None:
        state = {"stack": [], "history": [], "last_level1": None, "last_level2": None}
    elif isinstance(hierarchy, dict):
        # Already a state object
        state = hierarchy
        state.setdefault("stack", [])
        state.setdefault("history", [])
        state.setdefault("last_level1", None)
        state.setdefault("last_level2", None)
    else:
        # Legacy list -> wrap into state structure
        state = {
            "stack": [],
            "history": hierarchy,
            "last_level1": None,
            "last_level2": None,
        }

    heading_stack = state.get("stack", [])  # Retain between pages

    # Detailed regex patterns to capture prefixes and titles
    roman_re = re.compile(r"^([IVXLCDM]+)\.\s*(.+)")
    arabic_re = re.compile(r"^(\d+)\.\s*(.+)")
    letter_re = re.compile(r"^([a-zA-Z])[\.)]\s*(.+)")

    for bb in page_blocks:
        text = (bb.get("text") or "").strip()

        level = 0
        prefix = None
        title = None

        m = roman_re.match(text)
        if m:
            level = 1
            prefix, title = m.group(1), m.group(2)
        else:
            m = arabic_re.match(text)
            if m:
                level = 2
                prefix, title = m.group(1), m.group(2)
            else:
                m = letter_re.match(text)
                if m:
                    level = 3
                    prefix, title = m.group(1), m.group(2)

        if level > 0:
            # Heading detected – normalize the title according to these rules:
            # - Level 2 (numeric): if a colon exists, use the portion AFTER the colon as the value
            # - Level 3 (alphabetic): if a colon exists, use the portion BEFORE the colon as the label
            raw_title = (title or "").strip()
            norm_title = raw_title
            if ":" in raw_title:
                if level == 2:
                    # Take the substring after the final colon to avoid labels containing colons
                    _lab, _val = raw_title.rsplit(":", 1)
                    norm_title = _val.strip()
                elif level == 3:
                    _lab, _val = raw_title.split(":", 1)
                    norm_title = _lab.strip()

            title = norm_title

            # Drop nodes whose level is greater than or equal to the current level
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            parent_id = heading_stack[-1]["id"] if heading_stack else None

            # Fallback when no valid parent is available (page break or irregular format)
            if parent_id is None:
                if level == 2:
                    last_l1 = state.get("last_level1")
                    if last_l1:
                        parent_id = last_l1["id"]
                elif level == 3:
                    last_l2 = state.get("last_level2")
                    last_l1 = state.get("last_level1")
                    if last_l2:
                        parent_id = last_l2["id"]
                    elif last_l1:
                        parent_id = last_l1["id"]

            # Build the heading_id according to the convention
            if level == 1:
                heading_id = (prefix or "").upper()
            elif level == 2:
                heading_id = f"{parent_id}.{prefix}"
            else:  # level == 3
                heading_id = f"{parent_id}.{(prefix or '').lower()}"

            node = {
                "level": level,
                "id": heading_id,
                "title": title,
                "parent_id": parent_id,
            }
            heading_stack.append(node)

            # Update the trace of the nearest parent level
            if level == 1:
                state["last_level1"] = node
                state["last_level2"] = None
            elif level == 2:
                state["last_level2"] = node

            # Assign metadata to the block
            bb["is_heading"] = True
            bb["heading_id"] = heading_id
            bb["heading_title"] = title
            bb["heading_parent_id"] = parent_id
            bb["heading_level"] = level

            # Entity and section fields were removed
            bb["entity"] = None
            bb["section"] = None

            # Record history in state["history"] for debugging and compatibility (not used to build the stack)
            try:
                state["history"].append((level, None, None))
            except Exception:
                pass
        else:
            # Not a heading: inherit the nearest context if available
            bb["is_heading"] = False
            if heading_stack:
                top = heading_stack[-1]
                bb["heading_id"] = top["id"]
                bb["heading_title"] = top["title"]
                bb["heading_parent_id"] = top["parent_id"]
                bb["heading_level"] = top["level"]
                bb["entity"] = None
                bb["section"] = None
            else:
                bb["heading_id"] = None
                bb["heading_title"] = None
                bb["heading_parent_id"] = None
                bb["heading_level"] = None
                bb["entity"] = None
                bb["section"] = None

    # Update the state stack so the next page can inherit it
    state["stack"] = heading_stack
    return page_blocks, state


def extract_docx_blocks(docx_path: Path):
    """
    Extract blocks from a DOCX file.

    Returns a list of block dictionaries similar to ``extract_blocks`` for PDFs:
    [{"page": int, "block_index": int, "bbox": [x0, y0, x1, y1], "text": str, "w": float, "h": float,
      "font_size": float, "font_sizes": [float, ...], "is_heading": bool, "heading_id": str | None,
      "heading_title": str | None, "heading_parent_id": str | None, "heading_level": int | None}]
    """

    try:
        doc = Document(str(docx_path))
        blocks = []

        # Emulate page and bounding box metadata for DOCX (DOCX lacks native page concepts like PDF)
        page_width = 612.0  # Approximate A4 width
        page_height = 792.0  # Approximate A4 height
        current_y = 50.0  # Start at the top margin

        block_index = 0

        for paragraph in doc.paragraphs:
            text = paragraph.text
            if text is None:
                continue
            text = text.strip()
            if not text:
                continue

            # Analyze paragraph style to determine font size
            font_size = 12.0  # Default font size
            font_sizes = [font_size]

            # Inspect the paragraph style
            if (
                paragraph.style
                and paragraph.style.name
                and paragraph.style.name.startswith("Heading")
            ):
                # Heading styles typically have larger font sizes
                heading_level = paragraph.style.name.replace("Heading ", "")
                try:
                    level = int(heading_level)
                    font_size = max(
                        16.0 - level * 2, 12.0
                    )  # Heading 1=16pt, Heading 2=14pt, etc.
                except Exception:
                    font_size = 14.0
                font_sizes = [font_size]
            else:
                # Inspect runs within the paragraph to derive font size
                run_sizes = []
                for run in paragraph.runs:
                    if run.font.size:
                        size_pt = run.font.size.pt if run.font.size.pt else 12.0
                        run_sizes.append(size_pt)

                if run_sizes:
                    font_sizes = run_sizes
                    font_size = sum(run_sizes) / len(run_sizes)

            # Compute a synthetic bounding box
            text_height = font_size * 1.2  # Line height = font_size * 1.2
            x0, y0 = 50.0, current_y
            x1, y1 = page_width - 50.0, current_y + text_height

            block = {
                "page": 1,  # Treat the DOCX as a single page
                "block_index": block_index,
                "bbox": [x0, y0, x1, y1],
                "w": page_width,
                "h": page_height,
                "text": text,
                "font_size": round(font_size, 2),
                "font_sizes": [round(s, 2) for s in font_sizes],
                "entity": None,
                "section": None,
                "is_heading": False,  # Will be updated by assign_sections_to_blocks
            }

            blocks.append(block)

            # Update the y position for the next block
            current_y += text_height + 5.0  # Add spacing
            block_index += 1

    # Apply hierarchical heading assignment logic to DOCX blocks using the unified state
        state = {"stack": [], "history": []}
        blocks, _ = assign_sections_to_blocks(blocks, state)

        return blocks
    except Exception as e:
        print(f"Lỗi khi đọc file DOCX {docx_path}: {e}")
        return []


def filter_layout_noise(
    blocks, header_footer_margin_ratio=0.04
):  # Note: adjust header_footer_margin_ratio as needed
    """
    - Loại header/footer theo ngưỡng biên trang (4% chiều cao trang mặc định).
    """
    out = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        H = b["h"]
        top_margin = H * header_footer_margin_ratio
        bottom_margin = H * (1 - header_footer_margin_ratio)
    # Omit the block if it lies entirely within the header/footer region
        if y1 <= top_margin or y0 >= bottom_margin:
            continue
        out.append(b)
    return out


def extract_document_blocks(file_path: Path):
    """
    Trích xuất blocks từ file PDF hoặc DOCX.
    Tự động phát hiện loại file và sử dụng hàm trích xuất phù hợp.
    """
    if file_path.suffix.lower() == ".pdf":
        return extract_blocks(file_path)
    elif file_path.suffix.lower() == ".docx":
        return extract_docx_blocks(file_path)
    else:
        print(f"Định dạng file không được hỗ trợ: {file_path.suffix}")
        return []


def extract_headings_from_text(text):
    """
    Extract a list of headings from the provided text.

    Returns a list of dictionaries: [{"level": int, "text": str, "full_text": str}, ...]
    - level: Heading level (1, 2, 3, ...)
    - text: Heading content after removing the prefix
    - full_text: Complete heading text including the prefix
    """

    if not text:
        return []

    # Regex patterns for headings (similar to assign_sections_to_blocks)
    heading_patterns = [
        r"^[IVXLCDM]+\.\s*(.+)",  # Roman numerals: I. II. III.
        r"^\d+\.\s*(.+)",  # Numbers: 1. 2. 3.
        r"^[a-zA-Z]\)\s*(.+)",  # Letters: a) b) c)
        r"^[a-zA-Z]\.\s*(.+)",  # Letters: a. b. c.
    ]
    heading_regex = re.compile("|".join(heading_patterns))

    headings = []
    lines = text.split("\n")  # Assume the text is separated into lines

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = heading_regex.match(line)
        if not match:
            continue

        # Determine the heading level based on the matched pattern
        if re.match(r"^[IVXLCDM]+\.", line):
            level = 1  # Top-level
        elif re.match(r"^\d+\.", line):
            level = 2  # Sub-level
        elif re.match(r"^[a-zA-Z][\.\)]", line):
            level = 3  # Sub-sub-level
        else:
            level = 0

        # Extract the text after the prefix
        heading_text = (
            match.group(1) or match.group(2) or match.group(3) or match.group(4)
        )
        if not heading_text:
            continue

        heading_text = heading_text.strip()

        # Remove a trailing colon if present
        if heading_text.endswith(":"):
            heading_text = heading_text[:-1].strip()

        headings.append({"level": level, "text": heading_text, "full_text": line})

    return headings


def extract_content_by_headings(text):
    """
    Trích xuất nội dung văn bản được nhóm theo heading.
    Trả về dict: {heading_full_text: content_under_heading, ...}
    - heading_full_text: Toàn bộ text của heading (ví dụ: "1. Giới thiệu")
    - content_under_heading: Nội dung văn bản dưới heading đó, cho đến heading tiếp theo
    """
    if not text:
        return {}

    # Regex patterns for headings
    heading_patterns = [
        r"^[IVXLCDM]+\.\s*(.+)",  # Roman numerals: I. II. III.
        r"^\d+\.\s*(.+)",  # Numbers: 1. 2. 3.
        r"^[a-zA-Z]\)\s*(.+)",  # Letters: a) b) c)
        r"^[a-zA-Z]\.\s*(.+)",  # Letters: a. b. c.
    ]
    heading_regex = re.compile("|".join(heading_patterns), re.MULTILINE)

    # Find all matches along with their positions
    matches = []
    for match in heading_regex.finditer(text):
        start_pos = match.start()
        full_text = match.group(0).strip()
    # Determine the heading level
        if re.match(r"^[IVXLCDM]+\.", full_text):
            level = 1
        elif re.match(r"^\d+\.", full_text):
            level = 2
        elif re.match(r"^[a-zA-Z][\.\)]", full_text):
            level = 3
        else:
            level = 0
        matches.append((start_pos, full_text, level))

    if not matches:
        return {"No headings": text}

    # Sort by position
    matches.sort(key=lambda x: x[0])

    # Keep only level 1 and level 2 headings to avoid noise from sub-headings
    filtered_matches = [m for m in matches if m[2] in [1, 2]]

    content_by_heading = {}
    for i, (start_pos, heading_text, level) in enumerate(filtered_matches):
    # Determine end_pos: heading start of the next section or the end of text
        if i + 1 < len(filtered_matches):
            end_pos = filtered_matches[i + 1][0]
        else:
            end_pos = len(text)

    # Content from after the heading up to the next heading
        content = text[start_pos + len(heading_text) : end_pos].strip()
    # Remove heading patterns at the start of the content if present
        content = re.sub(
            r"^[IVXLCDM]+\.\s*|^ \d+\.\s*|^[a-zA-Z][\.\)]\s*", "", content
        ).strip()
        content_by_heading[heading_text] = content

    return content_by_heading


def main(pdf_path_str, output_file="extracted_content.txt"):
    """
    Main helper to extract content from a PDF and save it to a text file.
    - pdf_path_str: Path to the PDF file
    - output_file: Output filename (default: extracted_content.txt)
    """
    pdf_path = Path(pdf_path_str)

    if not pdf_path.exists():
        print(f"File không tồn tại: {pdf_path_str}")
        return

    # Extract blocks from the PDF
    blocks = extract_blocks(pdf_path)

    # Combine all text from the blocks
    full_text = "\n".join([b["text"] for b in blocks if b["text"]])

    # Apply heading-based extraction
    content_by_headings = extract_content_by_headings(full_text)

    # Xuất ra file text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Nội dung được nhóm theo heading:\n\n")
        for heading, content in content_by_headings.items():
            f.write(f"--- {heading} ---\n")
            f.write(f"{content}\n\n")

    print(f"Đã xuất nội dung ra file: {output_file}")
    print(f"Số lượng heading: {len(content_by_headings)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Sử dụng: python extractor.py <path_to_pdf> [output_file]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_content.txt"
    main(pdf_path, output_file)
