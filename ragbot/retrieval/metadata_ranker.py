"""
Enhanced Ranking System using metadata to improve search relevance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ragbot.chat.rag_engine import RetrievalResult


@dataclass
class RankingConfig:
    """Configuration values for metadata-aware ranking."""

    # Weighting factors for the blended score
    semantic_weight: float = 0.7  # Weight for semantic similarity
    metadata_weight: float = 0.3  # Weight for metadata-driven score

    # Metadata boost factors
    font_size_boost: float = 0.15  # Favor larger fonts (titles, headings)
    position_boost: float = 0.1  # Favor content near the top of the page
    section_boost: float = 0.2  # Favor important sections
    freshness_boost: float = 0.05  # Favor newer documents
    section_priority_boost: float = 1.5  # Strong boost when a section matches the query

    # Thresholds
    large_font_threshold: float = 14.0  # Font size considered large
    top_position_threshold: float = 200  # Y coordinate considered near the top

    # Section-aware settings
    enable_section_aware: bool = True  # Enable section-aware ranking heuristics


class MetadataRanker:
    """Ranking helper that leverages metadata to improve retrieved results."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()

    def calculate_metadata_score(self, meta: Dict[str, Any], query: str = "") -> float:
        """
        Calculate a metadata-based score using several heuristics:

        - Font size (large fonts usually indicate headings or emphasis)
        - Position on the page
        - Section labels
        - Content quality and structure
        """
        score = 0.0

        # 1. Font size boost — larger fonts are typically more important
        font_size = meta.get("font_size", 10.0)
        if font_size >= self.config.large_font_threshold:
            font_boost = min((font_size - 10) / 10, 1.0) * self.config.font_size_boost
            score += font_boost

        # 2. Position boost — content near the top of the page is often key
        bbox = meta.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            y_position = bbox[1]  # Y coordinate
            if y_position <= self.config.top_position_threshold:
                position_boost = (
                    self.config.top_position_threshold - y_position
                ) / self.config.top_position_threshold
                score += position_boost * self.config.position_boost

        # 3. Section boost — prefer highlighted or structured sections
        section = meta.get("section", "")
        if section:
            section = section.lower()
        else:
            section = ""
        is_heading = meta.get("is_heading", False)

        if is_heading or self._is_important_section(section):
            score += self.config.section_boost

        # 4. Content quality — score blocks that look well structured
        content_quality = self._assess_content_quality(meta)
        score += content_quality * 0.1

        # 5. Query relevance — match query terms against metadata
        if query:
            relevance = self._calculate_query_metadata_relevance(meta, query)
            score += relevance * 0.1

        return min(score, 1.0)  # Cap the score at 1.0

    def _is_important_section(self, section: str) -> bool:
        """Return True if the section name is considered important."""
        important_keywords = [
            # System overview / project context
            "Giới thiệu",
            "Tổng quan",
            "Hệ thống",
            "Dự án",
            # Features / capabilities
            "Mô tả",
            "Tính năng",
            "Chức năng",
            "Bảng điều khiển",
            "Thông tin",
            "Cài đặt",
            "Cảnh báo",
            "Quản lý",
            "Ứng dụng",
            "Xử lý",
            "Trực quan hóa",
            "Đồ họa",
            "Báo cáo",
            "Lưu trữ",
            "Dữ liệu",
            "Bảo mật",
            "Phân quyền",
            # Hardware / installation
            "Lắp đặt",
            "Kết nối",
            "Cảm biến",
            "Offset",
            "Thiết bị",
            "Đám mây",
            # Monitoring and automation
            "Quan trắc",
            "Giám sát",
            "Cập tàu",
            "Phát hiện",
            "Tự động hóa",
            "Âm thanh",
            "Giám sát hình ảnh",  # thay cho Audio, CCTV
            # Data & reporting
            "Danh sách bản ghi",
            "Lịch sử",
            "Đồng bộ",
            "Báo cáo dữ liệu",
            # Issues & remediation
            "Lỗi",
            "Mất kết nối",
            "Mất mục tiêu",
            "Tín hiệu yếu",
            # Documentation / delivery
            "Bảng báo giá",
            "Hợp đồng",
            "Phụ lục",
            "Ước tính",
            "Quy trình làm việc",
            "Hạn chế",
            "Công việc tương lai",
            "Kết luận",
            # Theoretical background
            "Tổng quan tài liệu",
            "Ứng dụng",
            "Phương pháp neo đậu",
            "Cấu trúc hệ thống",
            "Đo lường",
            "Hiệu suất",
            # Company / people
            "Reecotech",
            "Thành viên",
            "Lời cảm ơn",
            "Danh sách viết tắt",
            "Tài liệu tham khảo",
            # Proper nouns to preserve
            "IPC",
            "CCTV",
            "Laser",
            "LiDAR",
            "VNEMISOFT",
            "BAS",
        ]

        section_lower = section.lower()
        return any(keyword in section_lower for keyword in important_keywords)

    def _assess_content_quality(self, meta: Dict[str, Any]) -> float:
        """Evaluate content quality heuristics based on metadata."""
        quality_score = 0.0

        # Multiple font sizes imply structured content
        font_sizes = meta.get("font_sizes", [])
        if len(font_sizes) > 1:
            quality_score += 0.3

        # Bounding box indicates well-defined layout
        bbox = meta.get("bbox")
        if bbox and len(bbox) == 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            # Keep only reasonable dimensions
            if 50 <= width <= 500 and 10 <= height <= 200:
                quality_score += 0.2

        return min(quality_score, 1.0)

    def _calculate_query_metadata_relevance(
        self, meta: Dict[str, Any], query: str
    ) -> float:
        """Estimate how well the query aligns with metadata fields."""
        relevance = 0.0
        query_lower = query.lower()

        # Match against the PDF/file name
        pdf_name = meta.get("file_name", "").lower()
        if any(word in pdf_name for word in query_lower.split() if len(word) > 2):
            relevance += 0.3

        # Match against section labels
        section = meta.get("section", "")
        if section:
            section = section.lower()
            if any(word in section for word in query_lower.split() if len(word) > 2):
                relevance += 0.5

        return min(relevance, 1.0)

    def detect_section_query(
        self, query: str, all_results: List[RetrievalResult]
    ) -> Optional[str]:
        """
        Determine whether the query targets a specific section name.
        Returns the matching section (if any) using the same casing as the results.
        """
        if not self.config.enable_section_aware:
            return None

        query_lower = query.lower().strip()

    # Gather all sections available in the current results
        available_sections = set()
        for result in all_results:
            section = result.meta.get("section")
            if section and section.strip():
                available_sections.add(section.upper())

    # Check for direct matches first
        for section in available_sections:
            section_lower = section.lower()
            # Accept exact or partial matches
            if (
                query_lower == section_lower
                or query_lower in section_lower
                or section_lower in query_lower
            ):
                return section

    # Fall back to keyword-based matching
        section_keywords = {
            "GIỚI THIỆU": [
                "giới thiệu",
                "introduction",
                "tổng quan",
                "overview",
                "project",
                "system",
                "dự án",
                "hệ thống",
            ],
            "TÍNH NĂNG": [
                "tính năng",
                "feature",
                "chức năng",
                "function",
                "mô tả",
                "description",
                "bảng điều khiển",
                "dashboard",
                "thông tin",
                "cài đặt",
                "quản lý",
                "ứng dụng",
                "xử lý",
                "trực quan hóa",
                "visualization",
                "đồ họa",
                "bảo mật",
                "phân quyền",
                "cảnh báo",
                "alert",
            ],
            "HƯỚNG DẪN / CÀI ĐẶT": [
                "hướng dẫn",
                "instruction",
                "guide",
                "cài đặt",
                "setup",
                "lắp đặt",
                "kết nối",
                "cảm biến",
                "sensor",
                "ipc",
                "offset",
                "sử dụng",
            ],
            "HỆ THỐNG / THIẾT BỊ": [
                "hệ thống",
                "system",
                "quan trắc",
                "monitoring",
                "giám sát",
                "surveillance",
                "cập tàu",
                "berthing",
                "phát hiện",
                "detection",
                "tự động hóa",
                "automation",
                "âm thanh",
                "audio",
                "cctv",
                "giám sát hình ảnh",
                "laser",
                "lidar",
                "thiết bị",
                "equipment",
                "đám mây",
                "cloud",
                "bas",
            ],
            "DỮ LIỆU & BÁO CÁO": [
                "dữ liệu",
                "data",
                "báo cáo",
                "report",
                "lưu trữ",
                "storage",
                "đồng bộ",
                "sync",
                "danh sách bản ghi",
                "record",
                "lịch sử",
                "history",
            ],
            "LỖI & KHẮC PHỤC": [
                "lỗi",
                "error",
                "mất kết nối",
                "disconnection",
                "tín hiệu yếu",
                "weak signal",
                "mất mục tiêu",
                "lost target",
                "khắc phục",
                "troubleshooting",
            ],
            "TÀI LIỆU DỰ ÁN": [
                "bảng báo giá",
                "quotation",
                "hợp đồng",
                "contract",
                "phụ lục",
                "appendix",
                "ước tính",
                "estimate",
                "quy trình làm việc",
                "workflow",
                "hạn chế",
                "limitation",
                "công việc tương lai",
                "future works",
                "kết luận",
                "conclusion",
            ],
            "LÝ THUYẾT / NGHIÊN CỨU": [
                "tổng quan tài liệu",
                "literature review",
                "ứng dụng",
                "application",
                "phương pháp neo đậu",
                "mooring",
                "cấu trúc hệ thống",
                "system structure",
                "đo lường",
                "measurement",
                "hiệu suất",
                "performance",
            ],
            "CÔNG TY / NHÂN SỰ": [
                "reecotech",
                "thành viên",
                "members",
                "lời cảm ơn",
                "acknowledgement",
                "danh sách viết tắt",
                "acronyms",
                "tài liệu tham khảo",
                "references",
            ],
        }

        for section, keywords in section_keywords.items():
            if section in available_sections:
                if any(keyword in query_lower for keyword in keywords):
                    return section

        return None

    def apply_section_aware_ranking(
        self,
        results: List[RetrievalResult],
        target_section: str,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Apply section-aware ranking: prioritize content from the target section.
        Filters by the original cosine score and sorts by the boosted score.
        """
        section_results = []  # Results belonging to the target section
        other_results = []  # Results from other sections

        for result in results:
            # Read the original cosine score for filtering
            original_cosine_score = result.meta.get("original_score", result.score)

            # Skip entries below the minimum score threshold
            if original_cosine_score < min_score:
                continue

            result_section = result.meta.get("section", "")
            if result_section and result_section.upper() == target_section.upper():
                # Boost the score for section content (ranking only)
                boosted_score = min(
                    result.score * self.config.section_priority_boost, 1.0
                )
                enhanced_result = RetrievalResult(
                    content=result.content,
                    score=boosted_score,  # Boosted score used for ranking
                    meta={
                        **result.meta,
                        "original_score": original_cosine_score,  # Preserve the original cosine score
                        "section_boosted": True,
                    },
                )
                section_results.append(enhanced_result)
            else:
                other_results.append(result)

        # Sort boosted section results first
        section_results.sort(key=lambda x: x.score, reverse=True)
        other_results.sort(key=lambda x: x.score, reverse=True)

        # Return section results first, followed by the remaining entries
        return section_results + other_results

    def rerank_results(
        self, results: List[RetrievalResult], query: str = "", min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """Reorder results using a composite score while preserving original cosine scores."""
        if not results:
            return results

        # Step 1: filter by original cosine score (min_score)
        filtered_results = []
        for result in results:
            original_cosine_score = result.meta.get("original_score", result.score)
            if original_cosine_score >= min_score:
                filtered_results.append(result)

        if not filtered_results:
            return []

        # Step 2: compute composite scores for ranking
        scored_results = []
        for result in filtered_results:
            metadata_score = self.calculate_metadata_score(result.meta, query)
            original_cosine_score = result.meta.get("original_score", result.score)
            composite_score = (
                self.config.semantic_weight * original_cosine_score
                + self.config.metadata_weight * metadata_score
            )
            # Store the composite score inside metadata for transparency
            meta = result.meta.copy()
            meta["composite_score"] = composite_score
            scored_results.append(
                (
                    composite_score,
                    RetrievalResult(
                        score=original_cosine_score, content=result.content, meta=meta
                    ),
                )
            )

    # Step 3: sort by composite score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

    # Step 4: return RetrievalResult instances ordered by the computed ranking
        return [r for _, r in scored_results]

    def filter_by_metadata(
        self,
        results: List[RetrievalResult],
        pdf_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        font_size_min: Optional[float] = None,
        section_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Filter results using common metadata constraints."""
        filtered = results

        if pdf_filter:
            filtered = [
                r
                for r in filtered
                if pdf_filter.lower() in r.meta.get("pdf_name", "").lower()
            ]

        if page_filter is not None:
            filtered = [r for r in filtered if r.meta.get("page") == page_filter]

        if font_size_min is not None:
            filtered = [
                r for r in filtered if r.meta.get("font_size", 0) >= font_size_min
            ]

        if section_filter:
            filtered = [
                r
                for r in filtered
                if r.meta.get("section")
                and section_filter.lower() in r.meta.get("section", "").lower()
            ]

        return filtered

    def get_ranking_explanation(self, result: RetrievalResult) -> str:
        """Explain why a result received a high rank."""
        explanations = []
        meta = result.meta

        # Font size
        font_size = meta.get("font_size", 10.0)
        if font_size >= self.config.large_font_threshold:
            explanations.append(f"📝 Large font ({font_size}pt) — likely a heading")

        # Position
        bbox = meta.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4 and bbox[1] <= self.config.top_position_threshold:
            explanations.append("🔝 Near the top of the page")

        # Section
        if meta.get("is_heading", False):
            explanations.append("📋 Marked as a heading")

        # Scores
        original_score = meta.get("original_score", result.score)
        metadata_score = meta.get("metadata_score", 0)
        explanations.append(f"🎯 Original score: {original_score:.3f}")
        explanations.append(f"🏷️ Metadata score: {metadata_score:.3f}")

        return " | ".join(explanations)


def create_smart_ranker(config: Optional[Dict[str, Any]] = None) -> MetadataRanker:
    """Factory helper to create a ranker with optional configuration overrides."""
    ranking_config = RankingConfig()

    if config:
        for key, value in config.items():
            if hasattr(ranking_config, key):
                setattr(ranking_config, key, value)

    return MetadataRanker(ranking_config)


# Preset configurations with section-aware ranking options
RANKING_PRESETS = {
    "balanced": {
        "semantic_weight": 0.7,
        "metadata_weight": 0.3,
        "font_size_boost": 0.15,
        "section_boost": 0.2,
        "enable_section_aware": True,
        "section_priority_boost": 1.5,
    },
    "metadata_heavy": {
        "semantic_weight": 0.5,
        "metadata_weight": 0.5,
        "font_size_boost": 0.25,
        "section_boost": 0.3,
        "enable_section_aware": True,
        "section_priority_boost": 1.8,
    },
    "semantic_focus": {
        "semantic_weight": 0.9,
        "metadata_weight": 0.1,
        "font_size_boost": 0.05,
        "section_boost": 0.1,
    "enable_section_aware": False,  # Disable section-aware behaviour for semantic focus
        "section_priority_boost": 1.0,
    },
    "title_priority": {
        "semantic_weight": 0.6,
        "metadata_weight": 0.4,
        "font_size_boost": 0.3,
        "section_boost": 0.25,
        "large_font_threshold": 12.0,
        "enable_section_aware": True,
        "section_priority_boost": 1.4,
    },
    "section_aware": {  # Preset tailored for section-specific queries
        "semantic_weight": 0.6,
        "metadata_weight": 0.4,
        "font_size_boost": 0.2,
        "section_boost": 0.3,
        "enable_section_aware": True,
        "section_priority_boost": 2.0,  # Apply a strong boost for section content
    },
}
