"""Source formatting helpers shared by chat orchestration.

`process_single_source` / `convert_sources_to_dict` normalize retrieval results
(RetrievalResult objects, dicts, or unknown shapes) into the dict format the
frontend expects. Bodies are moved verbatim from `ChatbotService`; the only
`self.<x>` dependency was `self.logger`, now an injected `logger` argument.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional


def process_single_source(
    source: Any,
    include_rank: bool = False,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
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
    logger = logger or logging.getLogger(__name__)
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
            logger.warning(f"Unknown source type: {type(source)}")
            return None

    except Exception as e:
        logger.error(f"Error processing source: {str(e)}")
        return None


def convert_sources_to_dict(
    sources: List, logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Convert sources from RetrievalResult objects or dictionaries to standardized dict format

    Args:
        sources: List of sources (RetrievalResult objects or dicts)

    Returns:
        List of standardized source dictionaries
    """
    logger = logger or logging.getLogger(__name__)
    try:
        frontend_sources = []
        for i, source in enumerate(sources):
            try:
                # logger.debug(f"Processing source {i}: {type(source)}")
                processed_source = process_single_source(source, logger=logger)
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
                        logger.error(
                            f"Fallback processing failed for source {i}: {str(fallback_error)}"
                        )
                        continue
            except Exception as e:
                logger.error(f"Error processing source {i}: {str(e)}")
                logger.error(f"Source details: {source}")
                continue

        logger.info(f"Successfully processed {len(frontend_sources)} sources")
        return frontend_sources

    except Exception as e:
        logger.error(f"Error in source conversion: {str(e)}")
        return []
