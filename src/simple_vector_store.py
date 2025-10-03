import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.utils.calculations import cosine_similarity


class SimpleVectorStore:
    """
    Lightweight in-memory vector store that loads items from a JSONL file.

    Each JSONL line is expected to include:
      - content: str
      - embedding: List[float]
      - meta: dict (optional)
    """

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = []  # keep original items
        self.vectors: List[List[float]] = []
        for obj in items:
            emb = obj.get("embedding")
            content = obj.get("content")
            if not isinstance(emb, list) or not content:
                continue
            # filter out empty or zero-length embeddings
            if len(emb) == 0:
                continue
            self.items.append(obj)
            self.vectors.append([float(x) for x in emb])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SimpleVectorStore":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embedded JSONL not found: {p}")
        items: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except json.JSONDecodeError as e:
                    # skip invalid lines but keep going
                    print(
                        f"[vector_store] Skipping invalid JSON at line {line_no}: {e}"
                    )
        return cls(items)

    @classmethod
    def from_jsonl_paths(cls, paths: List[str] | List[Path]) -> "SimpleVectorStore":
        """Load items from multiple JSONL files and create a store.

        Args:
            paths: Iterable of file paths (str or Path)

        Returns:
            SimpleVectorStore
        """
        items: List[Dict[str, Any]] = []
        for p in paths:
            pth = Path(p)
            if not pth.exists():
                print(f"[vector_store] Warning: embedded file not found: {pth}")
                continue
            with pth.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        items.append(obj)
                    except json.JSONDecodeError as e:
                        print(
                            f"[vector_store] Skipping invalid JSON in {pth} at line {line_no}: {e}"
                        )

        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def search_by_vector(
        self, query_vec: Sequence[float], top_k: int = 5, min_score: float = 0.5
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top_k items sorted by cosine similarity descending.

        Returns list of (score, item) tuples.
        """
        if not self.vectors:
            return []
        scores: List[Tuple[float, int]] = []
        for idx, v in enumerate(self.vectors):
            s = cosine_similarity(query_vec, v)
            if s >= min_score:
                scores.append((s, idx))
        scores.sort(key=lambda t: t[0], reverse=True)
        results: List[Tuple[float, Dict[str, Any]]] = []

        # Chỉ trả về các kết quả đã pass min_score filter
        actual_top_k = min(top_k, len(scores))

        for score, idx in scores[:actual_top_k]:
            results.append((score, self.items[idx]))
        return results

    def search_by_text(
        self,
        query_text: str,
        embed_fn,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Convenience: embed query_text with embed_fn(text)->vector and search."""
        qv = embed_fn(query_text)
        return self.search_by_vector(qv, top_k=top_k, min_score=min_score)
