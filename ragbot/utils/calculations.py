import math
from pathlib import Path
from typing import Sequence


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return _dot(a, b) / (_norm(a) * _norm(b))
