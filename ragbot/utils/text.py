"""Pure text helpers shared by chat orchestration and retrieval."""


def normalize_text(s: str) -> str:
    """Normalize text by lowercasing, removing extra spaces, and handling common plurals."""
    if not s:
        return ""
    text = " ".join(s.lower().split())
    # Simple plural normalization
    plural_map = {
        "lasers": "laser",
        "sensors": "sensor",
        "cảm biến": "cảm biến",  # already singular
        "thiết bị": "thiết bị",
        "hệ thống": "hệ thống",
        "trạm": "trạm",
        "máy": "máy",
    }
    for plural, singular in plural_map.items():
        text = text.replace(plural, singular)
    return text


def dedupe_preserve_order(items):
    """Deduplicate list while preserving order."""
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        key = " ".join(str(it).split())
        if key.lower() not in seen:
            seen.add(key.lower())
            out.append(key)
    return out
