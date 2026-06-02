from ragbot.utils.text import dedupe_preserve_order, normalize_text


def test_normalize_text_strips_and_lowercases_whitespace():
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_normalize_text_handles_empty():
    assert normalize_text("") == ""


def test_dedupe_preserve_order_keeps_first_occurrence_order():
    assert dedupe_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_dedupe_preserve_order_empty():
    assert dedupe_preserve_order([]) == []
