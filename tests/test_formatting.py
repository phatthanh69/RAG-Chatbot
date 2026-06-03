from ragbot.chat.formatting import (
    convert_sources_to_dict,
    process_single_source,
)


class _RetrievalLike:
    """Mimics RetrievalResult: has score/content/get_pdf_name/get_clean_pdf_name."""

    def __init__(self):
        self.score = 0.91
        self.content = "BAS controls berthing"
        self.meta = {"k": "v"}

    def get_pdf_name(self):
        return "doc1.pdf"

    def get_page(self):
        return 3

    def get_clean_pdf_name(self):
        return "doc1"


def test_process_retrieval_like_source():
    out = process_single_source(_RetrievalLike())
    assert out == {
        "score": 0.91,
        "content": "BAS controls berthing",
        "pdf_name": "doc1.pdf",
        "page": 3,
        "clean_pdf_name": "doc1",
        "meta": {"k": "v"},
    }


def test_process_dict_source_cleans_pdf_name():
    src = {"score": 0.5, "content": "x", "meta": {"pdf_name": "my_file-name.pdf", "page": 2}}
    out = process_single_source(src)
    assert out["pdf_name"] == "my_file-name.pdf"
    assert out["clean_pdf_name"] == "my file name"
    assert out["page"] == "2"


def test_process_dict_source_with_rank():
    out = process_single_source({"meta": {}}, include_rank=True, rank=7)
    assert out["rank"] == 7
    assert out["clean_pdf_name"] == "Unknown"


def test_convert_sources_to_dict_shape():
    out = convert_sources_to_dict([_RetrievalLike(), {"meta": {}}])
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(d, dict) for d in out)


def test_convert_sources_unknown_type_uses_fallback():
    # A bare string is neither RetrievalResult-like nor a dict -> fallback entry.
    out = convert_sources_to_dict(["just a string"])
    assert len(out) == 1
    assert out[0]["pdf_name"] == "Unknown"
    assert out[0]["content"] == "just a string"
