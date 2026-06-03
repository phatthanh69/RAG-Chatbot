from ragbot.chat.classification import QuestionClassifier


def test_fallback_classification_returns_dict_with_type():
    # The rule-based fallback needs no genai client or heading provider.
    clf = QuestionClassifier()
    result = clf.fallback_classify("Giá của thiết bị BAS là bao nhiêu?")
    assert isinstance(result, dict)
    assert result["question_type"] == "BlueEco_BAS"
    assert "heading_info" in result


def test_fallback_classification_preserves_question():
    clf = QuestionClassifier()
    question = "Cảm biến laser là gì?"
    result = clf.fallback_classify(question)
    assert result["rewritten_question"] == question


def test_fallback_classification_reads_active_headings_from_session():
    clf = QuestionClassifier()
    session_data = {"metadata": {"active_headings": ["LS-BE-001"]}}
    result = clf.fallback_classify("thông số kỹ thuật", session_data)
    assert result["heading_info"]["active_headings"] == ["LS-BE-001"]
