from ragbot.chat.prompt_builder import build_prompt, get_recent_context


def test_get_recent_context_returns_last_n():
    # Real history items use message_type ("user"/"bot") and content.
    history = []
    for i in range(5):
        history.append({"message_type": "user", "content": f"q{i}"})
        history.append({"message_type": "bot", "content": f"a{i}"})
    out = get_recent_context(history, last_n=2)  # last 4 messages -> q3/a3, q4/a4
    assert "q4" in out and "q3" in out
    assert "q0" not in out


def test_get_recent_context_empty():
    assert get_recent_context([], last_n=3) == ""


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt("What is BAS?", sources=[], context="BAS = Building Automation")
    assert "What is BAS?" in prompt
    assert "BAS = Building Automation" in prompt
