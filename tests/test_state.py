from skillspector.state import llm_call_record


def test_llm_call_record_includes_token_fields() -> None:
    assert llm_call_record("node", ok=True) == {
        "node": "node",
        "ok": True,
        "error": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
