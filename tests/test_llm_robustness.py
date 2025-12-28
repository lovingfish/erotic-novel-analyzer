import json

import pytest

import backend


def test_prepare_llm_content_head_middle_tail(monkeypatch):
    monkeypatch.setenv("LLM_CONTENT_MAX_CHARS", "60")
    monkeypatch.setenv("LLM_CONTENT_STRATEGY", "head_middle_tail")

    content = ("A" * 120) + ("B" * 120) + ("C" * 120)
    out = backend._prepare_llm_content(content, "Meta")

    assert len(out) <= 60
    assert "...[TRUNCATED]..." in out
    assert "A" in out
    assert "B" in out
    assert "C" in out


def test_prepare_llm_content_section_override(monkeypatch):
    monkeypatch.setenv("LLM_CONTENT_MAX_CHARS", "60")
    monkeypatch.setenv("LLM_CONTENT_MAX_CHARS_META", "20")
    monkeypatch.setenv("LLM_CONTENT_STRATEGY", "head")

    content = "x" * 100
    out = backend._prepare_llm_content(content, "Meta")

    assert out == "x" * 20


def test_call_llm_json_parse_repair(monkeypatch):
    monkeypatch.setenv("LLM_USE_FUNCTION_CALLING", "false")
    monkeypatch.setenv("LLM_REPAIR_ENABLED", "true")

    calls: list[str] = []

    def fake_call_llm_with_response(
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        retry_count: int = 3,
        *,
        timeout: int = 180,
        temperature: float = 1.0,
    ) -> tuple[str, str]:
        calls.append(prompt)
        if "strict JSON repairer" in prompt:
            txt = json.dumps({"fixed": 1}, ensure_ascii=False)
            return txt, txt
        return "not a json", "not a json"

    monkeypatch.setattr(backend, "call_llm_with_response", fake_call_llm_with_response)

    repair_state: dict[str, object] = {"attempted": False}
    data = backend._call_llm_json(
        "http://example.com/v1",
        "sk-test",
        "gpt-test",
        "PROMPT",
        "Meta",
        repair_state=repair_state,
    )

    assert data == {"fixed": 1}
    assert len(calls) == 2
    assert "Bad output" in calls[1]
    assert repair_state.get("attempted") is True
    assert repair_state.get("reason") == "parse"


def test_analyze_meta_validation_repair(monkeypatch):
    monkeypatch.setenv("LLM_USE_FUNCTION_CALLING", "false")
    monkeypatch.setenv("LLM_REPAIR_ENABLED", "true")
    monkeypatch.setenv("LLM_CONTENT_MAX_CHARS", "60")
    monkeypatch.setenv("LLM_CONTENT_STRATEGY", "head_tail")

    monkeypatch.setattr(backend, "_get_llm_config", lambda: ("http://example.com/v1", "sk-test", "gpt-test"))

    calls: list[str] = []

    def fake_call_llm_with_response(
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        retry_count: int = 3,
        *,
        timeout: int = 180,
        temperature: float = 1.0,
    ) -> tuple[str, str]:
        calls.append(prompt)
        if "strict JSON repairer" in prompt:
            payload = {
                "novel_info": {
                    "world_setting": "现代",
                    "world_tags": ["都市"],
                    "chapter_count": 1,
                    "is_completed": False,
                    "completion_note": "",
                },
                "summary": "好的",
            }
        else:
            payload = {"novel_info": {}, "summary": ""}
        txt = json.dumps(payload, ensure_ascii=False)
        return txt, txt

    monkeypatch.setattr(backend, "call_llm_with_response", fake_call_llm_with_response)

    res = backend.analyze_meta(backend.AnalyzeContentRequest(content="章节" * 200))
    assert res["analysis"]["summary"] == "好的"
    assert len(calls) == 2
    assert "...[TRUNCATED]..." in calls[0]


def test_analyze_meta_no_double_repair_when_parse_repair_already_used(monkeypatch):
    monkeypatch.setenv("LLM_USE_FUNCTION_CALLING", "false")
    monkeypatch.setenv("LLM_REPAIR_ENABLED", "true")

    monkeypatch.setattr(backend, "_get_llm_config", lambda: ("http://example.com/v1", "sk-test", "gpt-test"))

    calls: list[str] = []

    def fake_call_llm_with_response(
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        retry_count: int = 3,
        *,
        timeout: int = 180,
        temperature: float = 1.0,
    ) -> tuple[str, str]:
        calls.append(prompt)
        if "strict JSON repairer" in prompt:
            payload = {"novel_info": {}, "summary": ""}
            txt = json.dumps(payload, ensure_ascii=False)
            return txt, txt
        return "not a json", "not a json"

    monkeypatch.setattr(backend, "call_llm_with_response", fake_call_llm_with_response)

    with pytest.raises(backend.HTTPException) as exc:
        backend.analyze_meta(backend.AnalyzeContentRequest(content="x" * 200))

    assert exc.value.status_code == 422
    assert len(calls) == 2
