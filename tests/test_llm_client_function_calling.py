from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from novel_analyzer.config_loader import (
    ContentProcessingConfig,
    DefaultsConfig,
    LLMConfig,
    RepairConfig,
    RepairTemplateConfig,
    RetryPolicy,
    SectionConfig,
)
from novel_analyzer.llm_client import LLMClient, LLMRuntime
from novel_analyzer.schemas import MetaOutput


class _FakeResponse:
    def __init__(self, status_code: int, *, text: str, json_obj: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._json_obj = json_obj

    def json(self):
        if self._json_obj is None:
            raise ValueError("no json")
        return self._json_obj


def _make_cfg(*, repair_enabled: bool = False) -> LLMConfig:
    return LLMConfig(
        defaults=DefaultsConfig(
            timeout_seconds=10,
            retry=RetryPolicy(
                count=1,
                backoff="linear",
                base_wait_seconds=0,
                max_wait_seconds=0,
                retryable_status_codes=(429, 502, 503, 504),
            ),
        ),
        content_processing=ContentProcessingConfig(
            max_chars=100,
            strategy="head",
            boundary_aware=False,
            boundary_search_window=200,
            truncation_marker_template="...[TRUNCATED]...",
        ),
        repair=RepairConfig(enabled=repair_enabled, max_attempts=1, prompt_head_max_chars=1000, bad_output_max_chars=1000),
        sections={
            "meta": SectionConfig(
                temperature=0.0,
                tool_name="extract_meta",
                description="meta",
                prompt_template="x",
            )
        },
        repair_template=RepairTemplateConfig(temperature=0.0, prompt_template="repair"),
    )


def test_function_calling_parses_tool_calls(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg(repair_enabled=False)
    client = LLMClient(runtime, cfg)

    captured_payloads: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        captured_payloads.append(json)
        args = {
            "novel_info": {
                "world_setting": "现代",
                "world_tags": ["都市"],
                "chapter_count": 1,
                "is_completed": False,
                "completion_note": "",
            },
            "summary": "好的",
        }
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "extract_meta",
                                    "arguments": json_module.dumps(args, ensure_ascii=False),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        return _FakeResponse(200, text=json_module.dumps(data, ensure_ascii=False), json_obj=data)

    import novel_analyzer.llm_client as llm_client_mod

    json_module = json
    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="meta", prompt="PROMPT", output_model=MetaOutput)
    assert out.summary == "好的"

    assert captured_payloads
    payload = captured_payloads[0]
    assert "tools" in payload and "tool_choice" in payload


def test_function_calling_falls_back_to_legacy_functions(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg(repair_enabled=False)
    client = LLMClient(runtime, cfg)

    calls: list[dict] = []

    args = {
        "novel_info": {
            "world_setting": "现代",
            "world_tags": ["都市"],
            "chapter_count": 1,
            "is_completed": False,
            "completion_note": "",
        },
        "summary": "ok",
    }

    legacy_data = {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "extract_meta",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    }
                }
            }
        ]
    }

    responses = [
        _FakeResponse(400, text="unsupported tools/tool_choice"),
        _FakeResponse(200, text=json.dumps(legacy_data, ensure_ascii=False), json_obj=legacy_data),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return responses.pop(0)

    import novel_analyzer.llm_client as llm_client_mod

    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="meta", prompt="PROMPT", output_model=MetaOutput)
    assert out.summary == "ok"

    assert len(calls) == 2
    assert "tools" in calls[0]
    assert "functions" in calls[1]


def test_meta_normalizes_null_novel_info(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg(repair_enabled=False)
    client = LLMClient(runtime, cfg)

    bad_args = {"novel_info": None, "summary": "ok"}
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "extract_meta",
                                "arguments": json.dumps(bad_args, ensure_ascii=False),
                            }
                        }
                    ]
                }
            }
        ]
    }

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, text=json_module.dumps(data, ensure_ascii=False), json_obj=data)

    import novel_analyzer.llm_client as llm_client_mod

    json_module = json
    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="meta", prompt="PROMPT", output_model=MetaOutput)
    assert out.summary == "ok"
    assert out.novel_info.world_setting == "未知"
    assert out.novel_info.is_completed is False


def test_meta_falls_back_to_message_content_when_no_tool_calls(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg(repair_enabled=False)
    client = LLMClient(runtime, cfg)

    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "这是摘要",
                }
            }
        ]
    }

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, text=json_module.dumps(data, ensure_ascii=False), json_obj=data)

    import novel_analyzer.llm_client as llm_client_mod

    json_module = json
    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="meta", prompt="PROMPT", output_model=MetaOutput)
    assert out.summary == "这是摘要"
    assert out.novel_info.world_setting == "未知"
