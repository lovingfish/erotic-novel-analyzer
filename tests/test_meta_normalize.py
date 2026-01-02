from __future__ import annotations

import json
import sys
from pathlib import Path


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


def _make_cfg() -> LLMConfig:
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
        repair=RepairConfig(enabled=False, max_attempts=0, prompt_head_max_chars=5000, bad_output_max_chars=5000),
        sections={
            "meta": SectionConfig(
                temperature=0.0,
                tool_name="extract_meta",
                description="meta",
                prompt_template="x",
            )
        },
        repair_template=RepairTemplateConfig(
            temperature=0.0,
            prompt_template="x",
        ),
    )


def test_meta_normalize_parses_stringified_novel_info(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg()
    client = LLMClient(runtime, cfg)

    novel_info_obj = {
        "world_setting": "修仙/架空",
        "world_tags": ["修仙", "宗门", "双修"],
        "chapter_count": 68,
        "is_completed": False,
        "completion_note": "连载中",
    }

    bad_args = {
        "novel_info": json.dumps(novel_info_obj, ensure_ascii=False),
        "summary": "这是一段摘要。",
    }

    resp_obj = {
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

    responses = [_FakeResponse(200, text=json.dumps(resp_obj, ensure_ascii=False), json_obj=resp_obj)]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    import novel_analyzer.llm_client as llm_client_mod

    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="meta", prompt="REQ\n\n## Novel Content\nX", output_model=MetaOutput)
    assert out.novel_info.world_setting == "修仙/架空"
    assert out.novel_info.world_tags == ["修仙", "宗门", "双修"]
    assert out.novel_info.chapter_count == 68
    assert out.novel_info.is_completed is False
    assert out.novel_info.completion_note == "连载中"
    assert out.summary == "这是一段摘要。"

