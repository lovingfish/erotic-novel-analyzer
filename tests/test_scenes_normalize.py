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
from novel_analyzer.schemas import ScenesOutput


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
            "scenes": SectionConfig(
                temperature=0.0,
                tool_name="extract_scenes",
                description="scenes",
                prompt_template="x",
            )
        },
        repair_template=RepairTemplateConfig(
            temperature=0.0,
            prompt_template="x",
        ),
    )


def test_scenes_normalize_parses_stringified_sex_scenes(monkeypatch):
    runtime = LLMRuntime(api_url="http://example.com/v1", api_key="sk", model="m")
    cfg = _make_cfg()
    client = LLMClient(runtime, cfg)

    bad_args = {
        "first_sex_scenes": [
            {
                "chapter": "未知",
                "location": "芝加哥",
                "description": "餐厅挑逗",
                "participants": ["零", "我"],
                "type": "足交挑逗",
            }
        ],
        "sex_scenes": json.dumps(
            {
                "total_count": 1,
                "scenes": [
                    {
                        "chapter": "未知",
                        "location": "酒店",
                        "description": "寸止",
                        "participants": ["零", "我"],
                        "type": "腋交寸止",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        "evolution": [{"chapter": "未知", "stage": "确立", "description": "关系变化", "from": "零"}],
        "extra_top_level": "x",
    }

    def make_data(args: dict):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "extract_scenes",
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                }
                            }
                        ]
                    }
                }
            ]
        }

    resp_obj = make_data(bad_args)

    responses = [_FakeResponse(200, text=json.dumps(resp_obj, ensure_ascii=False), json_obj=resp_obj)]

    def fake_post(url, headers=None, json=None, timeout=None):
        return responses.pop(0)

    import novel_analyzer.llm_client as llm_client_mod

    monkeypatch.setattr(llm_client_mod.requests, "post", fake_post)

    out = client.call_section(section="scenes", prompt="REQ\n\n## Novel Content\nX", output_model=ScenesOutput)
    assert out.sex_scenes.total_count == 1
    assert out.sex_scenes.scenes[0].location == "酒店"
    assert out.first_sex_scenes[0].participants == ["零", "我"]
    assert out.evolution[0].stage == "确立"

