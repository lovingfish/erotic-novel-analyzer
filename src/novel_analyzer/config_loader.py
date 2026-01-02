from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONFIG_REL_PATH = Path("config") / "llm.yaml"


@dataclass(frozen=True)
class RetryPolicy:
    count: int
    backoff: str
    base_wait_seconds: float
    max_wait_seconds: float
    retryable_status_codes: tuple[int, ...]


@dataclass(frozen=True)
class DefaultsConfig:
    timeout_seconds: int
    retry: RetryPolicy


@dataclass(frozen=True)
class ContentProcessingConfig:
    max_chars: int
    strategy: str
    boundary_aware: bool
    boundary_search_window: int
    truncation_marker_template: str


@dataclass(frozen=True)
class RepairConfig:
    enabled: bool
    max_attempts: int
    prompt_head_max_chars: int
    bad_output_max_chars: int


@dataclass(frozen=True)
class SectionConfig:
    temperature: float
    tool_name: str
    description: str
    prompt_template: str


@dataclass(frozen=True)
class RepairTemplateConfig:
    temperature: float
    prompt_template: str


@dataclass(frozen=True)
class LLMConfig:
    defaults: DefaultsConfig
    content_processing: ContentProcessingConfig
    repair: RepairConfig
    sections: dict[str, SectionConfig]
    repair_template: RepairTemplateConfig


def _require_dict(obj: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"配置解析失败：{ctx} 不是对象")
    return obj


def _require_str(obj: Any, ctx: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ValueError(f"配置解析失败：{ctx} 不是非空字符串")
    return obj


def _require_int(obj: Any, ctx: str) -> int:
    try:
        return int(obj)
    except Exception as e:
        raise ValueError(f"配置解析失败：{ctx} 不是整数") from e


def _require_float(obj: Any, ctx: str) -> float:
    try:
        return float(obj)
    except Exception as e:
        raise ValueError(f"配置解析失败：{ctx} 不是数字") from e


def _read_text(path: Path, ctx: str) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"配置解析失败：找不到文件 {ctx}: {path}")
    return path.read_text(encoding="utf-8")


def _env_bool(name: str) -> bool | None:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _env_int(name: str) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def load_llm_config(repo_root: Path) -> LLMConfig:
    config_path = (repo_root / CONFIG_REL_PATH).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    root = _require_dict(raw, "root")

    defaults_raw = _require_dict(root.get("defaults"), "defaults")
    timeout_seconds = _require_int(defaults_raw.get("timeout_seconds"), "defaults.timeout_seconds")

    retry_raw = _require_dict(defaults_raw.get("retry"), "defaults.retry")
    retry_count = _require_int(retry_raw.get("count"), "defaults.retry.count")
    backoff = (_require_str(retry_raw.get("backoff"), "defaults.retry.backoff").strip().lower())
    if backoff not in {"exponential", "linear"}:
        raise ValueError("配置解析失败：defaults.retry.backoff 仅支持 exponential|linear")
    base_wait_seconds = float(retry_raw.get("base_wait_seconds", 2))
    max_wait_seconds = float(retry_raw.get("max_wait_seconds", 20))

    status_codes_raw = retry_raw.get("retryable_status_codes")
    if not isinstance(status_codes_raw, list) or not status_codes_raw:
        raise ValueError("配置解析失败：defaults.retry.retryable_status_codes 必须是非空数组")
    retryable_status_codes = tuple(int(x) for x in status_codes_raw)

    defaults_cfg = DefaultsConfig(
        timeout_seconds=timeout_seconds,
        retry=RetryPolicy(
            count=retry_count,
            backoff=backoff,
            base_wait_seconds=base_wait_seconds,
            max_wait_seconds=max_wait_seconds,
            retryable_status_codes=retryable_status_codes,
        ),
    )

    cp_raw = _require_dict(root.get("content_processing"), "content_processing")
    cp_cfg = ContentProcessingConfig(
        max_chars=_require_int(cp_raw.get("max_chars"), "content_processing.max_chars"),
        strategy=_require_str(cp_raw.get("strategy"), "content_processing.strategy").strip().lower(),
        boundary_aware=bool(cp_raw.get("boundary_aware", True)),
        boundary_search_window=_require_int(
            cp_raw.get("boundary_search_window", 200),
            "content_processing.boundary_search_window",
        ),
        truncation_marker_template=_require_str(
            cp_raw.get("truncation_marker_template"),
            "content_processing.truncation_marker_template",
        ),
    )

    repair_raw = _require_dict(root.get("repair"), "repair")
    repair_enabled = bool(repair_raw.get("enabled", True))
    env_enabled = _env_bool("LLM_REPAIR_ENABLED")
    if env_enabled is not None:
        repair_enabled = env_enabled

    repair_max_attempts = _require_int(repair_raw.get("max_attempts", 1), "repair.max_attempts")
    env_max_attempts = _env_int("LLM_REPAIR_MAX_ATTEMPTS")
    if env_max_attempts is not None:
        repair_max_attempts = env_max_attempts

    repair_cfg = RepairConfig(
        enabled=repair_enabled,
        max_attempts=repair_max_attempts,
        prompt_head_max_chars=_require_int(repair_raw.get("prompt_head_max_chars", 8000), "repair.prompt_head_max_chars"),
        bad_output_max_chars=_require_int(repair_raw.get("bad_output_max_chars", 6000), "repair.bad_output_max_chars"),
    )

    sections_raw = _require_dict(root.get("sections"), "sections")

    config_dir = config_path.parent

    repair_section_raw = _require_dict(sections_raw.get("repair"), "sections.repair")
    repair_prompt_file = Path(_require_str(repair_section_raw.get("prompt_file"), "sections.repair.prompt_file"))
    repair_template = RepairTemplateConfig(
        temperature=_require_float(repair_section_raw.get("temperature"), "sections.repair.temperature"),
        prompt_template=_read_text(config_dir / repair_prompt_file, "repair prompt_file"),
    )

    required_sections = ["meta", "core", "scenes", "thunder"]
    sections: dict[str, SectionConfig] = {}
    for name in required_sections:
        sec_raw = _require_dict(sections_raw.get(name), f"sections.{name}")
        prompt_file = Path(_require_str(sec_raw.get("prompt_file"), f"sections.{name}.prompt_file"))
        sections[name] = SectionConfig(
            temperature=_require_float(sec_raw.get("temperature"), f"sections.{name}.temperature"),
            tool_name=_require_str(sec_raw.get("tool_name"), f"sections.{name}.tool_name"),
            description=_require_str(sec_raw.get("description"), f"sections.{name}.description"),
            prompt_template=_read_text(config_dir / prompt_file, f"sections.{name}.prompt_file"),
        )

    return LLMConfig(
        defaults=defaults_cfg,
        content_processing=cp_cfg,
        repair=repair_cfg,
        sections=sections,
        repair_template=repair_template,
    )
