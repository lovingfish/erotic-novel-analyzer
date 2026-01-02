from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config_loader import LLMConfig
from . import observability
from . import llm_dumps
from .prompts import extract_requirements_excerpt, render, truncate_text


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMRuntime:
    api_url: str
    api_key: str
    model: str


class LLMClientError(Exception):
    def __init__(self, message: str, *, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


def _inline_refs(schema: Any) -> Any:
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_inline_refs(x) for x in schema]
        return schema

    defs = schema.get("$defs")

    if "$ref" in schema and isinstance(schema["$ref"], str) and isinstance(defs, dict):
        ref = schema["$ref"]
        prefix = "#/$defs/"
        if ref.startswith(prefix):
            key = ref[len(prefix) :]
            target = defs.get(key)
            if target is None:
                return schema
            return _inline_refs(target)

    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "$defs":
            continue
        out[k] = _inline_refs(v)
    return out


def _schema_summary(model: type[BaseModel], *, max_chars: int = 1200) -> str:
    schema = _inline_refs(model.model_json_schema())
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = schema.get("required") if isinstance(schema, dict) else None

    lines: list[str] = []
    if isinstance(required, list) and required:
        lines.append("Required keys: " + ", ".join(str(x) for x in required))

    if isinstance(props, dict) and props:
        for k, v in props.items():
            if not isinstance(v, dict):
                continue
            t = v.get("type")
            if not t and "anyOf" in v:
                t = "anyOf"
            lines.append(f"- {k}: {t}")

    text = "\n".join(lines) if lines else json.dumps(schema, ensure_ascii=False)
    return truncate_text(text, max_chars)


def _format_validation_errors(err: ValidationError, *, max_items: int = 20) -> list[str]:
    out: list[str] = []
    for item in err.errors()[:max_items]:
        loc = ".".join(str(p) for p in (item.get("loc") or []))
        msg = str(item.get("msg") or "")
        typ = str(item.get("type") or "")
        if loc:
            out.append(f"{loc}: {msg} ({typ})")
        else:
            out.append(f"{msg} ({typ})")
    if len(err.errors()) > max_items:
        out.append(f"...({len(err.errors()) - max_items} more)")
    return out


def _backoff_seconds(backoff: str, base: float, attempt_index: int, max_wait: float) -> float:
    if backoff == "linear":
        wait = base * (attempt_index + 1)
    else:
        wait = base * (2 ** attempt_index)
    return min(max_wait, max(0.0, float(wait)))


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) < 3:
        return s
    if not lines[-1].strip().startswith("```"):
        return s
    return "\n".join(lines[1:-1]).strip()


def _try_parse_json_dict(text: str) -> dict[str, Any] | None:
    s = _strip_code_fences(text)
    if not s:
        return None
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    left = s.find("{")
    right = s.rfind("}")
    if left >= 0 and right > left:
        candidate = s[left : right + 1]
        try:
            parsed2 = json.loads(candidate)
        except Exception:
            parsed2 = None
        if isinstance(parsed2, dict):
            return parsed2

    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "是", "对"}:
            return True
        if s in {"0", "false", "no", "n", "否", "不"}:
            return False
    return default


def _normalize_meta_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    summary = args.get("summary")
    if isinstance(summary, str) and summary.strip():
        out["summary"] = summary.strip()
    else:
        out["summary"] = ""

    raw_info = args.get("novel_info")
    info: dict[str, Any] = {}
    if isinstance(raw_info, dict):
        info = raw_info
    elif isinstance(raw_info, str):
        parsed = _try_parse_json_dict(raw_info)
        if isinstance(parsed, dict):
            info = parsed

    world_setting = info.get("world_setting")
    if isinstance(world_setting, str) and world_setting.strip():
        ws = world_setting.strip()
    else:
        ws = "未知"

    raw_tags = info.get("world_tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                tags.append(s)
    elif isinstance(raw_tags, str):
        s = raw_tags.strip()
        if s:
            tags = [s]

    chapter_count = info.get("chapter_count")
    try:
        cc = int(chapter_count)
    except Exception:
        cc = 0
    if cc < 0:
        cc = 0

    is_completed = _coerce_bool(info.get("is_completed"), False)

    completion_note = info.get("completion_note")
    note = completion_note.strip() if isinstance(completion_note, str) else ""

    out["novel_info"] = {
        "world_setting": ws,
        "world_tags": tags,
        "chapter_count": cc,
        "is_completed": is_completed,
        "completion_note": note,
    }
    return out


_PARTICIPANTS_SPLIT_RE = re.compile(r"[，,、/+\s]+")


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _normalize_nonempty_str(value: Any, *, default: str) -> str:
    if isinstance(value, str):
        s = value.strip()
        return s if s else default
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _normalize_participants(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        parts = [p.strip() for p in _PARTICIPANTS_SPLIT_RE.split(s) if p and p.strip()]
        return parts

    if value is None:
        return []

    s = str(value).strip()
    return [s] if s else []


def _normalize_scene_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    return {
        "participants": _normalize_participants(value.get("participants")),
        "chapter": _normalize_nonempty_str(value.get("chapter"), default="未知"),
        "location": _normalize_nonempty_str(value.get("location"), default="未知"),
        "description": _normalize_nonempty_str(value.get("description"), default="未提及"),
    }


def _normalize_evolution_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    return {
        "chapter": _normalize_nonempty_str(value.get("chapter"), default="未知"),
        "stage": _normalize_nonempty_str(value.get("stage"), default="未提及"),
        "description": _normalize_nonempty_str(value.get("description"), default="未提及"),
    }


def _normalize_sex_scenes(value: Any) -> Any:
    raw = value
    if isinstance(raw, str):
        parsed = _try_parse_json_dict(raw)
        if parsed is None:
            return value
        raw = parsed

    if not isinstance(raw, dict):
        return value

    raw_scenes = raw.get("scenes")
    items = _coerce_list(raw_scenes)
    scenes = [x for x in (_normalize_scene_entry(item) for item in items) if x is not None]

    tc_raw = raw.get("total_count")
    if tc_raw is None:
        tc_raw = raw.get("total")
    if tc_raw is None:
        tc_raw = raw.get("count")

    try:
        total_count = int(tc_raw)
    except Exception:
        total_count = len(scenes)

    if total_count < 0:
        total_count = 0
    if total_count < len(scenes):
        total_count = len(scenes)

    return {"total_count": total_count, "scenes": scenes}


def _normalize_scenes_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    first_items = _coerce_list(args.get("first_sex_scenes"))
    out["first_sex_scenes"] = [x for x in (_normalize_scene_entry(item) for item in first_items) if x is not None]

    sex_scenes = _normalize_sex_scenes(args.get("sex_scenes"))
    if sex_scenes is not None:
        out["sex_scenes"] = sex_scenes

    evo_items = _coerce_list(args.get("evolution"))
    out["evolution"] = [x for x in (_normalize_evolution_entry(item) for item in evo_items) if x is not None]

    return out


class LLMClient:
    def __init__(self, runtime: LLMRuntime, cfg: LLMConfig):
        self._runtime = runtime
        self._cfg = cfg

    def call_section(self, *, section: str, prompt: str, output_model: type[T]) -> T:
        if section not in self._cfg.sections:
            raise LLMClientError(f"未知 section: {section}")

        sec = self._cfg.sections[section]
        tool = self._build_tool(section=section, output_model=output_model)

        args, raw = self._call_tool_with_retry(
            section=section,
            prompt=prompt,
            tool=tool,
            tool_name=sec.tool_name,
            temperature=sec.temperature,
            stage="primary",
        )

        if args is None:
            validated: T | None = None
            errors = ["未返回 function call arguments 或 arguments 无法解析"]
        else:
            args = self._normalize_args(section=section, args=args)
            validated, errors = self._validate_args(output_model, args)

        if validated is not None:
            return validated

        if not self._cfg.repair.enabled or self._cfg.repair.max_attempts <= 0:
            raise LLMClientError(f"{section} schema 校验失败", raw_response=raw)

        repair_ok = False
        repair_errors: list[str] = errors
        bad_output: Any = args if args is not None else (raw or "")

        for _ in range(int(self._cfg.repair.max_attempts)):
            repair_prompt = self._build_repair_prompt(
                target_section=section,
                tool_name=sec.tool_name,
                original_prompt=prompt,
                output_model=output_model,
                bad_output=bad_output,
                validation_errors=errors,
            )
            repair_args, repair_raw = self._call_tool_with_retry(
                section=section,
                prompt=repair_prompt,
                tool=tool,
                tool_name=sec.tool_name,
                temperature=self._cfg.repair_template.temperature,
                stage="repair",
            )

            if repair_args is None:
                validated = None
                repair_errors = ["Repair 未返回 function call arguments 或 arguments 无法解析"]
                raw = repair_raw
                continue

            repair_args = self._normalize_args(section=section, args=repair_args)
            validated, repair_errors = self._validate_args(output_model, repair_args)
            raw = repair_raw
            if validated is not None:
                repair_ok = True
                break

        observability.repair(section=section, success=repair_ok, reason="schema", errors=repair_errors)

        if validated is not None:
            return validated

        raise LLMClientError(f"{section} Repair 失败（schema 校验仍不通过）", raw_response=raw)

    def _validate_args(self, output_model: type[T], args: dict[str, Any]) -> tuple[T | None, list[str]]:
        try:
            return output_model.model_validate(args), []
        except ValidationError as e:
            return None, _format_validation_errors(e)

    def _normalize_args(self, *, section: str, args: dict[str, Any]) -> dict[str, Any]:
        if section == "meta":
            return _normalize_meta_args(args)
        if section == "scenes":
            return _normalize_scenes_args(args)
        return args

    def _build_tool(self, *, section: str, output_model: type[BaseModel]) -> dict[str, Any]:
        sec = self._cfg.sections[section]
        schema = _inline_refs(output_model.model_json_schema())
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise LLMClientError(f"{section} schema 非 object")

        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": sec.tool_name,
                "description": sec.description,
                "parameters": schema,
            },
        }

    def _build_repair_prompt(
        self,
        *,
        target_section: str,
        tool_name: str,
        original_prompt: str,
        output_model: type[BaseModel],
        bad_output: Any,
        validation_errors: list[str],
    ) -> str:
        original_requirements = extract_requirements_excerpt(original_prompt)
        original_requirements = truncate_text(original_requirements, self._cfg.repair.prompt_head_max_chars)

        schema_summary = _schema_summary(output_model)

        try:
            bad_text = json.dumps(bad_output, ensure_ascii=False)
        except Exception:
            bad_text = str(bad_output)
        bad_text = truncate_text(bad_text, self._cfg.repair.bad_output_max_chars)

        errors_text = "\n".join(f"- {e}" for e in (validation_errors or ["(none)"]))

        return render(
            self._cfg.repair_template.prompt_template,
            target_section=target_section,
            tool_name=tool_name,
            original_requirements=original_requirements,
            schema_summary=schema_summary,
            bad_output=bad_text,
            validation_errors=errors_text,
        )

    def _call_tool_with_retry(
        self,
        *,
        section: str,
        prompt: str,
        tool: dict[str, Any],
        tool_name: str,
        temperature: float,
        stage: str,
    ) -> tuple[dict[str, Any] | None, str]:
        timeout = int(self._cfg.defaults.timeout_seconds)
        retry = self._cfg.defaults.retry

        url = f"{self._runtime.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._runtime.api_key}",
            "Content-Type": "application/json",
        }

        payload_tools = {
            "model": self._runtime.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "stream": False,
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }

        def do_request(payload: dict[str, Any]) -> requests.Response:
            return requests.post(url, headers=headers, json=payload, timeout=timeout)

        last_raw = ""
        last_err = ""
        for attempt in range(int(retry.count)):
            attempt_index = attempt + 1
            try:
                res = do_request(payload_tools)
            except requests.exceptions.Timeout:
                last_err = "timeout"
                wait = _backoff_seconds(retry.backoff, retry.base_wait_seconds, attempt, retry.max_wait_seconds)
                llm_dumps.write_dump(
                    section=section,
                    stage=stage,
                    attempt=attempt_index,
                    protocol="tools",
                    model=self._runtime.model,
                    tool_name=tool_name,
                    temperature=float(temperature),
                    prompt=prompt,
                    request_payload=payload_tools,
                    response_status_code=None,
                    response_text=None,
                    response_json=None,
                    extracted_args=None,
                    note="timeout",
                )
                if attempt < int(retry.count) - 1:
                    observability.retry(
                        section=section,
                        attempt=attempt + 1,
                        max_attempts=int(retry.count),
                        reason=last_err,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                raise LLMClientError(f"{section} 调用超时")
            except requests.RequestException as e:
                raise LLMClientError(f"{section} 请求失败: {e}")

            response_text = res.text or ""
            last_raw = response_text[:3000]

            response_json: Any | None
            try:
                response_json = res.json()
            except Exception:
                response_json = None
            protocol = "tools"
            request_payload: dict[str, Any] | None = payload_tools
            protocol_fallback = False

            if res.status_code == 400:
                err_text = response_text
                if "tools" in err_text or "tool_choice" in err_text:
                    observability.function_calling_protocol_fallback(
                        section=section,
                        reason="server rejects tools/tool_choice, trying legacy functions/function_call",
                    )
                    llm_dumps.write_dump(
                        section=section,
                        stage=stage,
                        attempt=attempt_index,
                        protocol="tools",
                        model=self._runtime.model,
                        tool_name=tool_name,
                        temperature=float(temperature),
                        prompt=prompt,
                        request_payload=payload_tools,
                        response_status_code=int(res.status_code),
                        response_text=response_text,
                        response_json=response_json,
                        extracted_args=None,
                        note="protocol fallback trigger",
                    )
                    protocol_fallback = True
                    legacy_payload = {
                        "model": self._runtime.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": float(temperature),
                        "stream": False,
                        "functions": [tool.get("function") or {}],
                        "function_call": {"name": tool_name},
                    }
                    res = do_request(legacy_payload)
                    protocol = "legacy"
                    request_payload = legacy_payload

                    response_text = res.text or ""
                    last_raw = response_text[:3000]

                    try:
                        response_json = res.json()
                    except Exception:
                        response_json = None

            extracted_args: dict[str, Any] | None = None
            notes: list[str] = []
            if protocol_fallback:
                notes.append("protocol fallback")
            if res.status_code != 200:
                notes.append(f"status {res.status_code}")
            if response_json is None:
                notes.append("json parse failed")

            if res.status_code == 200 and isinstance(response_json, dict):
                extracted_args = self._extract_tool_arguments(response_json, tool_name=tool_name, section=section)
                if extracted_args is None:
                    notes.append("tool arguments missing/unparsable")

            llm_dumps.write_dump(
                section=section,
                stage=stage,
                attempt=attempt_index,
                protocol=protocol,
                model=self._runtime.model,
                tool_name=tool_name,
                temperature=float(temperature),
                prompt=prompt,
                request_payload=request_payload,
                response_status_code=int(res.status_code),
                response_text=response_text,
                response_json=response_json,
                extracted_args=extracted_args,
                note="; ".join(notes),
            )

            if res.status_code in set(retry.retryable_status_codes):
                last_err = f"http_{res.status_code}"
                wait = _backoff_seconds(retry.backoff, retry.base_wait_seconds, attempt, retry.max_wait_seconds)
                if attempt < int(retry.count) - 1:
                    observability.retry(
                        section=section,
                        attempt=attempt + 1,
                        max_attempts=int(retry.count),
                        reason=last_err,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue

            if res.status_code != 200:
                raise LLMClientError(f"{section} API错误: {res.status_code}", raw_response=last_raw)

            if not isinstance(response_json, dict):
                raise LLMClientError(f"{section} 返回非JSON", raw_response=last_raw)

            args = extracted_args
            if args is None:
                observability.missing_tool_call(section=section, reason="no tool_calls/function_call or unparsable arguments")
                return None, last_raw

            return args, last_raw

        raise LLMClientError(f"{section} 调用失败: {last_err}", raw_response=last_raw)

    def _extract_tool_arguments(self, data: dict[str, Any], *, tool_name: str, section: str) -> dict[str, Any] | None:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                fn = tc.get("function") or {}
                if str(fn.get("name") or "").strip() != tool_name:
                    continue
                args = fn.get("arguments")
                return self._parse_arguments(args)

        fn_call = message.get("function_call")
        if isinstance(fn_call, dict) and str(fn_call.get("name") or "").strip() == tool_name:
            return self._parse_arguments(fn_call.get("arguments"))

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parsed = _try_parse_json_dict(content)
            if parsed is not None:
                return parsed
            if section == "meta":
                return {"novel_info": {}, "summary": content.strip()}

        return None

    def _parse_arguments(self, args: Any) -> dict[str, Any] | None:
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            s = args.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None
