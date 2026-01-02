from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _is_truthy(value: str) -> bool:
    s = (value or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def enabled() -> bool:
    raw = os.getenv("LLM_DUMP_ENABLED")
    if raw is None or not str(raw).strip():
        return _is_truthy(os.getenv("DEBUG", ""))
    return _is_truthy(str(raw))


def dump_dir() -> Path:
    raw = (os.getenv("LLM_DUMP_DIR") or "").strip()
    base = Path(raw) if raw else Path("llm_dumps")
    return base.expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return value
    s = value or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _sanitize_request_payload(payload: dict[str, Any] | None, *, max_message_chars: int) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    out: dict[str, Any] = dict(payload)
    messages = out.get("messages")
    if isinstance(messages, list):
        new_messages: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = _truncate(content, max_message_chars)
                if len(content) > max_message_chars:
                    new_msg["content"] += f"\n...[TRUNCATED {len(content)} chars]..."
            new_messages.append(new_msg)
        out["messages"] = new_messages
    return out


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_dump(
    *,
    section: str,
    stage: str,
    attempt: int,
    protocol: str,
    model: str,
    tool_name: str,
    temperature: float,
    prompt: str,
    request_payload: dict[str, Any] | None,
    response_status_code: int | None,
    response_text: str | None,
    response_json: Any | None,
    extracted_args: dict[str, Any] | None,
    note: str = "",
) -> str | None:
    if not enabled():
        return None

    max_prompt_chars = _env_int("LLM_DUMP_MAX_PROMPT_CHARS", 12000)
    max_response_chars = _env_int("LLM_DUMP_MAX_RESPONSE_CHARS", 30000)

    out_dir = dump_dir()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    now = _now_utc()
    ts_compact = now.strftime("%Y%m%dT%H%M%S.%fZ")
    nonce = os.urandom(4).hex()

    safe_section = "".join(ch for ch in section if ch.isalnum() or ch in {"-", "_"}).strip() or "section"
    safe_stage = "".join(ch for ch in stage if ch.isalnum() or ch in {"-", "_"}).strip() or "stage"
    safe_protocol = "".join(ch for ch in protocol if ch.isalnum() or ch in {"-", "_"}).strip() or "protocol"

    filename = f"{ts_compact}_{safe_section}_{safe_stage}_{safe_protocol}_a{int(attempt)}_{nonce}.json"
    path = out_dir / filename

    record = {
        "ts": now.isoformat(),
        "section": section,
        "stage": stage,
        "attempt": attempt,
        "protocol": protocol,
        "model": model,
        "tool_name": tool_name,
        "temperature": temperature,
        "prompt": _truncate(prompt or "", max_prompt_chars),
        "request_payload": _sanitize_request_payload(request_payload, max_message_chars=max_prompt_chars),
        "response_status_code": response_status_code,
        "response_text": _truncate(response_text or "", max_response_chars),
        "response_json": response_json,
        "extracted_args": extracted_args,
        "note": note or "",
    }

    try:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return None

    return filename


def _is_safe_dump_id(dump_id: str) -> bool:
    if not dump_id or not isinstance(dump_id, str):
        return False
    if "/" in dump_id or "\\" in dump_id:
        return False
    if ".." in dump_id:
        return False
    return dump_id.endswith(".json")


def list_dumps(*, limit: int = 200) -> list[dict[str, Any]]:
    out_dir = dump_dir()
    if not out_dir.exists() or not out_dir.is_dir():
        return []

    files = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for p in files[: max(0, int(limit))]:
        try:
            st = p.stat()
        except Exception:
            continue
        out.append(
            {
                "id": p.name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return out


def read_dump(dump_id: str) -> dict[str, Any]:
    if not _is_safe_dump_id(dump_id):
        raise ValueError("非法 dump id")

    out_dir = dump_dir()
    path = (out_dir / dump_id).resolve()
    if out_dir.resolve() not in path.parents:
        raise ValueError("非法 dump 路径")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("dump 不存在")

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("dump 内容不是对象")
    return data


def clear_dumps() -> int:
    out_dir = dump_dir()
    if not out_dir.exists() or not out_dir.is_dir():
        return 0

    count = 0
    for p in out_dir.glob("*.json"):
        try:
            p.unlink()
            count += 1
        except Exception:
            continue
    return count
