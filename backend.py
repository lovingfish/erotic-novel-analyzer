# -*- coding: utf-8 -*-
"""
小说分析器后端
基于FastAPI的轻量级Web服务
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from dotenv import load_dotenv
import requests

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from novel_analyzer.config_loader import load_llm_config
from novel_analyzer.content_processor import prepare_content
from novel_analyzer.llm_client import LLMClient, LLMRuntime, LLMClientError
from novel_analyzer.prompts import render
from novel_analyzer import llm_dumps
from novel_analyzer.schemas import (
    MetaOutput,
    CoreOutput,
    ScenesOutput,
    ThunderOutput,
    LewdElementsOutput,
    Character,
    Relationship,
)
from novel_analyzer.validators import (
    validate_core_consistency,
    validate_scenes_consistency,
    validate_thunder_consistency,
    validate_lewd_elements_consistency,
)

load_dotenv()

DEBUG = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}

app = FastAPI(
    title="小说分析器",
    description="基于LLM的小说分析工具 - 多角色、多关系、性癖分析",
    version="4.0.0",
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

LLM_CFG = load_llm_config(BASE_DIR)


class AnalyzeContentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str


class AnalyzeScenesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    characters: list[Dict[str, Any]]
    relationships: list[Dict[str, Any]]


class AnalyzeThunderzonesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    characters: list[Dict[str, Any]]
    relationships: list[Dict[str, Any]]


class AnalyzeLewdElementsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    characters: list[Dict[str, Any]]
    relationships: list[Dict[str, Any]]


def _validate_api_url(api_url: str) -> str:
    url = (api_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="API URL必须是http/https且包含主机，例如：https://example.com/v1")
    return url.rstrip("/")


def _get_llm_runtime() -> LLMRuntime:
    api_url = os.getenv("API_BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL_NAME", "").strip()
    if not api_url or not api_key or not model:
        raise HTTPException(status_code=400, detail="服务端未配置API（请在.env中设置API_BASE_URL/API_KEY/MODEL_NAME）")

    return LLMRuntime(api_url=_validate_api_url(api_url), api_key=api_key, model=model)


def _llm_client() -> LLMClient:
    return LLMClient(_get_llm_runtime(), LLM_CFG)


def _raise_errors(section: str, errors: list[str]) -> None:
    if not errors:
        return
    detail = f"{section} 校验失败:\n" + "\n".join(f"- {e}" for e in errors)
    raise HTTPException(status_code=422, detail=detail)


def _raise_pydantic_error(section: str, e: ValidationError) -> None:
    lines: list[str] = []
    for item in e.errors()[:20]:
        loc = ".".join(str(p) for p in (item.get("loc") or []))
        msg = str(item.get("msg") or "")
        if loc:
            lines.append(f"- {loc}: {msg}")
        else:
            lines.append(f"- {msg}")
    if len(e.errors()) > 20:
        lines.append(f"- ...({len(e.errors()) - 20} more)")
    raise HTTPException(status_code=422, detail=f"{section} 输入校验失败:\n" + "\n".join(lines))


def _raise_llm_error(section: str, e: LLMClientError) -> None:
    detail = f"{section} 调用失败: {e}"
    if DEBUG and e.raw_response:
        detail += f"\n\n原始响应(截断):\n{e.raw_response[:2000]}"
    raise HTTPException(status_code=422, detail=detail)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
def get_server_config():
    return {
        "api_url": os.getenv("API_BASE_URL", ""),
        "model": os.getenv("MODEL_NAME", ""),
        "repair_enabled": bool(getattr(LLM_CFG.repair, "enabled", False)),
        "repair_max_attempts": int(getattr(LLM_CFG.repair, "max_attempts", 0)),
        "llm_dump_enabled": bool(llm_dumps.enabled()),
        "llm_dump_dir": str(llm_dumps.dump_dir()),
    }


@app.get("/api/debug/llm-dumps")
def list_llm_dumps(limit: int = 200):
    return {
        "enabled": llm_dumps.enabled(),
        "dir": str(llm_dumps.dump_dir()),
        "items": llm_dumps.list_dumps(limit=limit),
    }


@app.get("/api/debug/llm-dumps/{dump_id}")
def read_llm_dump(dump_id: str):
    try:
        data = llm_dumps.read_dump(dump_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dump 不存在")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 dump 失败: {e}")
    return {"dump": data}


@app.delete("/api/debug/llm-dumps")
def clear_llm_dumps():
    try:
        deleted = llm_dumps.clear_dumps()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空失败: {e}")
    return {"deleted": deleted}


@app.get("/api/test-connection")
def test_connection():
    """测试 API 连接 + Function Calling 支持"""
    runtime = _get_llm_runtime()

    url = f"{runtime.api_url}/chat/completions"
    headers = {"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"}

    tool_name = "ping"
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Return {ok:true} to confirm function calling works.",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }

    payload = {
        "model": runtime.model,
        "messages": [{"role": "user", "content": "Call the function tool ping."}],
        "temperature": 0,
        "stream": False,
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 400 and ("tools" in (res.text or "") or "tool_choice" in (res.text or "")):
            legacy_payload = {
                "model": runtime.model,
                "messages": [{"role": "user", "content": "Call the function ping."}],
                "temperature": 0,
                "stream": False,
                "functions": [tool["function"]],
                "function_call": {"name": tool_name},
            }
            res = requests.post(url, headers=headers, json=legacy_payload, timeout=30)

        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=f"API错误: {res.text}")

        data = res.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        args = None
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            fn = (tool_calls[0].get("function") or {})
            args = fn.get("arguments")
        elif isinstance(message.get("function_call"), dict):
            args = (message.get("function_call") or {}).get("arguments")

        if isinstance(args, str):
            args = json.loads(args)

        if not isinstance(args, dict) or args.get("ok") is not True:
            raise HTTPException(status_code=400, detail="服务端/模型未按 function calling 返回 tool arguments")

        return {"status": "success", "message": "连接成功（Function Calling 正常）"}

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="请求超时")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"连接失败: {str(e)}")


@app.post("/api/analyze/meta")
def analyze_meta(req: AnalyzeContentRequest):
    """分析小说基础信息 + 剧情总结"""
    client = _llm_client()

    sec = LLM_CFG.sections["meta"]
    content = prepare_content(req.content, LLM_CFG, section="meta")
    prompt = render(sec.prompt_template, tool_name=sec.tool_name, content=content)

    try:
        out = client.call_section(section="meta", prompt=prompt, output_model=MetaOutput)
    except LLMClientError as e:
        _raise_llm_error("Meta", e)

    return {"analysis": out.model_dump()}


@app.post("/api/analyze/core")
def analyze_core(req: AnalyzeContentRequest):
    """分析角色 + 关系 + 淫荡指数"""
    client = _llm_client()

    sec = LLM_CFG.sections["core"]
    content = prepare_content(req.content, LLM_CFG, section="core")
    prompt = render(sec.prompt_template, tool_name=sec.tool_name, content=content)

    try:
        out = client.call_section(section="core", prompt=prompt, output_model=CoreOutput)
    except LLMClientError as e:
        _raise_llm_error("Core", e)

    errors = validate_core_consistency(out)
    _raise_errors("Core", errors)

    return {"analysis": out.model_dump(by_alias=True)}


@app.post("/api/analyze/scenes")
def analyze_scenes(req: AnalyzeScenesRequest):
    """分析首次场景 + 统计 + 关系发展"""
    client = _llm_client()

    try:
        characters = [Character.model_validate(c) for c in req.characters]
        relationships = [Relationship.model_validate(r) for r in req.relationships]
    except ValidationError as e:
        _raise_pydantic_error("Scenes 输入", e)

    names = {c.name for c in characters}
    rel_errors: list[str] = []
    for idx, r in enumerate(relationships):
        if r.from_ not in names:
            rel_errors.append(f"relationships[{idx}].from 不在角色表: {r.from_}")
        if r.to not in names:
            rel_errors.append(f"relationships[{idx}].to 不在角色表: {r.to}")
    _raise_errors("Scenes 输入关系", rel_errors)

    allowed_names_json = json.dumps(sorted(names), ensure_ascii=False)
    relationships_json = json.dumps([r.model_dump(by_alias=True) for r in relationships], ensure_ascii=False)

    sec = LLM_CFG.sections["scenes"]
    content = prepare_content(req.content, LLM_CFG, section="scenes")
    prompt = render(
        sec.prompt_template,
        tool_name=sec.tool_name,
        allowed_names_json=allowed_names_json,
        relationships_json=relationships_json,
        content=content,
    )

    try:
        out = client.call_section(section="scenes", prompt=prompt, output_model=ScenesOutput)
    except LLMClientError as e:
        _raise_llm_error("Scenes", e)

    errors = validate_scenes_consistency(out, names)
    _raise_errors("Scenes", errors)

    return {"analysis": out.model_dump()}


@app.post("/api/analyze/thunderzones")
def analyze_thunderzones(req: AnalyzeThunderzonesRequest):
    """分析雷点"""
    client = _llm_client()

    try:
        characters = [Character.model_validate(c) for c in req.characters]
        relationships = [Relationship.model_validate(r) for r in req.relationships]
    except ValidationError as e:
        _raise_pydantic_error("Thunder 输入", e)

    names = {c.name for c in characters}
    rel_errors: list[str] = []
    for idx, r in enumerate(relationships):
        if r.from_ not in names:
            rel_errors.append(f"relationships[{idx}].from 不在角色表: {r.from_}")
        if r.to not in names:
            rel_errors.append(f"relationships[{idx}].to 不在角色表: {r.to}")
    _raise_errors("Thunder 输入关系", rel_errors)

    allowed_names_json = json.dumps(sorted(names), ensure_ascii=False)
    relationships_json = json.dumps([r.model_dump(by_alias=True) for r in relationships], ensure_ascii=False)

    sec = LLM_CFG.sections["thunder"]
    content = prepare_content(req.content, LLM_CFG, section="thunder")
    prompt = render(
        sec.prompt_template,
        tool_name=sec.tool_name,
        allowed_names_json=allowed_names_json,
        relationships_json=relationships_json,
        content=content,
    )

    try:
        out = client.call_section(section="thunder", prompt=prompt, output_model=ThunderOutput)
    except LLMClientError as e:
        _raise_llm_error("Thunder", e)

    errors = validate_thunder_consistency(out, names)
    _raise_errors("Thunder", errors)

    return {"analysis": out.model_dump()}


@app.post("/api/analyze/lewd-elements")
def analyze_lewd_elements(req: AnalyzeLewdElementsRequest):
    """分析涩情元素（非雷点标签）"""
    client = _llm_client()

    try:
        characters = [Character.model_validate(c) for c in req.characters]
        relationships = [Relationship.model_validate(r) for r in req.relationships]
    except ValidationError as e:
        _raise_pydantic_error("LewdElements 输入", e)

    names = {c.name for c in characters}
    rel_errors: list[str] = []
    for idx, r in enumerate(relationships):
        if r.from_ not in names:
            rel_errors.append(f"relationships[{idx}].from 不在角色表: {r.from_}")
        if r.to not in names:
            rel_errors.append(f"relationships[{idx}].to 不在角色表: {r.to}")
    _raise_errors("LewdElements 输入关系", rel_errors)

    allowed_names_json = json.dumps(sorted(names), ensure_ascii=False)

    sec = LLM_CFG.sections["lewd_elements"]
    content = prepare_content(req.content, LLM_CFG, section="lewd_elements")
    prompt = render(
        sec.prompt_template,
        tool_name=sec.tool_name,
        allowed_names_json=allowed_names_json,
        content=content,
    )

    try:
        out = client.call_section(section="lewd_elements", prompt=prompt, output_model=LewdElementsOutput)
    except LLMClientError as e:
        _raise_llm_error("LewdElements", e)

    errors = validate_lewd_elements_consistency(out, names)
    _raise_errors("LewdElements", errors)

    return {"analysis": out.model_dump()}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "6103"))
    log_level = os.getenv("LOG_LEVEL", "warning")
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"\n  ➜  Local:   http://{display_host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
