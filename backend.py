# -*- coding: utf-8 -*-
"""
小说分析器后端
基于FastAPI的轻量级Web服务
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
import requests

load_dotenv()

DEBUG = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}

app = FastAPI(
    title="小说分析器",
    description="基于LLM的小说分析工具 - 多角色、多关系、性癖分析",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NOVEL_PATH = Path(os.getenv("NOVEL_PATH", str(BASE_DIR.parent))).resolve()


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


def _get_llm_config() -> tuple[str, str, str]:
    api_url = os.getenv("API_BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL_NAME", "").strip()
    if not api_url or not api_key or not model:
        raise HTTPException(status_code=400, detail="服务端未配置API（请在.env中设置API_BASE_URL/API_KEY/MODEL_NAME）")
    return _validate_api_url(api_url), api_key, model


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """从LLM响应中提取JSON"""
    if not text or len(text.strip()) < 5:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        pass

    return {}


class LLMCallError(Exception):
    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


def call_llm_with_response(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    retry_count: int = 3,
    *,
    timeout: int = 180,
    temperature: float = 1.0,
) -> tuple[str, str]:
    last_error = None
    last_raw_response = ""

    for attempt in range(retry_count):
        try:
            response = requests.post(
                f"{api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=timeout,
                stream=True,
            )

            if response.status_code != 200:
                last_raw_response = (response.text or "")[:3000]
                if response.status_code in [502, 503, 504]:
                    wait_time = 5 * (attempt + 1)
                    last_error = f"网关超时({response.status_code})"
                    time.sleep(wait_time)
                    continue

                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    last_error = "请求频繁(429)"
                    time.sleep(wait_time)
                    continue

                error_msg = f"API错误: {response.status_code}"
                if response.status_code == 401:
                    error_msg = "API密钥无效"
                elif response.status_code == 403:
                    error_msg = "无权限访问"
                elif response.status_code == 400:
                    error_msg = "请求参数错误"
                elif response.status_code == 421:
                    error_msg = "内容审核拦截(421)"
                raise LLMCallError(error_msg, last_raw_response)

            content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                line_text = line.decode("utf-8").strip()
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        delta = data_json.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content", "") or delta.get("reasoning_content", "") or ""
                        content += chunk
                    except json.JSONDecodeError:
                        continue

            last_raw_response = content[:3000]

            if not content or len(content.strip()) < 10:
                raise LLMCallError("返回内容过短", last_raw_response)

            return content, last_raw_response

        except requests.exceptions.Timeout:
            last_error = f"超时({attempt + 1}/{retry_count})"
            if attempt < retry_count - 1:
                time.sleep(3)
                continue
            raise LLMCallError(last_error, last_raw_response)
        except LLMCallError:
            raise
        except Exception as e:
            raise LLMCallError(str(e), last_raw_response)

    raise LLMCallError(last_error or "调用失败", last_raw_response)


def _call_llm_json(api_url: str, api_key: str, model: str, prompt: str, section: str) -> Dict[str, Any]:
    raw_response = ""
    try:
        content, raw_response = call_llm_with_response(
            api_url, api_key, model, prompt, temperature=0.7
        )
    except LLMCallError as e:
        detail = f"{section} 调用失败: {e}"
        if e.raw_response and DEBUG:
            detail += f"\n\n原始响应:\n{e.raw_response[:2000]}"
        raise HTTPException(status_code=422, detail=detail)

    data = extract_json_from_response(content)
    if not data:
        detail = f"{section} 解析失败: 返回非JSON或字段缺失"
        if DEBUG and raw_response:
            detail += f"\n\n原始响应:\n{raw_response[:2000]}"
        raise HTTPException(status_code=422, detail=detail)
    return data


def _normalize_gender(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"

    male_values = {"male", "man", "m", "男", "男性", "男生", "男子", "男主", "男的", "男性角色"}
    female_values = {"female", "woman", "f", "女", "女性", "女生", "女子", "女主", "女的", "女性角色"}

    if raw in male_values:
        return "male"
    if raw in female_values:
        return "female"

    if "男" in raw and "女" not in raw:
        return "male"
    if "女" in raw and "男" not in raw:
        return "female"

    return "unknown"


def _normalize_severity(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    lower = raw.lower()
    if raw in {"高", "中", "低"}:
        return raw
    if lower in {"high", "h"}:
        return "高"
    if lower in {"medium", "mid", "m"}:
        return "中"
    if lower in {"low", "l"}:
        return "低"
    return ""


def _stringify(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _validate_characters(characters: Any) -> tuple[list[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    cleaned: list[Dict[str, Any]] = []

    if not isinstance(characters, list):
        return [], ["characters 不是数组"]

    seen_names: set[str] = set()
    for idx, c in enumerate(characters):
        if not isinstance(c, dict):
            errors.append(f"characters[{idx}] 不是对象")
            continue
        name = _stringify(c.get("name"))
        if not name:
            errors.append(f"characters[{idx}] 缺少 name")
            continue
        if name in seen_names:
            errors.append(f"characters[{idx}] 名字重复: {name}")
        seen_names.add(name)

        raw_gender = c.get("gender")
        gender = _normalize_gender(raw_gender)
        if gender == "unknown":
            errors.append(f"characters[{idx}] 性别不规范: {name}({raw_gender})")

        identity = _stringify(c.get("identity"))
        personality = _stringify(c.get("personality"))
        sexual_preferences = _stringify(c.get("sexual_preferences"))
        if not identity:
            errors.append(f"characters[{idx}] 缺少 identity: {name}")
        if not personality:
            errors.append(f"characters[{idx}] 缺少 personality: {name}")
        if not sexual_preferences:
            errors.append(f"characters[{idx}] 缺少 sexual_preferences: {name}")

        cleaned_char = {
            "name": name,
            "gender": gender,
            "identity": identity,
            "personality": personality,
            "sexual_preferences": sexual_preferences,
        }

        if gender == "female":
            lewdness_score = c.get("lewdness_score")
            lewdness_analysis = _stringify(c.get("lewdness_analysis"))
            if lewdness_score is None or lewdness_analysis == "":
                errors.append(f"characters[{idx}] 女性角色缺少淫荡指数: {name}")
            try:
                cleaned_char["lewdness_score"] = int(lewdness_score)
            except Exception:
                errors.append(f"characters[{idx}] 淫荡指数非整数: {name}")
                cleaned_char["lewdness_score"] = lewdness_score
            cleaned_char["lewdness_analysis"] = lewdness_analysis

        cleaned.append(cleaned_char)

    return cleaned, errors


def _validate_relationships(relationships: Any, character_names: set[str]) -> tuple[list[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    cleaned: list[Dict[str, Any]] = []

    if not isinstance(relationships, list):
        return [], ["relationships 不是数组"]

    for idx, r in enumerate(relationships):
        if not isinstance(r, dict):
            errors.append(f"relationships[{idx}] 不是对象")
            continue
        frm = _stringify(r.get("from"))
        to = _stringify(r.get("to"))
        if not frm or not to:
            errors.append(f"relationships[{idx}] 缺少 from/to")
        if frm and frm not in character_names:
            errors.append(f"relationships[{idx}] from 不在角色表: {frm}")
        if to and to not in character_names:
            errors.append(f"relationships[{idx}] to 不在角色表: {to}")

        rel = {
            "from": frm,
            "to": to,
            "type": _stringify(r.get("type")),
            "start_way": _stringify(r.get("start_way")),
            "description": _stringify(r.get("description")),
        }
        if not rel["type"]:
            errors.append(f"relationships[{idx}] 缺少 type: {frm}-{to}")
        if not rel["start_way"]:
            errors.append(f"relationships[{idx}] 缺少 start_way: {frm}-{to}")
        if not rel["description"]:
            errors.append(f"relationships[{idx}] 缺少 description: {frm}-{to}")
        cleaned.append(rel)

    return cleaned, errors


def _validate_scenes(scenes: Any, scene_name: str, character_names: set[str]) -> tuple[list[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    cleaned: list[Dict[str, Any]] = []

    if not isinstance(scenes, list):
        return [], [f"{scene_name} 不是数组"]

    for idx, s in enumerate(scenes):
        if not isinstance(s, dict):
            errors.append(f"{scene_name}[{idx}] 不是对象")
            continue
        participants = s.get("participants")
        if not isinstance(participants, list) or not participants:
            errors.append(f"{scene_name}[{idx}] participants 为空")
            participants_list: list[str] = []
        else:
            participants_list = [_stringify(p) for p in participants if _stringify(p)]
            for p in participants_list:
                if p not in character_names:
                    errors.append(f"{scene_name}[{idx}] 参与者不在角色表: {p}")

        chapter = _stringify(s.get("chapter"))
        location = _stringify(s.get("location"))
        description = _stringify(s.get("description"))
        if not chapter:
            errors.append(f"{scene_name}[{idx}] 缺少 chapter（可用“未知”）")
        if not location:
            errors.append(f"{scene_name}[{idx}] 缺少 location（可用“未知”）")
        if not description:
            errors.append(f"{scene_name}[{idx}] 缺少 description")

        cleaned.append({
            "participants": participants_list,
            "chapter": chapter,
            "location": location,
            "description": description,
        })

    return cleaned, errors


def _validate_evolution(evolution: Any) -> tuple[list[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    cleaned: list[Dict[str, Any]] = []

    if not isinstance(evolution, list):
        return [], ["evolution 不是数组"]

    for idx, item in enumerate(evolution):
        if not isinstance(item, dict):
            errors.append(f"evolution[{idx}] 不是对象")
            continue
        chapter = _stringify(item.get("chapter"))
        stage = _stringify(item.get("stage"))
        description = _stringify(item.get("description"))
        if not chapter:
            errors.append(f"evolution[{idx}] 缺少 chapter（可用“未知”）")
        if not stage:
            errors.append(f"evolution[{idx}] 缺少 stage")
        if not description:
            errors.append(f"evolution[{idx}] 缺少 description")
        cleaned.append({
            "chapter": chapter,
            "stage": stage,
            "description": description,
        })

    return cleaned, errors


def _validate_thunderzones(thunderzones: Any, character_names: set[str]) -> tuple[list[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    cleaned: list[Dict[str, Any]] = []

    if not isinstance(thunderzones, list):
        return [], ["thunderzones 不是数组"]

    for idx, tz in enumerate(thunderzones):
        if not isinstance(tz, dict):
            errors.append(f"thunderzones[{idx}] 不是对象")
            continue
        tz_type = _stringify(tz.get("type"))
        severity = _normalize_severity(tz.get("severity"))
        description = _stringify(tz.get("description"))
        involved = tz.get("involved_characters")
        if not tz_type:
            errors.append(f"thunderzones[{idx}] 缺少 type")
        if not severity:
            errors.append(f"thunderzones[{idx}] severity 不规范")
        if not description:
            description = "未知"
        if not isinstance(involved, list) or not involved:
            errors.append(f"thunderzones[{idx}] involved_characters 为空")
            involved_names: list[str] = []
        else:
            involved_names = [_stringify(p) for p in involved if _stringify(p)]
            for p in involved_names:
                if p not in character_names:
                    errors.append(f"thunderzones[{idx}] 角色不在角色表: {p}")
        chapter_location = _stringify(tz.get("chapter_location"))
        relationship_context = _stringify(tz.get("relationship_context"))
        cleaned.append({
            "type": tz_type,
            "severity": severity,
            "description": description,
            "involved_characters": involved_names,
            "chapter_location": chapter_location,
            "relationship_context": relationship_context,
        })

    return cleaned, errors


def _raise_if_errors(errors: list[str], section: str) -> None:
    if not errors:
        return
    detail = f"{section} 校验失败:\n" + "\n".join(f"- {e}" for e in errors)
    raise HTTPException(status_code=422, detail=detail)


def _validate_meta(data: Dict[str, Any]) -> tuple[Dict[str, Any], str, list[str]]:
    errors: list[str] = []
    novel_info = data.get("novel_info")
    if not isinstance(novel_info, dict):
        return {}, "", ["novel_info 不是对象"]

    world_setting = _stringify(novel_info.get("world_setting"))
    world_tags_raw = novel_info.get("world_tags")
    if not isinstance(world_tags_raw, list):
        errors.append("novel_info.world_tags 不是数组")
        world_tags = []
    else:
        world_tags = [_stringify(tag) for tag in world_tags_raw if _stringify(tag)]

    chapter_count_raw = novel_info.get("chapter_count", 0)
    try:
        chapter_count = int(chapter_count_raw)
    except Exception:
        errors.append("novel_info.chapter_count 非整数")
        chapter_count = 0

    is_completed_raw = novel_info.get("is_completed")
    if isinstance(is_completed_raw, bool):
        is_completed = is_completed_raw
    elif isinstance(is_completed_raw, str):
        if is_completed_raw.strip().lower() in {"true", "yes", "1"}:
            is_completed = True
        elif is_completed_raw.strip().lower() in {"false", "no", "0"}:
            is_completed = False
        else:
            errors.append("novel_info.is_completed 非布尔值")
            is_completed = False
    else:
        errors.append("novel_info.is_completed 非布尔值")
        is_completed = False

    completion_note = _stringify(novel_info.get("completion_note"))
    summary = _stringify(data.get("summary"))
    if not summary:
        errors.append("summary 为空")

    cleaned_info = {
        "world_setting": world_setting,
        "world_tags": world_tags,
        "chapter_count": chapter_count,
        "is_completed": is_completed,
        "completion_note": completion_note,
    }
    return cleaned_info, summary, errors


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _validate_api_url(api_url: str) -> str:
    url = (api_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="API URL必须是http/https且包含主机，例如：https://example.com/v1")
    return url.rstrip("/")


def _safe_novel_path(base_path: Path, user_path: str) -> Path:
    base = base_path.resolve()
    raw = (user_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="路径不能为空")

    try:
        candidate = (base / raw).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="非法路径")

    try:
        candidate.relative_to(base)
    except Exception:
        raise HTTPException(status_code=403, detail="非法路径")

    if candidate.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="仅支持.txt文件")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return candidate


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
def get_server_config():
    return {
        "api_url": os.getenv("API_BASE_URL", ""),
        "model": os.getenv("MODEL_NAME", "")
    }


@app.get("/api/novels")
def scan_novels():
    """递归扫描所有.txt小说文件"""
    if not DEFAULT_NOVEL_PATH.exists():
        raise HTTPException(status_code=400, detail=f"小说目录不存在: {DEFAULT_NOVEL_PATH}（可通过NOVEL_PATH配置）")

    base_path = str(DEFAULT_NOVEL_PATH)

    exclude_keywords = {'venv', '__pycache__', '.git', 'node_modules', 'pip', 'site-packages', 'dist-info', '.tox', '.eggs', 'novel-analyzer'}

    novels = []

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in exclude_keywords]

        folder_name = os.path.basename(root)

        if root == base_path:
            continue

        root_lower = root.lower()
        if any(keyword in root_lower for keyword in exclude_keywords):
            continue

        txt_files = []
        for f in files:
            if f.endswith('.txt') and not f.startswith('.'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_path)
                txt_files.append({
                    "name": f,
                    "path": rel_path,
                    "size": os.path.getsize(full_path)
                })

        if txt_files:
            novels.append({
                "folder": folder_name,
                "path": folder_name,
                "files": sorted(txt_files, key=lambda x: x['name'])
            })

    return {"novels": novels, "total": sum(len(n['files']) for n in novels)}


@app.get("/api/novel/{path:path}")
def read_novel(path: str):
    """读取指定小说内容（限制长度）"""
    full_path = _safe_novel_path(DEFAULT_NOVEL_PATH, path)

    try:
        content = full_path.read_text(encoding="utf-8", errors="ignore")

        return {
            "name": full_path.name,
            "path": str(Path(path).as_posix()),
            "content": content,
            "length": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取失败: {str(e)}")


@app.get("/api/test-connection")
def test_connection():
    """测试API连接"""
    api_url, api_key, model = _get_llm_config()

    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            },
            timeout=30
        )

        if response.status_code == 200:
            return {"status": "success", "message": "连接成功"}
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', response.text)
            except:
                error_msg = response.text
            raise HTTPException(status_code=response.status_code, detail=f"API错误: {error_msg}")

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="请求超时")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"连接失败: {str(e)}")


@app.post("/api/analyze/meta")
def analyze_meta(req: AnalyzeContentRequest):
    """分析小说基础信息 + 剧情总结"""
    api_url, api_key, model = _get_llm_config()

    prompt = f"""
You are a professional literary analyst specializing in adult fiction.

## Task
Extract novel metadata and write a 200-300 Chinese character summary.

## Output JSON ONLY
- Return ONLY one JSON object. No extra text.
- Use double quotes for keys/strings.
- Required keys: novel_info, summary.

### Schema
{{
  "novel_info": {{
    "world_setting": "",
    "world_tags": [],
    "chapter_count": 0,
    "is_completed": false,
    "completion_note": ""
  }},
  "summary": ""
}}

## Novel Content
{req.content}
"""

    data = _call_llm_json(api_url, api_key, model, prompt, "Meta")
    novel_info, summary, errors = _validate_meta(data)
    _raise_if_errors(errors, "Meta")
    return {"analysis": {"novel_info": novel_info, "summary": summary}}


@app.post("/api/analyze/core")
def analyze_core(req: AnalyzeContentRequest):
    """分析角色 + 关系 + 淫荡指数"""
    api_url, api_key, model = _get_llm_config()

    prompt = f"""
As a professional literary analyst specializing in adult fiction, extract ONLY characters who engage in sexual activities and their sexual relationships.

## Requirements
- Characters: ONLY those with sexual activity.
- Gender MUST be "male" or "female" (lowercase English). No other values.
- Include: name, gender, identity, personality, sexual_preferences.
- For female characters: lewdness_score (1-100 int) and lewdness_analysis (required).
- First-person narrator ("我") must be included if involved; infer name/alias if possible, else use "我".
- Map every sexual relationship with from/to/type/start_way/description.

## Output JSON ONLY
{{
  "characters": [],
  "relationships": []
}}

## Novel Content
{req.content}
"""

    data = _call_llm_json(api_url, api_key, model, prompt, "Core")
    characters, errors = _validate_characters(data.get("characters"))
    if not characters:
        errors.append("characters 为空")
    names = {c["name"] for c in characters}
    relationships, rel_errors = _validate_relationships(
        data.get("relationships"), names
    )
    errors.extend(rel_errors)
    _raise_if_errors(errors, "Core")
    return {"analysis": {"characters": characters, "relationships": relationships}}


@app.post("/api/analyze/scenes")
def analyze_scenes(req: AnalyzeScenesRequest):
    """分析首次场景 + 统计 + 关系发展"""
    api_url, api_key, model = _get_llm_config()

    characters, char_errors = _validate_characters(req.characters)
    _raise_if_errors(char_errors, "Scenes 输入角色")
    names = {c["name"] for c in characters}
    relationships, rel_errors = _validate_relationships(req.relationships, names)
    _raise_if_errors(rel_errors, "Scenes 输入关系")

    allowed_names_json = json.dumps([c["name"] for c in characters], ensure_ascii=False)
    relationships_json = json.dumps(relationships, ensure_ascii=False)

    prompt = f"""
You are analyzing intimacy scenes and relationship evolution. Use ONLY the provided character names.

### Allowed character names (MUST use exactly):
{allowed_names_json}

### Known relationships (reference only):
{relationships_json}

## Output JSON ONLY
{{
  "first_sex_scenes": [],
  "sex_scenes": {{
    "total_count": 0,
    "scenes": []
  }},
  "evolution": []
}}

## Rules
- participants must be from allowed names; do NOT invent new names.
- chapter/location/description MUST be non-empty strings. If unknown, write "未知" or "未提及".
- evolution.chapter/stage/description MUST be non-empty strings. If unknown, write "未知" or "未提及".
- sex_scenes.total_count must be an integer.

## Novel Content
{req.content}
"""

    data = _call_llm_json(api_url, api_key, model, prompt, "Scenes")
    errors: list[str] = []
    first_scenes, first_errors = _validate_scenes(
        data.get("first_sex_scenes"), "first_sex_scenes", names
    )
    errors.extend(first_errors)

    sex_data = data.get("sex_scenes")
    if not isinstance(sex_data, dict):
        errors.append("sex_scenes 不是对象")
        sex_data = {}
    sex_scenes, sex_errors = _validate_scenes(
        sex_data.get("scenes"), "sex_scenes.scenes", names
    )
    errors.extend(sex_errors)
    total_count_raw = sex_data.get("total_count", len(sex_scenes))
    try:
        total_count = int(total_count_raw)
    except Exception:
        errors.append("sex_scenes.total_count 非整数")
        total_count = len(sex_scenes)
    if total_count < len(sex_scenes):
        total_count = len(sex_scenes)

    evolution, evo_errors = _validate_evolution(data.get("evolution"))
    errors.extend(evo_errors)
    _raise_if_errors(errors, "Scenes")
    return {
        "analysis": {
            "first_sex_scenes": first_scenes,
            "sex_scenes": {"total_count": total_count, "scenes": sex_scenes},
            "evolution": evolution,
        }
    }


@app.post("/api/analyze/thunderzones")
def analyze_thunderzones(req: AnalyzeThunderzonesRequest):
    """分析雷点"""
    api_url, api_key, model = _get_llm_config()

    characters, char_errors = _validate_characters(req.characters)
    _raise_if_errors(char_errors, "Thunder 输入角色")
    names = {c["name"] for c in characters}
    relationships, rel_errors = _validate_relationships(req.relationships, names)
    _raise_if_errors(rel_errors, "Thunder 输入关系")

    allowed_names_json = json.dumps([c["name"] for c in characters], ensure_ascii=False)
    relationships_json = json.dumps(relationships, ensure_ascii=False)

    prompt = f"""
You are detecting thunderzones (reader deal-breakers) in an adult novel. Use ONLY the provided character names.

### Allowed character names (MUST use exactly):
{allowed_names_json}

### Known relationships (reference only):
{relationships_json}

## Thunderzone types
- 绿帽/Cuckold
- NTR (Netorare)
- 女性舔狗
- 恶堕
- 其他

Severity must be: 高 / 中 / 低.

## Output JSON ONLY
{{
  "thunderzones": [],
  "thunderzone_summary": ""
}}

## Rules
- involved_characters must be from allowed names; do NOT invent new names.
- description MUST be non-empty. If unknown, write "未知" or "未提及".
- chapter_location is OPTIONAL. If unknown, you can leave it empty.
- relationship_context can be empty, but if provided must be a string.
- If no thunderzones, return empty array and a short summary.
- Use EXACT keys: type, severity, description, involved_characters, chapter_location, relationship_context.
- If you cannot provide a non-empty description for an item, DO NOT include that item.

## Novel Content
{req.content}
"""

    data = _call_llm_json(api_url, api_key, model, prompt, "Thunder")
    thunderzones, errors = _validate_thunderzones(data.get("thunderzones"), names)
    summary = _stringify(data.get("thunderzone_summary"))
    _raise_if_errors(errors, "Thunder")
    return {"analysis": {"thunderzones": thunderzones, "thunderzone_summary": summary}}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "6103"))
    log_level = os.getenv("LOG_LEVEL", "warning")
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"\n  ➜  Local:   http://{display_host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
