# -*- coding: utf-8 -*-
"""
小说分析器后端
基于FastAPI的轻量级Web服务
"""

import os
import json
import re
import copy
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


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    original_analysis: Dict[str, Any]


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
                },
                timeout=timeout,
            )

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

            if response.status_code != 200:
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

            result = response.json()
            message = (result.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content", "") or message.get("reasoning_content", "") or ""

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


def merge_analysis_results(original: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(original or {})

    def norm_text(value: Any) -> str:
        return str(value or "").strip()

    def norm_participants(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return tuple()
        items = [norm_text(x) for x in value if norm_text(x)]
        return tuple(sorted(set(items)))

    def better_value(old: Any, new: Any) -> Any:
        if old is None or old == "" or old == [] or old == {}:
            return new
        if new is None or new == "" or new == [] or new == {}:
            return old
        if isinstance(old, str) and isinstance(new, str):
            return new if len(new.strip()) > len(old.strip()) else old
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            return new if new > old else old
        if isinstance(old, list) and isinstance(new, list):
            return new if len(new) > len(old) else old
        if isinstance(old, dict) and isinstance(new, dict):
            return new if len(new.keys()) > len(old.keys()) else old
        return old

    def merge_dict_inplace(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        for k, v in incoming.items():
            target[k] = better_value(target.get(k), v)

    if not isinstance(merged, dict):
        return {}

    merged.setdefault("characters", [])
    if not isinstance(merged["characters"], list):
        merged["characters"] = []

    existing_chars: Dict[str, Dict[str, Any]] = {}
    for c in merged["characters"]:
        if isinstance(c, dict):
            name = norm_text(c.get("name"))
            if name:
                existing_chars[name] = c

    for c in (verification or {}).get("missing_characters") or []:
        if not isinstance(c, dict):
            continue
        name = norm_text(c.get("name"))
        if not name:
            continue
        if name not in existing_chars:
            merged["characters"].append(c)
            existing_chars[name] = c
        else:
            merge_dict_inplace(existing_chars[name], c)

    merged.setdefault("relationships", [])
    if not isinstance(merged["relationships"], list):
        merged["relationships"] = []

    existing_rels: Dict[tuple[str, str], Dict[str, Any]] = {}
    for r in merged["relationships"]:
        if isinstance(r, dict):
            key = (norm_text(r.get("from")), norm_text(r.get("to")))
            if key[0] and key[1]:
                existing_rels[key] = r

    for r in (verification or {}).get("missing_relationships") or []:
        if not isinstance(r, dict):
            continue
        key = (norm_text(r.get("from")), norm_text(r.get("to")))
        if not key[0] or not key[1]:
            continue
        if key not in existing_rels:
            merged["relationships"].append(r)
            existing_rels[key] = r
        else:
            merge_dict_inplace(existing_rels[key], r)

    merged.setdefault("first_sex_scenes", [])
    if not isinstance(merged["first_sex_scenes"], list):
        merged["first_sex_scenes"] = []

    existing_first: Dict[tuple[str, ...], Dict[str, Any]] = {}
    for s in merged["first_sex_scenes"]:
        if isinstance(s, dict):
            key = norm_participants(s.get("participants"))
            if key:
                existing_first[key] = s

    for s in (verification or {}).get("missing_first_sex_scenes") or []:
        if not isinstance(s, dict):
            continue
        key = norm_participants(s.get("participants"))
        if not key:
            continue
        if key not in existing_first:
            merged["first_sex_scenes"].append(s)
            existing_first[key] = s
        else:
            merge_dict_inplace(existing_first[key], s)

    sex_scenes = merged.get("sex_scenes") or {}
    if not isinstance(sex_scenes, dict):
        sex_scenes = {}
    sex_scenes.setdefault("scenes", [])
    if not isinstance(sex_scenes["scenes"], list):
        sex_scenes["scenes"] = []

    def norm_scene_key(scene: Dict[str, Any]) -> tuple[str, tuple[str, ...]]:
        return (norm_text(scene.get("chapter")), norm_participants(scene.get("participants")))

    existing_scenes: Dict[tuple[str, tuple[str, ...]], Dict[str, Any]] = {}
    for s in sex_scenes["scenes"]:
        if isinstance(s, dict):
            key = norm_scene_key(s)
            if key[0] and key[1]:
                existing_scenes[key] = s

    for s in (verification or {}).get("missing_sex_scenes") or []:
        if not isinstance(s, dict):
            continue
        key = norm_scene_key(s)
        if not key[0] or not key[1]:
            continue
        if key not in existing_scenes:
            sex_scenes["scenes"].append(s)
            existing_scenes[key] = s
        else:
            merge_dict_inplace(existing_scenes[key], s)

    original_total = sex_scenes.get("total_count", 0)
    try:
        original_total_int = int(original_total)
    except Exception:
        original_total_int = 0
    sex_scenes["total_count"] = max(original_total_int, len(sex_scenes["scenes"]))
    merged["sex_scenes"] = sex_scenes

    merged.setdefault("evolution", [])
    if not isinstance(merged["evolution"], list):
        merged["evolution"] = []

    existing_evo: Dict[str, Dict[str, Any]] = {}
    for e in merged["evolution"]:
        if isinstance(e, dict):
            key = norm_text(e.get("chapter"))
            if key:
                existing_evo[key] = e

    for e in (verification or {}).get("missing_evolution") or []:
        if not isinstance(e, dict):
            continue
        key = norm_text(e.get("chapter"))
        if not key:
            continue
        if key not in existing_evo:
            merged["evolution"].append(e)
            existing_evo[key] = e
        else:
            merge_dict_inplace(existing_evo[key], e)

    merged.setdefault("thunderzones", [])
    if not isinstance(merged["thunderzones"], list):
        merged["thunderzones"] = []

    def norm_th_key(t: Dict[str, Any]) -> tuple[str, str]:
        return (norm_text(t.get("type")), norm_text(t.get("chapter_location")))

    existing_th: Dict[tuple[str, str], Dict[str, Any]] = {}
    for t in merged["thunderzones"]:
        if isinstance(t, dict):
            key = norm_th_key(t)
            if key[0] and key[1]:
                existing_th[key] = t

    for t in (verification or {}).get("missing_thunderzones") or []:
        if not isinstance(t, dict):
            continue
        key = norm_th_key(t)
        if not key[0] or not key[1]:
            continue
        if key not in existing_th:
            merged["thunderzones"].append(t)
            existing_th[key] = t
        else:
            merge_dict_inplace(existing_th[key], t)

    return merged


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _ensure_analysis_defaults(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee required keys exist with sensible empty defaults to keep UI stable."""
    if not isinstance(analysis, dict):
        analysis = {}

    analysis.setdefault("novel_info", {})
    if not isinstance(analysis["novel_info"], dict):
        analysis["novel_info"] = {}

    analysis.setdefault("characters", [])
    if not isinstance(analysis["characters"], list):
        analysis["characters"] = []

    analysis.setdefault("relationships", [])
    if not isinstance(analysis["relationships"], list):
        analysis["relationships"] = []

    analysis.setdefault("first_sex_scenes", [])
    if not isinstance(analysis["first_sex_scenes"], list):
        analysis["first_sex_scenes"] = []

    sex_scenes = analysis.get("sex_scenes")
    if not isinstance(sex_scenes, dict):
        sex_scenes = {}
    sex_scenes.setdefault("total_count", 0)
    sex_scenes.setdefault("scenes", [])
    if not isinstance(sex_scenes["scenes"], list):
        sex_scenes["scenes"] = []
    analysis["sex_scenes"] = sex_scenes

    analysis.setdefault("evolution", [])
    if not isinstance(analysis["evolution"], list):
        analysis["evolution"] = []

    analysis.setdefault("thunderzones", [])
    if not isinstance(analysis["thunderzones"], list):
        analysis["thunderzones"] = []

    analysis.setdefault("thunderzone_summary", "")
    analysis.setdefault("summary", analysis.get("summary") or "")

    return analysis

def _validate_and_fix_analysis(analysis: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """
    Light schema validation/repair to keep frontend parsable.
    Returns (cleaned_analysis, errors).
    """
    errors: list[str] = []
    data = _ensure_analysis_defaults(copy.deepcopy(analysis or {}))

    # characters
    chars_in = data.get("characters")
    fixed_chars = []
    if isinstance(chars_in, list):
        for idx, c in enumerate(chars_in):
            if not isinstance(c, dict):
                errors.append(f"characters[{idx}] not object -> dropped")
                continue
            name = str(c.get("name") or "").strip()
            if not name:
                errors.append(f"characters[{idx}] missing name -> dropped")
                continue
            gender = str(c.get("gender") or "").strip() or "unknown"
            c["gender"] = gender
            fixed_chars.append(c)
    else:
        errors.append("characters not list -> reset")
    data["characters"] = fixed_chars

    # relationships
    rels_in = data.get("relationships")
    fixed_rels = []
    if isinstance(rels_in, list):
        for idx, r in enumerate(rels_in):
            if not isinstance(r, dict):
                errors.append(f"relationships[{idx}] not object -> dropped")
                continue
            frm = str(r.get("from") or "").strip()
            to = str(r.get("to") or "").strip()
            if not frm or not to:
                errors.append(f"relationships[{idx}] missing from/to -> dropped")
                continue
            fixed_rels.append(r)
    else:
        errors.append("relationships not list -> reset")
    data["relationships"] = fixed_rels

    # first_sex_scenes
    fss_in = data.get("first_sex_scenes")
    fixed_fss = []
    if isinstance(fss_in, list):
        for idx, s in enumerate(fss_in):
            if not isinstance(s, dict):
                errors.append(f"first_sex_scenes[{idx}] not object -> dropped")
                continue
            participants = s.get("participants")
            if not isinstance(participants, list) or not participants:
                errors.append(f"first_sex_scenes[{idx}] missing participants -> dropped")
                continue
            s["participants"] = [str(p).strip() for p in participants if str(p).strip()]
            fixed_fss.append(s)
    else:
        errors.append("first_sex_scenes not list -> reset")
    data["first_sex_scenes"] = fixed_fss

    # sex_scenes
    sex = data.get("sex_scenes") or {}
    if not isinstance(sex, dict):
        sex = {}
        errors.append("sex_scenes not object -> reset")
    scenes_in = sex.get("scenes")
    fixed_scenes = []
    if isinstance(scenes_in, list):
        for idx, s in enumerate(scenes_in):
            if not isinstance(s, dict):
                errors.append(f"sex_scenes.scenes[{idx}] not object -> dropped")
                continue
            participants = s.get("participants")
            if not isinstance(participants, list) or not participants:
                errors.append(f"sex_scenes.scenes[{idx}] missing participants -> dropped")
                continue
            s["participants"] = [str(p).strip() for p in participants if str(p).strip()]
            fixed_scenes.append(s)
    else:
        errors.append("sex_scenes.scenes not list -> reset")
    sex["scenes"] = fixed_scenes
    try:
        sex["total_count"] = max(int(sex.get("total_count") or 0), len(fixed_scenes))
    except Exception:
        sex["total_count"] = len(fixed_scenes)
        errors.append("sex_scenes.total_count invalid -> recalculated")
    data["sex_scenes"] = sex

    # evolution
    evo_in = data.get("evolution")
    if not isinstance(evo_in, list):
        data["evolution"] = []
        errors.append("evolution not list -> reset")

    # thunderzones
    th_in = data.get("thunderzones")
    fixed_th = []
    if isinstance(th_in, list):
        for idx, t in enumerate(th_in):
            if not isinstance(t, dict):
                errors.append(f"thunderzones[{idx}] not object -> dropped")
                continue
            if not t.get("type") and not t.get("description"):
                errors.append(f"thunderzones[{idx}] missing type/description -> dropped")
                continue
            fixed_th.append(t)
    else:
        errors.append("thunderzones not list -> reset")
    data["thunderzones"] = fixed_th

    # summary / thunderzone_summary
    for key in ("summary", "thunderzone_summary"):
        val = data.get(key)
        if not isinstance(val, str):
            data[key] = "" if val is None else str(val)
            errors.append(f"{key} not string -> coerced")

    # novel_info
    if not isinstance(data.get("novel_info"), dict):
        data["novel_info"] = {}
        errors.append("novel_info not object -> reset")

    return data, errors


def _reconcile_entities(analysis: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], dict]:
    """
    Cross-check participants vs characters/relationships; auto-add missing pieces.
    Returns (analysis, errors, stats)
    stats keys: added_characters, added_relationships
    """
    errors: list[str] = []
    stats = {"added_characters": 0, "added_relationships": 0}
    data = copy.deepcopy(analysis or {})

    characters = data.get("characters") or []
    relationships = data.get("relationships") or []
    first_scenes = data.get("first_sex_scenes") or []
    sex = data.get("sex_scenes") or {}
    sex_scenes = sex.get("scenes") or []

    name_set = {c.get("name") for c in characters if isinstance(c, dict) and c.get("name")}

    def _collect_participants():
        parts = set()
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            frm = str(rel.get("from") or "").strip()
            to = str(rel.get("to") or "").strip()
            if frm:
                parts.add(frm)
            if to:
                parts.add(to)
        for scene in list(first_scenes) + list(sex_scenes):
            if not isinstance(scene, dict):
                continue
            for p in scene.get("participants") or []:
                p_name = str(p or "").strip()
                if p_name:
                    parts.add(p_name)
        return parts

    participants = _collect_participants()

    # Add missing characters referenced elsewhere
    for p in sorted(participants):
        if p not in name_set:
            characters.append({
                "name": p,
                "gender": "unknown",
                "identity": "",
                "personality": "",
                "sexual_preferences": ""
            })
            name_set.add(p)
            stats["added_characters"] += 1
            errors.append(f"added missing character: {p}")

    # Build relationship keys to avoid duplicates (undirected)
    rel_keys = set()
    cleaned_rels = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        frm = str(rel.get("from") or "").strip()
        to = str(rel.get("to") or "").strip()
        if not frm or not to:
            continue
        key = tuple(sorted([frm, to]))
        if key in rel_keys:
            continue
        rel_keys.add(key)
        cleaned_rels.append(rel)
    relationships = cleaned_rels

    # Auto-add relationships from sex scenes (pairwise)
    for scene in sex_scenes:
        if not isinstance(scene, dict):
            continue
        participants_list = [str(p).strip() for p in scene.get("participants") or [] if str(p).strip()]
        if len(participants_list) < 2:
            continue
        for i in range(len(participants_list)):
            for j in range(i + 1, len(participants_list)):
                a, b = participants_list[i], participants_list[j]
                key = tuple(sorted([a, b]))
                if key in rel_keys:
                    continue
                rel_keys.add(key)
                relationships.append({
                    "from": a,
                    "to": b,
                    "type": "性关系",
                    "start_way": "性场景自动补全",
                    "description": "auto-added from sex scene"
                })
                stats["added_relationships"] += 1
                errors.append(f"added missing relationship: {a} - {b}")

    data["characters"] = characters
    data["relationships"] = relationships
    return data, errors, stats



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
        dirs[:] = [d for d in dirs if d not in exclude_keywords and not d.startswith('.')]

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

        if len(content) > 80000:
            content = content[:80000] + "\n\n... (内容已截断，分析可能不完整)"

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


@app.post("/api/analyze")
def analyze_novel(req: AnalyzeRequest):
    """调用LLM分析小说 - 支持多角色多关系"""
    api_url, api_key, model = _get_llm_config()

    prompt_v1 = f"""
As a professional literary analyst specializing in adult fiction, analyze this novel comprehensively.

## Analysis Requirements

### 0. NOVEL METADATA
Extract basic novel information:
- World setting/background (genre, time period, universe type)
- Estimated chapter count (count chapter markers like "第X章", "Chapter X", etc.)
- Completion status (based on ending - does story conclude or feel unfinished?)

### 1. SEXUAL CHARACTERS ONLY
Identify ONLY characters who engage in sexual activities (只包含有性行为的角色):
- Name/alias
- Gender role: male/female
- Identity: occupation, age, social status
- Personality traits
- SEXUAL PREFERENCES & KINKS: Describe what this character enjoys in bed:
  - Position preferences
  - Role in sex (dominant/submissive/equal)
  - Specific kinks: anal, oral, vaginal, BDSM, foot fetish, cum, creampie, gangbang, etc.
  - Any fetishes mentioned: foot worship, body worship, anal play, etc.
  - Personality in bed: aggressive, shy, experienced, virgin, etc.
- FOR FEMALE CHARACTERS ONLY - 淫荡指数 (Lewdness Index) - REQUIRED FOR ALL FEMALES:
  - Score 1-100 based on: sexual frequency, initiative, variety of partners, openness to kinks
  - EVERY female character MUST have lewdness_score and lewdness_analysis
  - Provide brief analysis explaining the score
- **CRITICAL - FIRST-PERSON NARRATOR**: Many Chinese adult novels use first-person narration ("我"). If the narrator participates in ANY sexual activity, they MUST be listed as a character.
  - Determine their name/alias from how others address them (e.g., "哥哥", "老公", "男友", name, or simply "主角(哥哥)" / "主角(男主)" if no specific name).
  - Infer gender from context (pronouns, how addressed, role in sex scenes). Default to male if addressed as 哥哥/兄长/老公.
  - DO NOT omit the narrator just because they use "我". The narrator is a real character.

### 2. ALL SEXUAL RELATIONSHIPS
Map every sexual relationship in the novel:
- Who with whom
- Relationship type: one-night stand/regular/FWB/lover/spouse/etc.
- How it started

### 3. FIRST SEX SCENES FOR EACH PAIR
For each sexual pair, find their first intimate scene:
- Characters involved
- Chapter location
- Scene description (under 50 chars)

### 4. COMPLETE INTIMACY STATISTICS
For ALL sexual activities in the novel:
- Total scene count
- For each scene: chapter, participants, location, description

### 5. SEXUAL EVOLUTION
Track how sexual relationships develop:
- Key milestones for each pair

### 6. DETAILED SUMMARY
Write a comprehensive summary (200-300 characters) covering:
- Main plot and story arc
- Core sexual themes and dynamics
- Key character relationships

### 7. THUNDERZONE DETECTION (雷点检测)
Identify all potential "thunderzones" (deal-breakers) that might upset readers.
For each thunderzone found, provide: type, severity, characters involved, chapter location, and description.

雷点类型定义:
- 绿帽/Cuckold: Male character's partner has sex with others (willingly, coerced, or unknowingly)
- NTR (Netorare): Protagonist's romantic partner/lover/spouse is taken by another character
- 女性舔狗/Female Doormat: Female character who is excessively submissive, desperately pursues a man with no dignity
- 恶堕/Fall from Grace: A character who was pure/virtuous/innocent becomes corrupted, sexually liberated, or morally degraded
- 其他雷点/Other: Any other content that might be a deal-breaker (例如: 乱伦、SM程度过重、角色死亡等)

Severity 判定标准:
- 高/High: 核心剧情涉及，影响主角或主要配角，涉及详细描写
- 中/Medium: 支线剧情涉及，影响次要角色，有一定描写
- 低/Low: 背景提及、回忆片段、一笔带过

## Output Format (JSON ONLY)

### STRICT JSON OUTPUT RULES (MUST FOLLOW)
- Return ONLY one JSON object. No extra text, no explanations.
- Final output MUST start with `{{` and end with `}}`.
- Must be valid JSON: double quotes for keys/strings, no trailing commas, no comments, no NaN/Infinity.
- Do NOT output `null` for required arrays/objects. Use `[]`, `{{}}`, or `""`.
- Required top-level keys (MUST exist even if empty):
  - `novel_info`, `characters`, `relationships`, `first_sex_scenes`, `sex_scenes`, `evolution`, `summary`, `thunderzones`, `thunderzone_summary`
- If you are unsure about any field, keep the key and use empty values; do not omit keys.

### Schema skeleton (reference only; your FINAL answer must NOT include code fences)
{{
  "novel_info": {{
    "world_setting": "",
    "chapter_count": 0,
    "is_completed": false,
    "completion_note": ""
  }},
  "characters": [],
  "relationships": [],
  "first_sex_scenes": [],
  "sex_scenes": {{
    "total_count": 0,
    "scenes": []
  }},
  "evolution": [],
  "summary": "",
  "thunderzones": [],
  "thunderzone_summary": ""
}}

### Field requirements (when items exist)
- `characters[i]` MUST include: `name`, `gender`, `identity`, `personality`, `sexual_preferences`.
  - For FEMALE characters also include `lewdness_score` (1-100 integer) and `lewdness_analysis`.
- `relationships[i]` MUST include: `from`, `to`, `type`, `start_way`, `description` (all strings).
- `first_sex_scenes[i]` MUST include: `participants` (non-empty string array), `chapter`, `location`, `description`.
- `sex_scenes.scenes[i]` MUST include: `chapter`, `participants` (non-empty string array), `location`, `description`.
- `evolution[i]` MUST include: `chapter`, `stage`, `description`.
- `thunderzones[i]` MUST include: `type`, `severity` (高/中/低), `description`, `involved_characters` (string array), `chapter_location`, `relationship_context`.

### If you cannot produce valid JSON
Return the schema skeleton EXACTLY (with empty arrays/strings) and nothing else.

## Notes
- ONLY include characters who have sexual activities (not all characters)
- Be thorough about sexual preferences
- Output MUST be valid JSON only

## Novel Content

{req.content}
"""

    prompts = [
        ("multi-character", prompt_v1)
    ]

    analysis = None
    last_error = None
    raw_response = None

    for style_name, prompt in prompts:
        try:
            content, _raw = call_llm_with_response(api_url, api_key, model, prompt, temperature=0.7)
            analysis = extract_json_from_response(content)

            if analysis and ('characters' in analysis or 'sex_scenes' in analysis):
                break
            else:
                last_error = f"数据格式异常(内容长:{len(content)})"
                raw_response = content[:1000]
        except LLMCallError as e:
            last_error = str(e)
            if e.raw_response:
                raw_response = e.raw_response
            continue
        except Exception as e:
            last_error = str(e)
            continue

    if not analysis:
        error_msg = f"分析失败: {last_error}"
        if raw_response and DEBUG:
            error_msg += f"\n\n原始响应:\n{raw_response[:2000]}"
        elif "返回内容过短" in last_error:
            error_msg += "\n\n原始响应内容太短，API可能拦截了请求"

        raise HTTPException(status_code=422, detail=error_msg)
    analysis = _ensure_analysis_defaults(analysis)

    # Local schema validation & reconciliation only (single LLM call)
    analysis, _ = _validate_and_fix_analysis(analysis)
    analysis, _, _ = _reconcile_entities(analysis)
    analysis, _ = _validate_and_fix_analysis(analysis)

    if not analysis.get("characters"):
        raise HTTPException(status_code=422, detail="分析失败: 无有效角色（模型输出无法修复）")
    return {"analysis": analysis}


def _perform_verification(api_url: str, api_key: str, model: str, *, content: str, original: Dict[str, Any], raise_on_error: bool = True):
    """Run LLM-based completeness/consistency review. Can soft-fail without raising."""
    content_clean = (content or "").strip()
    if not content_clean:
        msg = "content不能为空"
        if raise_on_error:
            raise HTTPException(status_code=400, detail=msg)
        return original, {}, msg

    if not isinstance(original, dict) or not original:
        msg = "original_analysis不能为空"
        if raise_on_error:
            raise HTTPException(status_code=400, detail=msg)
        return original, {}, msg

    existing_characters = []
    for c in original.get("characters") or []:
        if isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            if name:
                existing_characters.append(name)

    existing_relationships = []
    for r in original.get("relationships") or []:
        if isinstance(r, dict):
            frm = str(r.get("from") or "").strip()
            to = str(r.get("to") or "").strip()
            if frm and to:
                existing_relationships.append({"from": frm, "to": to})

    verify_prompt = f"""
You are reviewing an existing novel analysis for completeness.

CRITICAL: The original analysis may have missed characters, especially:
1. First-person narrators ("我", "主角") - these ARE characters and MUST be included
2. Male characters who participate in sexual activities
3. Characters only mentioned by role (e.g., "丈夫", "男友", "老公")
4. Missing relationship pairs and missing scenes

Original analysis found these characters (names only): {json.dumps(existing_characters, ensure_ascii=False)}
Original analysis found these relationships (pairs): {json.dumps(existing_relationships, ensure_ascii=False)}

Your task:
1. Identify ANY sexual participants that are MISSING from the original character list
2. Identify ANY sexual relationships that are MISSING from the original relationships list
3. For first-person narratives, the narrator IS a character - identify their name/alias and gender
4. Output ONLY the missing items (do NOT repeat existing items)

Output JSON ONLY (missing items only). If nothing is missing, return empty arrays and corrections as "".

```json
{{
  "missing_characters": [],
  "missing_relationships": [],
  "missing_first_sex_scenes": [],
  "missing_sex_scenes": [],
  "missing_evolution": [],
  "missing_thunderzones": [],
  "corrections": ""
}}
```

Novel Content:

{content_clean}
"""

    verification = {}
    last_error = None
    raw_response = None

    try:
        verify_text, _raw = call_llm_with_response(api_url, api_key, model, verify_prompt, temperature=0.3)
        verification = extract_json_from_response(verify_text) or {}
        if not isinstance(verification, dict):
            verification = {}
    except LLMCallError as e:
        last_error = str(e)
        raw_response = e.raw_response
    except Exception as e:
        last_error = str(e)

    if not verification and last_error:
        error_msg = f"校验失败: {last_error}"
        if raw_response and DEBUG:
            error_msg += f"\n\n原始响应:\n{raw_response[:2000]}"
        if raise_on_error:
            raise HTTPException(status_code=422, detail=error_msg)
        return original, {}, error_msg

    merged = merge_analysis_results(original, verification)
    return merged, verification, None


@app.post("/api/verify")
def verify_analysis(req: VerifyRequest):
    api_url, api_key, model = _get_llm_config()
    merged, verification, error_msg = _perform_verification(
        api_url,
        api_key,
        model,
        content=req.content,
        original=req.original_analysis,
        raise_on_error=True,
    )
    merged, val_errors = _validate_and_fix_analysis(merged)
    merged, recon_errors, recon_stats = _reconcile_entities(merged)
    merged, val_errors2 = _validate_and_fix_analysis(merged)

    if not merged.get("characters"):
        raise HTTPException(status_code=422, detail="校验失败: 无有效角色（模型输出无法修复）")

    validation_errors = val_errors + recon_errors + val_errors2

    return {
        "missing_characters": _safe_list(verification.get("missing_characters")),
        "missing_relationships": _safe_list(verification.get("missing_relationships")),
        "missing_first_sex_scenes": _safe_list(verification.get("missing_first_sex_scenes")),
        "missing_sex_scenes": _safe_list(verification.get("missing_sex_scenes")),
        "missing_evolution": _safe_list(verification.get("missing_evolution")),
        "missing_thunderzones": _safe_list(verification.get("missing_thunderzones")),
        "corrections": str(verification.get("corrections") or "").strip(),
        "merged_analysis": merged,
        "error": error_msg,
        "validation_errors": validation_errors,
        "added_characters": recon_stats.get("added_characters", 0),
        "added_relationships": recon_stats.get("added_relationships", 0),
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "6103"))
    log_level = os.getenv("LOG_LEVEL", "warning")
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"\n  ➜  Local:   http://{display_host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
