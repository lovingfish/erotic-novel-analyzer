# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server (Windows)
start.bat

# Start server (manual)
python backend.py

# Install dependencies
pip install -r requirements.txt
```

Server runs at `http://127.0.0.1:6103` by default.

## Architecture

Single-page app with FastAPI backend + vanilla JS frontend.

```
backend.py          # FastAPI server, LLM API calls, file scanning
templates/
  index.html        # Alpine.js app, all UI state/logic
static/
  style.css         # Tailwind/DaisyUI supplements, shadcn-style components
  chart-view.js     # SVG relationship graph, data rendering functions
```

### API Endpoints

| Endpoint                    | Method | Description                                  |
| --------------------------- | ------ | -------------------------------------------- |
| `/api/config`               | GET    | Server config (API URL, model name)          |
| `/api/novels`               | GET    | Scan novel directory (recursive `.txt` scan) |
| `/api/novel/{path}`         | GET    | Read novel content                           |
| `/api/test-connection`      | GET    | Test LLM API connection                      |
| `/api/analyze/meta`         | POST   | Novel metadata + summary                     |
| `/api/analyze/core`         | POST   | Characters + relationships + lewdness        |
| `/api/analyze/scenes`       | POST   | First scenes + stats + evolution             |
| `/api/analyze/thunderzones` | POST   | Thunderzone detection                        |

### Data Flow

1. `scanNovels()` → `GET /api/novels` → renders dropdown
2. `selectNovel()` → `GET /api/novel/{path}` → loads content
3. `analyzeNovel()` → `POST /api/analyze/meta` + `POST /api/analyze/core` (parallel)
4. `analyzeNovel()` → `POST /api/analyze/scenes` + `POST /api/analyze/thunderzones` (parallel, with角色/关系名单)
5. Merge results → `renderAllData()`

### Key Features

#### Thunderzone Detection
- Detects: 绿帽 (Cuckold), NTR, 女性舔狗 (Female Doormat), 恶堕 (Fall from Grace), 其他 (Other)
- Severity levels: 高 (High), 中 (Medium), 低 (Low)
- Includes: type, severity, description, involved_characters, relationship_context (chapter_location optional)

#### Lewdness Index (Female Characters)
- Score 1-100 for each female character
- Based on: sexual frequency, initiative, variety of partners, openness to kinks
- Includes detailed analysis explaining the score
- Female characters sorted by lewdness_score in UI
- Visual badge color coding:
  - 90+: 红色 (extremely lewd)
  - 70-89: 橙色 (very lewd)
  - 50-69: 黄色 (moderate)
  - 30-49: 绿色 (low)
  - <30: 蓝色 (pure)

#### Cross-Entity Consistency (Fail Fast)
- Relationships/scenes/thunderzones must reference names present in `characters`
- Backend validates and returns `HTTP 422` on any mismatch
- No auto-reconciliation/dedup/auto-add: prefer failing rather than silently producing inconsistent graphs

#### First-Person Narrator Handling
- Prompt-level requirement: include first-person narrator ("我") as a character if involved; infer alias if possible, else use "我"
- Backend does not auto-infer narrator name/gender; it only normalizes/validates the returned `gender` values

#### Export Report
- UI入口：分析完成后点击"导出"
- 调用链：`templates/index.html` → `doExport()` → `static/chart-view.js` → `exportReport()`
- 导出HTML：通过CDN加载 DaisyUI/Tailwind/Alpine.js，内联 `static/style.css`（保证离线打开也能保持一致样式）
- Tab一致性：导出使用与网页版相同的 7 个Tab（总结/雷点/角色/关系图/首次/统计/发展）
- 关系图导出：固定SVG画布 `1200x800`，并锁定主题颜色（dark/light）
- 文件名：使用 `sanitizeFilename()` 清理后下载

### Frontend Patterns
- Alpine.js `x-data="app()"` manages all state
- DOM rendering via `document.createElement()` (no virtual DOM)
- CSS uses oklch color space with DaisyUI theme variables (`--b1`, `--bc`, `--p`, etc.)
- 9 tabs: pipeline(进度), summary, thunderzones, characters, relationships, firstsex, count, progress(发展), logs
- Log system with levels: info, success, error, warning
- Theme toggle (dark/light) persists to localStorage

### Backend Patterns
- Config from `.env` via `python-dotenv`
- Path traversal protection in `_safe_novel_path()`
- LLM calls `POST {API_BASE_URL}/chat/completions` (OpenAI-compatible) with `stream=true`
- Stream parsing: assembles text from `choices[0].delta.content` (or `delta.reasoning_content`)
- JSON extraction is best-effort (`extract_json_from_response`): raw JSON → ```json code block → brace substring; still fragile to JSON5/truncation
- `call_llm_with_response()` with retry logic for 502/503/504/429
- Security middleware: X-Content-Type-Options, Referrer-Policy, X-Frame-Options
- Strict schema validation per analysis stage (`_validate_characters`, `_validate_relationships`, etc.)
- No auto-repair: invalid JSON / schema mismatch fails fast (HTTP 422)
- `DEBUG=true` will include a snippet of raw LLM response for call/parse failures (useful for prompt tuning); schema validation errors only include the validation error list

### Data Models

#### Analysis Response
```json
{
  "novel_info": {
    "world_setting": "string",
    "world_tags": ["string"],
    "chapter_count": 0,
    "is_completed": false,
    "completion_note": "string"
  },
  "characters": [
    {
      "name": "string",
      "gender": "male|female",
      "identity": "string",
      "personality": "string",
      "sexual_preferences": "string",
      "lewdness_score": 0,  // female only
      "lewdness_analysis": "string"  // female only
    }
  ],
  "relationships": [
    {
      "from": "string",
      "to": "string",
      "type": "string",
      "start_way": "string",
      "description": "string"
    }
  ],
  "first_sex_scenes": [
    {
      "participants": ["string"],
      "chapter": "string",
      "location": "string",
      "description": "string"
    }
  ],
  "sex_scenes": {
    "total_count": 0,
    "scenes": [
      {
        "chapter": "string",
        "participants": ["string"],
        "location": "string",
        "description": "string"
      }
    ]
  },
  "evolution": [
    {
      "chapter": "string",
      "stage": "string",
      "description": "string"
    }
  ],
  "thunderzones": [
    {
      "type": "string",
      "severity": "高|中|低",
      "description": "string",
      "involved_characters": ["string"],
      "chapter_location": "string (optional)",
      "relationship_context": "string"
    }
  ],
  "summary": "string",
  "thunderzone_summary": "string"
}
```

## Configuration

All config in `.env` (copy from `.env.example`):
- `NOVEL_PATH` - novel directory to scan
- `API_BASE_URL`, `API_KEY`, `MODEL_NAME` - OpenAI-compatible API
- `HOST`, `PORT` - server binding
- `LOG_LEVEL` - Python logging level (debug/info/warning/error/critical)
- `DEBUG` - Show raw LLM responses in error messages (true/false)

## Testing

### Unit Tests
```bash
python -m pytest tests/ -q
```


### E2E Tests (Export)
```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m pytest tests/test_export_report_e2e.py -q
python -m pytest tests/test_thunderzones_e2e.py -q
```

Sample export file: `tests/export/test_export_report.html` (generated via `python scripts/generate_export_sample.py`).

### Test Coverage
- Export report generation
- Thunderzone detection
- UI interaction (E2E)
