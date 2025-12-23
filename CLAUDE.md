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

### Data Flow
1. `scanNovels()` → `GET /api/novels` → renders dropdown
2. `selectNovel()` → `GET /api/novel/{path}` → loads content
3. `analyzeNovel()` → `POST /api/analyze`
   - single LLM call
   - JSON extraction
   - local repair (schema validation + reconciliation)
   → `renderAllData()`

### Export Report
- UI入口：分析完成后点击“导出”
- 调用链：`templates/index.html` → `doExport()` → `static/chart-view.js` → `exportReport()`
- 导出HTML：通过CDN加载 DaisyUI/Tailwind/Alpine.js，内联 `static/style.css`（保证离线打开也能保持一致样式）
- Tab一致性：导出使用与网页版相同的 6 个Tab（总结/角色/关系图/首次/统计/发展）
- 关系图导出：固定SVG画布 `1200x800`，并锁定主题颜色（dark/light）
- 文件名：使用 `sanitizeFilename()` 清理后下载

### E2E Tests (Export)
```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m pytest -q
```

Sample export file: `tests/export/test_export_report.html` (generated via `python scripts/generate_export_sample.py`).

### Frontend Patterns
- Alpine.js `x-data="app()"` manages all state
- DOM rendering via `document.createElement()` (no virtual DOM)
- CSS uses oklch color space with DaisyUI theme variables (`--b1`, `--bc`, `--p`, etc.)

### Backend Patterns
- Config from `.env` via `python-dotenv`
- Path traversal protection in `_safe_novel_path()`
- LLM response JSON extraction handles markdown code blocks

## Configuration

All config in `.env` (copy from `.env.example`):
- `NOVEL_PATH` - novel directory to scan
- `API_BASE_URL`, `API_KEY`, `MODEL_NAME` - OpenAI-compatible API
- `HOST`, `PORT` - server binding
