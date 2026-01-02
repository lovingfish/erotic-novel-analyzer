When the user trigger openspec flow, u must strictly follow the instructions of openspec below.


<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

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

# Run tests
python -m pytest -q
```

Server runs at `http://127.0.0.1:6103` by default.

## Architecture

Single-page app with FastAPI backend + vanilla JS frontend.

```text
backend.py                 # FastAPI routes + error mapping
src/novel_analyzer/        # LLM client + schemas + config loading + truncation + logging
config/llm.yaml            # Fixed LLM strategy config (read-only path)
config/prompts/*.j2        # Prompt templates (Jinja2)
templates/index.html       # Alpine.js app, all UI state/logic
static/                    # CSS + client-side rendering (chart-view.js)
```

### API Endpoints

- `/api/config` (GET) Server config (API URL, model name)
- `/api/novels` (GET) Scan novel directory (recursive `.txt` scan)
- `/api/novel/{path}` (GET) Read novel content
- `/api/test-connection` (GET) Test LLM API connection + Function Calling support
- `/api/analyze/meta` (POST) Novel metadata + summary
- `/api/analyze/core` (POST) Characters + relationships + lewdness
- `/api/analyze/scenes` (POST) First scenes + stats + evolution
- `/api/analyze/thunderzones` (POST) Thunderzone detection

### Data Flow

1. `scanNovels()` → `GET /api/novels` → renders dropdown
2. `selectNovel()` → `GET /api/novel/{path}` → loads content
3. `analyzeNovel()` → `POST /api/analyze/meta` + `POST /api/analyze/core` (parallel)
4. `analyzeNovel()` → `POST /api/analyze/scenes` + `POST /api/analyze/thunderzones` (parallel, with角色/关系名单)
5. Merge results → `renderAllData()`

## Backend Design (v3)

### Configuration

- `.env` (not committed): secrets / environment-specific settings only
  - `API_BASE_URL`, `API_KEY`, `MODEL_NAME`
  - `NOVEL_PATH`
  - `HOST`, `PORT`, `LOG_LEVEL`, `DEBUG`
- `config/llm.yaml` (committed): **LLM strategy only**
  - per-section temperature
  - truncation/sampling settings
  - retry/backoff policy
  - repair policy
- `config/prompts/*.j2`: prompt templates

### LLM Output Pipeline (Strict)

- Prompt is rendered from Jinja2 template.
- LLM is called via **Function Calling** (`tools` + `tool_choice`).
- Backend only consumes `tool_calls[].function.arguments` (or legacy `function_call.arguments` when protocol fallback is needed).
- Arguments are validated using Pydantic schemas in `src/novel_analyzer/schemas.py`.
- On schema validation failure, a single Repair Pass may run (still function-calling + same tool schema).

### Observability

Structured JSON logs emitted via logger `novel_analyzer.llm`:
- retry
- repair
- truncation
- protocol fallback

## Frontend Notes

- Alpine.js `x-data="app()"` manages all state.
- Rendering via DOM APIs; CSS via DaisyUI/Tailwind.
- Export uses CDN-loaded DaisyUI/Tailwind/Alpine.js and inlines `static/style.css`.

## Testing

### Unit + E2E

```bash
python -m pytest -q
```

E2E requires Playwright:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m pytest -q
```
