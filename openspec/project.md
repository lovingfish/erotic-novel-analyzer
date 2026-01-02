# Project Context

## Purpose
LLM-powered novel analyzer for adult fiction. Extracts metadata, characters, relationships, intimacy scenes, and "thunderzones" (reader deal-breakers) from Chinese `.txt` novels via structured Function Calling.

## Tech Stack
- **Backend**: Python 3.14, FastAPI, Uvicorn, Pydantic v2
- **Frontend**: Vanilla JS, Alpine.js, DaisyUI/Tailwind CSS (CDN)
- **LLM Integration**: OpenAI-compatible API with Function Calling (`tools` + `tool_choice`)
- **Templating**: Jinja2 (prompts in `config/prompts/*.j2`, HTML in `templates/`)
- **Config**: `.env` for secrets, `config/llm.yaml` for LLM strategy (read-only)
- **Testing**: pytest, Playwright (E2E)

## Project Conventions

### Code Style
- 4-space indentation (Python)
- Type hints for public helpers and Pydantic models
- Internal helpers prefixed with `_` (e.g., `_safe_novel_path`)
- Pydantic models use `extra="forbid"` and `ConfigDict`
- Chinese comments/UI text acceptable

### Architecture Patterns
- Single-page app: FastAPI serves `templates/index.html`
- All LLM output via Function Calling → Pydantic validation → optional repair pass
- Structured JSON logging for observability (`novel_analyzer.llm`)
- Security middleware adds `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`

### Testing Strategy
- Unit tests: `python -m pytest -q`
- E2E tests: Playwright + Chromium (`tests/test_*_e2e.py`)
- Run all tests before PR; fix all failures

### Git Workflow
- Branch from `main`, PR back to `main`
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Never commit `.env` or API keys

## Domain Context
- **Sections**: `meta` (metadata/summary), `core` (characters/relationships/lewdness), `scenes` (intimacy scenes/evolution), `thunder` (deal-breakers)
- **Lewdness score**: 1-100, required for female characters only
- **Thunderzones**: Content warnings with severity (高/中/低)
- Prompts are Jinja2 templates receiving `content`, `tool_name`, and optional `allowed_names_json`/`relationships_json`

## Important Constraints
- LLM output **must** use Function Calling; plain JSON/text responses are rejected
- `config/llm.yaml` is read-only at runtime (no dynamic config changes)
- File access restricted to `NOVEL_PATH` directory; only `.txt` files allowed
- `HOST=127.0.0.1` by default (no LAN exposure)

## External Dependencies
- OpenAI-compatible LLM API (configured via `API_BASE_URL`, `API_KEY`, `MODEL_NAME` in `.env`)
- CDN: DaisyUI, Tailwind CSS, Alpine.js, Lucide icons
