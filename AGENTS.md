# Repository Guidelines

## Project Structure & Module Organization

```
.
├── backend.py              # FastAPI server: file scanning + LLM calls + /api/*
├── templates/index.html    # Single-page UI (Alpine.js) served by FastAPI
├── static/                 # CSS + client-side rendering (e.g., chart-view.js)
├── requirements.txt        # Python runtime dependencies
├── .env.example            # Config template (copy to .env; never commit secrets)
└── start.bat               # Windows launcher (installs deps, starts server)
```

Keep changes cohesive: backend logic in `backend.py`, UI state/behavior in `templates/index.html`, reusable JS rendering helpers in `static/`.

## Build, Test, and Development Commands

- Create venv (optional): `python -m venv venv` then `.\venv\Scripts\activate` (Windows).
- Install deps: `pip install -r requirements.txt`
- Run (Windows): `start.bat`
- Run (manual): `python backend.py`
- Run with reload (dev): `uvicorn backend:app --reload --host 127.0.0.1 --port 6103`

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep functions small; validate external inputs (paths, URLs, LLM output); prefer type hints for public helpers and request/response models.
- Frontend: stay “vanilla” (Alpine.js + DOM APIs). Avoid adding build tooling unless there’s a strong reason.
- Naming: endpoints live under `/api/*`; Python internal helpers use a leading `_` (e.g., `_safe_novel_path`).

## Testing Guidelines

There is no automated test suite yet. For non-trivial changes, add `pytest` and a `tests/` folder (files named `test_*.py`).

Minimum checks before opening a PR:
- `python -m compileall backend.py`
- Manual smoke test: start the server, select a small `.txt`, run analysis, verify key tabs render.

## Commit & Pull Request Guidelines

- Follow Conventional Commits observed in history: `feat:`, `fix:`, `refactor:`, `docs:` + short summary (Chinese or English).
- PRs should include: what changed, why, how to verify, and screenshots/GIFs for UI changes.
- Call out security-impacting edits explicitly (path handling, host binding, file access). Backward compatibility is not required—prefer clarity over preserving old behavior.

## Security & Configuration Tips

- Keep `HOST=127.0.0.1` unless you intentionally want LAN exposure.
- Never commit `.env` or API keys; keep novel files local and treat them as sensitive input.
