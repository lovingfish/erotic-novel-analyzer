# Change: Replace novel scanning with file upload

## Why
当前实现依赖服务端递归扫描 `NOVEL_PATH` 并通过相对路径读取 `.txt` 文件，再把全文返回给前端。该模式对用户不灵活（必须整理目录/路径），同时把 UI 交互与服务端文件系统强绑定，增加维护与安全负担（路径校验、目录扫描、测试环境构造等）。

目标是改为“用户点击选择文件 → 浏览器读取文本 → 复用现有 `/api/analyze/*` 分析接口”，彻底移除旧的目录扫描/按路径读文件链路。

## What Changes
- **BREAKING** 移除 `GET /api/novels`（递归扫描目录）
- **BREAKING** 移除 `GET /api/novel/{path}`（按相对路径读取本地文件）
- **BREAKING** 移除 `NOVEL_PATH` 配置及 `_safe_novel_path()`（服务端不再访问本地小说文件系统）
- 前端新增文件选择/上传入口：选择 `.txt` 后读取内容并写入 `currentNovelContent`
- 保持既有分析接口不变：`/api/analyze/*` 仍以 JSON `content` 作为输入
- E2E 测试从“构造 NOVEL_PATH + 下拉选择”迁移为“Playwright 上传临时 .txt 文件”

## Impact
- Affected code: `backend.py`, `templates/index.html`, `tests/test_*_e2e.py`
- Affected docs: `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `openspec/project.md`, `AGENTS.md`, `.env.example`
- Migration note: 任何依赖 `/api/novels` 或 `/api/novel/{path}` 的代码/脚本需要同步移除；用户改为在 UI 中选择本地文件。

