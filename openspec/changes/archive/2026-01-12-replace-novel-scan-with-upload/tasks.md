# 实施任务清单

## 1. 后端：移除本地扫描/读盘 API（BREAKING）

- [x] 1.1 删除 `GET /api/novels` 与 `GET /api/novel/{path}`
  - 文件：`backend.py`
  - 验证：请求旧接口返回 404

- [x] 1.2 删除 `NOVEL_PATH`、`DEFAULT_NOVEL_PATH`、`_safe_novel_path()`
  - 文件：`backend.py`
  - 验证：不设置 `NOVEL_PATH` 也能启动服务并访问 `/api/config`

## 2. 前端：改为上传/选择文件导入

- [x] 2.1 替换“选择小说(下拉列表)”为“选择文件”入口
  - 文件：`templates/index.html`
  - 要求：支持选择单个 `.txt` 文件；选择后显示文件名，并更新进度（select/load）

- [x] 2.2 移除与旧接口耦合的逻辑
  - 文件：`templates/index.html`
  - 删除：`scanNovels()` / `renderFolderList()` / 对 `/api/novels` 与 `/api/novel/*` 的 fetch

- [x] 2.3 保持分析流程不变
  - 文件：`templates/index.html`
  - 要求：`analyzeNovel()` 与 `/api/analyze/*` 仍使用 `currentNovelContent`

## 3. 测试：E2E 全量迁移到文件上传

- [x] 3.1 更新 E2E：不再使用 `NOVEL_PATH` 搭建目录
  - 文件：`tests/test_thunderzones_e2e.py`, `tests/test_export_report_e2e.py`
  - 要求：生成临时 `.txt` 文件，并通过 Playwright 上传到页面后运行分析

- [x] 3.2 更新 E2E 选择小说辅助方法
  - 文件：`tests/test_thunderzones_e2e.py`, `tests/test_export_report_e2e.py`
  - 要求：从 `_select_first_novel()` 迁移为 `_upload_novel()`（稳定等待内容就绪）

## 4. 文档：删除旧接口/配置引用

- [x] 4.1 更新文档与说明
  - 文件：`README.md`, `CLAUDE.md`, `CHANGELOG.md`, `openspec/project.md`, `AGENTS.md`, `.env.example`
  - 要求：移除 `NOVEL_PATH`、`/api/novels`、`/api/novel/{path}` 的描述，替换为“本地文件上传导入”

## 5. 彻底清理：拒绝旧代码残留

- [x] 5.1 全库搜索旧关键词，确保零残留
  - 搜索关键词：`NOVEL_PATH`, `/api/novels`, `/api/novel/`, `_safe_novel_path`, `scanNovels(`
  - 要求：除变更提案/历史归档外，代码与当前文档不再出现上述关键词

- [x] 5.2 运行测试
  - `python -m pytest -q`
  - （如需）`python -m pytest tests/test_*_e2e.py -q`

## 6. OpenSpec 校验

- [x] 6.1 `openspec validate replace-novel-scan-with-upload --strict`
- [x] 6.2 完成后将本任务清单全部勾选为 `[x]`
