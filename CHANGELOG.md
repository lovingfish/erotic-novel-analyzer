# Changelog

本文件记录 LLM 交互链路的鲁棒性演进与关键架构变更。

## [4.0.0] - 2026-01-12

#### 破坏性变更
- 小说输入改为前端本地文件导入：不再扫描目录/按路径读取本地文件
- 移除 `/api/novels`、`/api/novel/{path}`、`NOVEL_PATH`、`_safe_novel_path()`

#### 前端
- 新增“选择文件”入口，支持 UTF-8/GB18030 自动识别，且可手动切换编码

#### 测试与文档
- E2E 流程改为“上传文件→分析”
- 文档移除旧接口与 `NOVEL_PATH` 配置

## [3.0.0] - 2025-12-28

### b1：彻底重构（强制 Function Calling + Pydantic Schema + 固定 YAML 配置）

#### 破坏性变更
- LLM 策略参数不再从 `.env` 读取（删除所有 `LLM_*` 配置入口）
- 后端固定只读 `config/llm.yaml` 作为 LLM 策略配置（温度/截断/retry/repair）
- Prompt 不再使用根目录 `prompts.py`，改为 `config/prompts/*.j2`（Jinja2 模板）
- 移除“纯文本 + 正则抽 JSON”主链路：仅接受 `tool_calls[].function.arguments`

#### 结构化输出与修复
- Schema 作为单一事实来源：Pydantic Schema 同时用于 tool schema 生成与运行时校验
- Repair Pass：当 schema 校验失败时，最多触发 1 次修复（仍使用同一 tool schema）
- `/api/test-connection` 增强：会显式验证 Function Calling 是否可用

#### 可观测性
- 关键路径新增结构化日志（JSON Lines）：retry / repair / truncation / 协议回退等事件

#### 代码结构
- `backend.py` 精简为路由层
- 新增 `src/novel_analyzer/*`：`llm_client.py`、`schemas.py`、`config_loader.py`、`content_processor.py`、`observability.py`、`validators.py`

#### 测试
- 替换旧 `tests/test_llm_robustness.py`：新增配置加载、截断、function calling、repair、日志等单测

---

## [2.x] - 2025-12-28 (legacy)

### a1：将四个分析端点的硬编码 Prompt 抽取到独立模块（`prompts.py`）

#### 代码改动
- 新增 `prompts.py`
  - `build_meta_prompt(content: str) -> str`
  - `build_core_prompt(content: str) -> str`
  - `build_scenes_prompt(content: str, allowed_names_json: str, relationships_json: str) -> str`
  - `build_thunderzones_prompt(content: str, allowed_names_json: str, relationships_json: str) -> str`
- `backend.py`
  - 分析端点不再内联大段 prompt 字符串，改为调用上述 `build_*_prompt`。

#### 架构设计
- 将系统分层为：
  - Prompt 层（`prompts.py`）：负责“需求/约束/输出格式”的表达。
  - 编排层（`backend.py` 端点）：负责组装参数、调用 LLM、校验与返回。
- 通过模块边界降低耦合：后续迭代 prompt 不需要在端点逻辑中穿插修改。

#### 为什么这样做 / 设计考量
- 可维护性：prompt 变化频繁，抽离后更易定位、更少干扰业务逻辑 diff。
- 可测试性：prompt 构建逻辑更集中，便于做“输入->prompt”快照/断言。
- 为后续鲁棒性增强做铺垫：Repair Pass 需要提取 prompt 头部（依赖 `## Novel Content` 标记），统一管理更可靠。

---

### a2：降低结构化抽取默认温度，并支持环境变量配置

#### 代码改动
- `backend.py`
  - `_call_llm_json(...)` 中对结构化抽取温度使用更低默认值（默认 `0.2`）。
  - 增加环境变量读取：`LLM_TEMPERATURE_STRUCTURED`，并对值做范围钳制（`0.0 ~ 2.0`）。
  - 同一温度参数同时用于：
    - 纯文本模式：`call_llm_with_response(..., temperature=...)`
    - Function Calling 模式：`call_llm_with_tool_call(..., temperature=...)`
- `.env.example`
  - 增加 `LLM_TEMPERATURE_STRUCTURED=0.2`

#### 架构设计
- 将温度视为“运行时可调策略”，由环境变量注入而不是写死在代码里。

#### 为什么这样做 / 设计考量
- 结构化 JSON 输出更依赖稳定性：较低温度可显著减少“发散输出、夹杂解释文本、字段漂移”等问题。
- 不同模型/不同 API 服务的最优温度不同：需要可配置以便线上快速调整。

---

### a3：加入可选的 Function Calling 结构化输出模式，并保留纯文本模式回退

#### 代码改动
- `backend.py`
  - 增加按 section 生成 tool schema：`_get_function_calling_tool(section)`（Meta/Core/Scenes/Thunder 四类）。
  - `_call_llm_json(...)` 增加开关：`LLM_USE_FUNCTION_CALLING`。
    - 开启时优先调用 `call_llm_with_tool_call(...)` 获取 tool arguments（结构化 JSON）。
    - 若 tool 调用失败或未返回可用数据，则自动回退到纯文本模式：`call_llm_with_response(...) + extract_json_from_response(...)`。

#### 架构设计
- 将“结构化获取”设计为可选能力：
  - 主路径：Function Calling（当 API/模型支持时更稳）。
  - 回退路径：纯文本 JSON 提取（适配不支持或不稳定的服务端实现）。
- 统一入口：四个端点只调用 `_call_llm_json(...)`，由该函数内部决定使用哪种模式。

#### 为什么这样做 / 设计考量
- 兼容性约束：目标 OpenAI 兼容 API 不一定支持 `response_format`，但模型普遍支持 Function Calling。
- 防脆弱：不能假设所有服务端都正确实现 tool calling，因此必须有可靠回退路径。
- 失败隔离：tool calling 失败不影响整体流程，仍可尽力通过纯文本解析完成任务。

---

### a4：加入长文本稳健性处理（内容采样/截断策略可配置）

#### 代码改动
- `backend.py`
  - 新增环境变量解析工具：`_env_bool(...)`、`_env_int(...)`。
  - 新增内容预处理：`_prepare_llm_content(content: str, section: str) -> str`
    - 通过 `LLM_CONTENT_MAX_CHARS` 控制最大字符数。
    - 通过 `LLM_CONTENT_STRATEGY` 控制采样策略：
      - `head` / `tail` / `head_tail` / `head_middle_tail`
    - 支持按 section 覆盖：
      - `LLM_CONTENT_MAX_CHARS_META`、`LLM_CONTENT_STRATEGY_META` 等。
  - 四个分析端点在构建 prompt 前调用 `_prepare_llm_content(...)`，避免超长输入导致模型输出被截断，出现“半截 JSON”。
- `.env.example`
  - 增加：
    - `LLM_CONTENT_MAX_CHARS=24000`
    - `LLM_CONTENT_STRATEGY=head_middle_tail`

#### 架构设计
- 在“prompt 构建前”做内容策略处理：
  - 端点层负责选择 section；
  - 内容策略集中在 `_prepare_llm_content`，避免多处重复实现。

#### 为什么这样做 / 设计考量
- 超长 prompt 是导致 JSON 不完整（被截断）的主要原因之一。
- 采用字符级（char）预算而非 token 估算：实现简单、无 tokenizer 依赖；对大多数场景足够有效。
- `head_middle_tail`：尽量保留开头/中段/结尾信息，兼顾角色引入、关键转折与结局信息；同时使用 `...[TRUNCATED]...` 标记让模型知道内容被截断。

---

### a5：加入 Repair Pass（解析/校验失败时自动二次修复并重试一次，可配置）

#### 代码改动
- `backend.py`
  - 新增 Repair Prompt 构建与截断：
    - `_truncate_text(...)`
    - `_extract_prompt_header(...)`（基于 `## Novel Content` 提取 prompt 头部摘要）
    - `_should_repair(section)`（支持全局 + 分 section 开关）
    - `_build_repair_prompt(section, original_prompt, bad_data, reason, errors)`
    - `_repair_llm_json(...)`（调用一次修复）
  - 在 `_call_llm_json(...)` 中加入“解析失败修复”：
    - 当 `extract_json_from_response` 失败时，若启用 Repair Pass，则用修复 prompt 再请求一次。
  - 在四个分析端点加入“校验失败修复”：
    - 首次校验失败时，把“模型输出 + 校验错误列表”喂给 Repair Prompt 再请求一次。
  - 使用 `repair_state` 共享状态：确保**同一请求最多只触发一次 Repair Pass**（解析修复与校验修复互斥），避免无限重试与成本失控。
- `.env.example`
  - 增加：
    - `LLM_REPAIR_ENABLED=true`
    - `LLM_REPAIR_PROMPT_HEAD_MAX_CHARS=8000`
    - `LLM_REPAIR_BAD_OUTPUT_MAX_CHARS=6000`
- `tests/test_llm_robustness.py`
  - 新增单测覆盖：
    - 长文本采样/截断策略与 section 覆盖
    - `_call_llm_json` 解析失败触发 Repair Pass
    - `analyze_meta` 校验失败触发 Repair Pass
    - 验证“解析修复已发生时不再二次触发校验修复”

#### 架构设计
- Repair Pass 是“可选的二次兜底层”，插入点明确：
  - 解析层失败：在 `_call_llm_json` 内处理。
  - 结构/字段校验失败：在端点校验之后处理（因为端点掌握更具体的业务校验规则与错误列表）。
- 统一开关策略：全局 + 分 section 覆盖，便于快速定位“哪个模块更容易坏”。

#### 为什么这样做 / 设计考量
- 模型输出的常见失败模式：
  - JSON 外层夹杂自然语言/markdown
  - 字段缺失/类型错误/空数组
  - 部分字段有效但整体不满足校验
- Repair Prompt 提供三类关键信息来提升修复成功率：
  - 原始要求摘要（prompt header）
  - 坏输出（bad output）
  - 具体校验错误列表（validation errors）
- 明确限制“只重试一次”：
  - 防止死循环与费用失控
  - 失败时优先返回原始错误信息（同时在 DEBUG 模式可查看截断的原始响应片段）

---

## 配置项汇总（v3）

### `.env`（仅 secrets / 环境特定）
- `API_BASE_URL`
- `API_KEY`
- `MODEL_NAME`
- `HOST` / `PORT`
- `LOG_LEVEL` / `DEBUG`

### `config/llm.yaml`（固定读取）
- `defaults.timeout_seconds`
- `defaults.retry.*`
- `content_processing.*`
- `repair.*`
- `sections.*`（per-section 温度、tool name、prompt 模板）
