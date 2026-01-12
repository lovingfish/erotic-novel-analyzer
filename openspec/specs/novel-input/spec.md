# novel-input Specification

## Purpose
Provide a flexible, local-first novel import flow: users select a `.txt` file in the browser, the frontend decodes it and feeds `content` into existing `/api/analyze/*` endpoints. The backend MUST NOT scan/read local novel files.
## Requirements
### Requirement: 本地文件选择导入小说
前端 SHALL 提供一个文件选择入口，允许用户从本机选择单个 `.txt` 文件作为待分析小说。

#### Scenario: 成功选择 `.txt` 文件
- **WHEN** 用户选择一个 `.txt` 文件
- **THEN** 系统读取文件内容并写入 `currentNovelContent`
- **AND** `selectedNovel.name` 显示为文件名
- **AND** 更新“选择/读取”进度为完成状态（可见字数/长度信息）

#### Scenario: 选择非 `.txt` 文件
- **WHEN** 用户选择一个非 `.txt` 文件
- **THEN** 系统拒绝加载并提示“仅支持 .txt”
- **AND** `currentNovelContent` 保持为空

### Requirement: 编码自动识别与手动切换
前端 SHALL 支持对导入文本进行编码解码，默认使用“自动”模式，并提供手动切换编码的能力。

#### Scenario: BOM 优先
- **WHEN** 文件包含 UTF BOM
- **THEN** 系统按 BOM 指示的编码解码（如 UTF-8 / UTF-16LE / UTF-16BE）

#### Scenario: 自动模式优先严格 UTF-8，再回退 GB18030
- **WHEN** 用户使用“自动”模式导入文件且无 BOM
- **THEN** 系统先尝试严格 UTF-8 解码
- **AND** 当严格 UTF-8 解码失败时回退到 GB18030（或 GBK 兼容回退）

#### Scenario: 手动选择编码
- **WHEN** 用户手动选择 “UTF-8” 或 “GB18030”
- **THEN** 系统按用户选择进行解码并重新载入内容

### Requirement: 禁止目录扫描与路径读盘 API
后端 MUST NOT 提供任何“递归扫描目录”或“按相对路径读取本地小说文件”的 API。

#### Scenario: 旧接口不再存在
- **WHEN** 客户端请求 `GET /api/novels` 或 `GET /api/novel/{path}`
- **THEN** 服务返回 404

### Requirement: 移除 NOVEL_PATH
系统 SHALL 移除 `NOVEL_PATH` 环境变量的使用与文档引用。

#### Scenario: 启动不依赖 NOVEL_PATH
- **WHEN** 服务启动且未设置 `NOVEL_PATH`
- **THEN** 服务仍可正常启动并提供 UI 与 `/api/analyze/*`
