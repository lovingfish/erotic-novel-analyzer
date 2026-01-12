# novel-input Delta Specification

## ADDED Requirements

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

## MODIFIED Requirements

无

## REMOVED Requirements

无（旧行为未在现有 specs 中定义；以“新增约束”方式表达移除结果）

