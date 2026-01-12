# ui-interaction-flow Specification

## Purpose
Define UI interaction rules around imported novel state and analysis lifecycle: which tabs are enabled before/after analysis, and where the “已导入文件” status card may appear.
## Requirements
### Requirement: Tab 启用逻辑
Tab 的启用状态 SHALL 根据分析结果状态和 tab 类型进行控制。

#### Scenario: 导入小说后，只有特定 tab 可用
- **WHEN** 用户导入小说文件但未开始分析（`hasAnyResult() === false`）
- **THEN** "进度" tab 可用
- **AND** "日志" tab 可用
- **AND** "调试" tab 可用
- **AND** "总结"、"雷点"、"角色"、"关系图"、"首次"、"统计"、"发展" tab 禁用（显示 40% 不透明度 + 不可点击）

#### Scenario: 分析完成后，所有 tab 可用
- **WHEN** 分析完成（`hasAnyResult() === true`）
- **THEN** 所有 10 个 tab 均可用
- **AND** 自动切换到"总结" tab

### Requirement: 导入文件状态卡片显示范围
"已导入文件"状态卡片 SHALL 仅在"进度" tab 显示，避免跨 tab 重复。

#### Scenario: 在"进度" tab 显示导入状态
- **WHEN** 用户导入小说文件且 `currentTab === 'pipeline'` 且 `!hasAnyResult()`
- **THEN** 显示"已导入文件"卡片
- **AND** 卡片包含：文件名、字数、"点击开始分析"提示

#### Scenario: 在其他 tab 不显示导入状态卡片
- **WHEN** 用户切换到"总结"、"日志"、"调试"或其他 tab
- **THEN** 不显示"已导入文件"卡片（即使未分析）

### Requirement: Tab 配置数据结构
tabs 数组 SHALL 配置每个 tab 的 id、name、以及可选的 `alwaysEnabled` 属性。

#### Scenario: "进度" tab 始终可用
- **WHEN** tabs 数组被初始化
- **THEN** "进度" tab 设置 `alwaysEnabled: true`

#### Scenario: "总结" tab 仅分析后可用
- **WHEN** tabs 数组被初始化
- **THEN** "总结" tab 不设置 `alwaysEnabled` 属性（或设置为 `false`/`undefined`）

#### Scenario: "日志" tab 始终可用
- **WHEN** tabs 数组被初始化
- **THEN** "日志" tab 设置 `alwaysEnabled: true`

#### Scenario: "调试" tab 始终可用
- **WHEN** tabs 数组被初始化
- **THEN** "调试" tab 设置 `alwaysEnabled: true`

### Requirement: E2E 测试覆盖
Tab 交互逻辑 SHALL 通过 E2E 测试验证，覆盖分析前后的 tab 状态和卡片显示逻辑。

#### Scenario: 测试导入文件后的 tab 禁用状态
- **WHEN** E2E 测试导入小说文件但不运行分析
- **THEN** 验证"进度"、"日志"、"调试" tab 可用
- **AND** 验证"总结"及其他分析 tab 被禁用（disabled 属性 + opacity-40 样式）

#### Scenario: 测试"已导入文件"卡片仅在"进度" tab 显示
- **WHEN** E2E 测试导入小说文件并切换 tab
- **THEN** 在"进度" tab 验证卡片可见
- **AND** 在"日志" tab 验证卡片不可见
- **AND** 在"调试" tab 验证卡片不可见

#### Scenario: 测试分析完成后的 tab 状态和自动切换
- **WHEN** E2E 测试运行完整分析流程
- **THEN** 验证自动切换到"总结" tab
- **AND** 验证所有 tab 均可用
- **AND** 验证"已导入文件"卡片在所有 tab 都不可见（因为 hasAnyResult() === true）
