# ui-interaction-flow Delta Specification

## ADDED Requirements

无

## MODIFIED Requirements

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

## REMOVED Requirements

无

