# Proposal: 紧凑化进度列表 UI

## 背景

当前"分析进度" UI 存在"大卡片套小卡片"的问题：
- 外层 `glass-card` 卡片包裹整个进度区域
- 内部每个 `progress-item` 又有独立的背景、边框、圆角
- 造成视觉冗余，不够紧凑

## 目标

在保留外层卡片的前提下，将进度项改为简洁的列表样式，使用分隔线替代独立卡片边框。

## 实施方案

### CSS 修改 (`static/style.css`)

将进度列表从"卡片项"改为"分隔线列表"：

```css
/* 进度列表 - 紧凑列表样式 */
.progress-list {
  display: flex;
  flex-direction: column;
  gap: 0;  /* 移除 gap，使用 border 分隔 */
}

.progress-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.875rem 1rem;
  background: transparent;  /* 移除卡片背景 */
  border: none;  /* 移除卡片边框 */
  border-bottom: 1px solid var(--glass-border);  /* 使用分隔线 */
  border-radius: 0;  /* 移除圆角 */
  transition: background 0.2s;
}

.progress-item:last-child {
  border-bottom: none;  /* 最后一项不需要分隔线 */
}

.progress-item:hover {
  background: var(--glass-hover);  /* 悬停时微弱高亮 */
}

/* 进行中状态添加轻微背景 */
.status-running .progress-item {
  background: oklch(var(--p) / 0.05);
}

/* 缩小图标尺寸 */
.progress-icon {
  width: 2rem;
  height: 2rem;
  /* 其他样式保持不变 */
}
```

### HTML

**无需修改**，现有结构完全兼容新样式。

## 效果对比

**变更前**：
```
┌─────────────────────────────┐
│  分析进度                    │
├─────────────────────────────┤
│ ┌─────────────────────────┐ │
│ │ ○ 选择小说   未开始      │ │
│ └─────────────────────────┘ │
│ ┌─────────────────────────┐ │
│ │ ○ 读取小说   未开始      │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
```

**变更后**：
```
┌─────────────────────────────┐
│  分析进度                    │
├─────────────────────────────┤
│ ○ 选择小说   未开始 ───────  │
│ ○ 读取小说   未开始 ───────  │
│ ○ 基础信息   未开始 ───────  │
└─────────────────────────────┘
```

## 验收标准

1. 进度项不再有独立卡片视觉效果
2. 列表项之间用分隔线清晰分隔
3. 状态图标颜色正常显示
4. 悬停效果正常
5. 响应式布局在移动端正常
