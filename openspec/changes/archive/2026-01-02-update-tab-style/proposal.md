# Change: Update Tab Style for Visual Consistency

## Why
当前的 tab 样式不一致：只有被选中的 tab 在浅色模式下有 `box-shadow` 边框效果，其他 tab 没有任何视觉边界。这导致选中和未选中状态的视觉差异不够统一。

## What Changes
- 将 pill-tab 样式从"胶囊按钮"改为"下划线 tab"风格
- 所有 tab 共享底部边框线作为基准
- 选中的 tab 使用主题色下划线高亮
- 移除浅色模式下的特殊 `box-shadow` 处理

## Impact
- Affected files: `static/style.css`
- No breaking changes
- Pure UI/cosmetic change
