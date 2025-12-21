/**
 * 小说分析器 - 可视化模块（多角色版）
 * 支持多角色、多关系、性癖分析
 */

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
}

function sanitizeFilename(name) {
    const base = String(name ?? '').replace(/[\\/:*?"<>|]+/g, '_').trim();
    return base || 'report';
}

// 获取DaisyUI主题颜色
function getThemeColors() {
    const style = getComputedStyle(document.documentElement);
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        primary: style.getPropertyValue('--p') ? `oklch(${style.getPropertyValue('--p')})` : '#6366f1',
        secondary: style.getPropertyValue('--s') ? `oklch(${style.getPropertyValue('--s')})` : '#ec4899',
        info: style.getPropertyValue('--in') ? `oklch(${style.getPropertyValue('--in')})` : '#3b82f6',
        error: style.getPropertyValue('--er') ? `oklch(${style.getPropertyValue('--er')})` : '#ef4444',
        bgBase: isDark ? '#1c1c1e' : '#f2f2f7',
        textPrimary: isDark ? '#ffffff' : '#000000',
        textSecondary: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)'
    };
}

function renderRelationshipGraph(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    if (!data || (!data.characters && !data.relationships)) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">🔗</div><div class="empty-text">暂无关系数据</div></div>';
        return;
    }

    const allCharacters = data.characters || [];
    const relationships = data.relationships || [];

    // 只显示在关系中出现的角色
    const charsInRelationships = new Set();
    relationships.forEach(rel => {
        charsInRelationships.add(rel.from);
        charsInRelationships.add(rel.to);
    });

    // 过滤出有关系的角色
    const characters = allCharacters.filter(c => charsInRelationships.has(c.name));

    if (characters.length === 0 && relationships.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">🔗</div><div class="empty-text">暂无性关系数据</div></div>';
        return;
    }

    const colors = getThemeColors();
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 400;
    const centerX = width / 2;
    const centerY = height / 2;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.style.background = 'transparent';

    // 计算节点位置 - 圆形分布
    const nodes = characters.map((char, i) => {
        const angle = (i / characters.length) * 2 * Math.PI - Math.PI / 2;
        const radius = Math.min(width, height) * 0.35;
        return {
            ...char,
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
            color: char.gender === 'male' ? colors.info : colors.error
        };
    });

    // 绘制关系连线
    relationships.forEach((rel, relIndex) => {
        const source = nodes.find(n => n.name === rel.from);
        const target = nodes.find(n => n.name === rel.to);
        if (!source || !target) return;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', source.x);
        line.setAttribute('y1', source.y);
        line.setAttribute('x2', target.x);
        line.setAttribute('y2', target.y);
        line.setAttribute('stroke', colors.primary);
        line.setAttribute('stroke-width', '2');
        line.setAttribute('stroke-opacity', '0.6');
        svg.appendChild(line);

        // 关系标签 - 沿线条方向偏移避免堆叠
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;

        // 计算垂直偏移方向
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const perpX = -dy / len;  // 垂直方向
        const perpY = dx / len;

        // 根据索引偏移标签位置
        const offsetAmount = (relIndex % 3 - 1) * 18;  // -18, 0, 18
        const labelX = midX + perpX * offsetAmount;
        const labelY = midY + perpY * offsetAmount;

        // 标签背景
        const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const textLen = (rel.type?.length || 4) * 8;
        bgRect.setAttribute('x', labelX - textLen / 2 - 4);
        bgRect.setAttribute('y', labelY - 10);
        bgRect.setAttribute('width', textLen + 8);
        bgRect.setAttribute('height', '16');
        bgRect.setAttribute('fill', colors.bgBase);
        bgRect.setAttribute('rx', '4');
        svg.appendChild(bgRect);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', labelX);
        text.setAttribute('y', labelY + 3);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', colors.textSecondary);
        text.setAttribute('font-size', '10');
        text.textContent = rel.type;
        svg.appendChild(text);
    });

    // 绘制角色节点
    nodes.forEach(node => {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.style.cursor = 'pointer';

        // 外圈光晕
        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        outer.setAttribute('cx', node.x);
        outer.setAttribute('cy', node.y);
        outer.setAttribute('r', '45');
        outer.setAttribute('fill', node.color);
        outer.setAttribute('fill-opacity', '0.2');
        g.appendChild(outer);

        // 节点圆
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', node.x);
        circle.setAttribute('cy', node.y);
        circle.setAttribute('r', '35');
        circle.setAttribute('fill', node.color);
        circle.setAttribute('stroke', colors.textPrimary);
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('stroke-opacity', '0.3');
        g.appendChild(circle);

        // 性别标签
        const genderIcon = node.gender === 'male' ? 'M' : 'F';
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', node.x);
        text.setAttribute('y', node.y + 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#ffffff');
        text.setAttribute('font-size', '16');
        text.setAttribute('font-weight', '600');
        text.textContent = genderIcon;
        g.appendChild(text);

        // 角色名
        const name = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        name.setAttribute('x', node.x);
        name.setAttribute('y', node.y + 55);
        name.setAttribute('text-anchor', 'middle');
        name.setAttribute('fill', colors.textPrimary);
        name.setAttribute('font-size', '12');
        name.textContent = node.name;
        g.appendChild(name);

        // 悬停提示
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${node.name}\n${node.identity}\n${node.personality}\n\n性癖: ${node.sexual_preferences || '未知'}`;
        g.appendChild(title);

        svg.appendChild(g);
    });

    container.appendChild(svg);
}

function renderCharacters(data) {
    const container = document.getElementById('mainCharacters');
    if (!container) return;

    if (!data || !data.characters) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">👥</div><div class="empty-text">暂无角色数据</div></div>';
        return;
    }

    const males = data.characters.filter(c => c.gender === 'male');
    let females = data.characters.filter(c => c.gender === 'female');
    
    // 按淫荡指数排序女性角色（有分数的排前面，无分数的排后面）
    females = females.sort((a, b) => {
        const scoreA = a.lewdness_score ?? -1;
        const scoreB = b.lewdness_score ?? -1;
        return scoreB - scoreA;
    });
    // 添加排名（只给有分数的排名）
    let rank = 1;
    females = females.map((char) => ({
        ...char,
        lewdness_rank: char.lewdness_score ? rank++ : null
    }));

    container.innerHTML = `
        <div class="multi-char-section">
            <h3>男性角色 (${males.length})</h3>
            <div class="char-grid">
                ${males.map(char => renderCharCard(char, 'male')).join('')}
            </div>
        </div>
        <div class="multi-char-section">
            <h3>女性角色 (${females.length}) - 淫荡指数排行</h3>
            <div class="char-grid">
                ${females.map(char => renderCharCard(char, 'female')).join('')}
            </div>
        </div>
    `;
}

function renderCharCard(char, type) {
    const hasLewdness = type === 'female' && char.lewdness_score;
    const lewdnessColor = hasLewdness ? getLewdnessColor(char.lewdness_score) : '#4b5563';
    const isFemale = type === 'female';
    
    return `
        <div class="char-card ${type}">
            <div class="char-header">
                <div class="char-avatar ${type}">${type === 'male' ? 'M' : 'F'}</div>
                <div class="char-info">
                    <div class="char-name">${escapeHtml(char.name)}</div>
                    <div class="char-role">${type === 'male' ? '男性' : '女性'}</div>
                </div>
                ${isFemale ? `
                <div class="lewdness-badge" style="background: ${lewdnessColor}">
                    <span class="lewdness-rank">${hasLewdness ? '#' + (char.lewdness_rank || '?') : '-'}</span>
                    <span class="lewdness-score">${hasLewdness ? char.lewdness_score : '?'}</span>
                </div>
                ` : ''}
            </div>
            <div class="char-details">
                <div class="char-detail">
                    <span class="detail-label">身份</span>
                    <span class="detail-value">${escapeHtml(char.identity || '未知')}</span>
                </div>
                <div class="char-detail">
                    <span class="detail-label">性格</span>
                    <span class="detail-value">${escapeHtml(char.personality || '未知')}</span>
                </div>
                <div class="char-detail sexual">
                    <span class="detail-label">性癖爱好</span>
                    <span class="detail-value">${escapeHtml(char.sexual_preferences || '未知')}</span>
                </div>
                ${hasLewdness && char.lewdness_analysis ? `
                <div class="char-detail lewdness">
                    <span class="detail-label">淫荡指数分析</span>
                    <span class="detail-value lewdness-text">${escapeHtml(char.lewdness_analysis)}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

function getLewdnessColor(score) {
    if (score >= 90) return '#ef4444';  // 红色 - 极度淫荡
    if (score >= 70) return '#f97316';  // 橙色 - 非常淫荡
    if (score >= 50) return '#eab308';  // 黄色 - 中等
    if (score >= 30) return '#22c55e';  // 绿色 - 较低
    return '#6366f1';  // 蓝色 - 纯洁
}

function renderFirstSexScene(data) {
    const container = document.getElementById('firstSexScene');
    if (!container) return;

    if (!data || !data.first_sex_scenes || data.first_sex_scenes.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">💕</div><div class="empty-text">暂无首次亲密数据</div></div>';
        return;
    }

    container.innerHTML = data.first_sex_scenes.map(scene => `
        <div class="sex-scene-card">
            <div class="scene-header">
                <span class="scene-badge">首次</span>
                <span class="scene-participants">${(scene.participants || []).map(p => escapeHtml(p)).join(' + ') || '?'}</span>
            </div>
            <div class="scene-chapter">${escapeHtml(scene.chapter)}</div>
            <div class="scene-location">${escapeHtml(scene.location)}</div>
            <div class="scene-description">${escapeHtml(scene.description)}</div>
        </div>
    `).join('');
}

function renderSexSceneCount(data) {
    const container = document.getElementById('sexSceneCount');
    if (!container) return;

    if (!data || !data.sex_scenes) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📊</div><div class="empty-text">暂无统计数据</div></div>';
        return;
    }

    const scenes = data.sex_scenes;
    container.innerHTML = `
        <div class="count-display">
            <div class="count-number">${scenes.total_count || 0}</div>
            <div class="count-label">次亲密接触</div>
        </div>
        <div class="scenes-timeline">
            ${(scenes.scenes || []).slice(0, 15).map((scene, i) => `
                <div class="scene-item">
                    <div class="scene-number">${i + 1}</div>
                    <div class="scene-info">
                        <div class="scene-participants-small">${(scene.participants || []).map(p => escapeHtml(p)).join(', ') || '?'}</div>
                        <div class="scene-chapter-small">${escapeHtml(scene.chapter)}</div>
                        <div class="scene-location-small">${escapeHtml(scene.location)}</div>
                    </div>
                </div>
            `).join('')}
            ${(scenes.scenes?.length || 0) > 15 ? `<div class="more-scenes">还有 ${scenes.scenes.length - 15} 次...</div>` : ''}
        </div>
    `;
}

function renderRelationshipProgress(data) {
    const container = document.getElementById('relationshipProgress');
    if (!container) return;

    if (!data || !data.evolution || data.evolution.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📈</div><div class="empty-text">暂无发展数据</div></div>';
        return;
    }

    container.innerHTML = `
        <div class="progress-timeline">
            ${data.evolution.map((p, i) => `
                <div class="progress-item">
                    <div class="progress-dot ${i === 0 ? 'first' : ''}"></div>
                    <div class="progress-content">
                        <div class="progress-chapter">${escapeHtml(p.chapter)}</div>
                        <div class="progress-stage">${escapeHtml(p.stage)}</div>
                        <div class="progress-desc">${escapeHtml(p.description)}</div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderRelationshipSummary(data) {
    const container = document.getElementById('relationshipSummary');
    if (!container) return;

    if (!data) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📖</div><div class="empty-text">暂无数据</div></div>';
        return;
    }

    const chars = data.characters || [];
    const males = chars.filter(c => c.gender === 'male');
    const females = chars.filter(c => c.gender === 'female');
    const novelInfo = data.novel_info || {};

    container.innerHTML = `
        ${novelInfo.world_setting ? `
        <div class="novel-meta-section">
            <div class="summary-title">小说信息</div>
            <div class="novel-meta-grid">
                <div class="novel-meta-item">
                    <span class="meta-label">世界观</span>
                    <span class="meta-value">${escapeHtml(novelInfo.world_setting)}</span>
                </div>
                <div class="novel-meta-item">
                    <span class="meta-label">章节数</span>
                    <span class="meta-value">${escapeHtml(novelInfo.chapter_count || '未知')}</span>
                </div>
                <div class="novel-meta-item">
                    <span class="meta-label">状态</span>
                    <span class="meta-value ${novelInfo.is_completed ? 'completed' : 'ongoing'}">${novelInfo.is_completed ? '已完结' : '连载中'}${novelInfo.completion_note ? ' - ' + escapeHtml(novelInfo.completion_note) : ''}</span>
                </div>
            </div>
        </div>
        ` : ''}
        
        <div class="char-names-section">
            <div class="summary-title">角色一览</div>
            <div class="char-names-grid">
                <div class="char-names-group male">
                    <div class="char-group-label">男性角色 (${males.length})</div>
                    <div class="char-names-list">
                        ${males.map(c => `<span class="char-name-tag male">${escapeHtml(c.name)}</span>`).join('')}
                    </div>
                </div>
                <div class="char-names-group female">
                    <div class="char-group-label">女性角色 (${females.length})</div>
                    <div class="char-names-list">
                        ${females.map(c => `<span class="char-name-tag female">${escapeHtml(c.name)}</span>`).join('')}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <div class="summary-title">剧情总结</div>
            <div class="summary-content">${escapeHtml(data.summary || '暂无总结')}</div>
        </div>
    `;
}

function exportReport(analysis, novelName) {
    const safeNovelName = escapeHtml(novelName);
    const html = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小说分析报告 - ${safeNovelName}</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #0f0f1a; color: #e8e8f0; }
        h1 { color: #6366f1; text-align: center; border-bottom: 2px solid #6366f1; padding-bottom: 15px; }
        h2 { color: #ec4899; margin-top: 30px; border-left: 4px solid #ec4899; padding-left: 15px; }
        .section { background: #1a1a2e; padding: 20px; border-radius: 12px; margin: 20px 0; }
        .male-card { border-left: 4px solid #6366f1; }
        .female-card { border-left: 4px solid #ec4899; }
        .card { background: #252542; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .stat { font-size: 48px; color: #ef4444; text-align: center; }
        .stat-label { text-align: center; color: #a0a0b8; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #3d3d5c; }
        th { color: #6366f1; }
    </style>
</head>
<body>
    <h1>${safeNovelName} - 分析报告</h1>

    <div class="section">
        <h2>角色分析</h2>
        ${(analysis.characters || []).map(char => `
            <div class="card ${char.gender === 'male' ? 'male-card' : 'female-card'}">
                <h3>${escapeHtml(char.name)} (${char.gender === 'male' ? '男' : '女'})</h3>
                <p>身份: ${escapeHtml(char.identity || '未知')}</p>
                <p>性格: ${escapeHtml(char.personality || '未知')}</p>
                <p><strong>性癖爱好:</strong> ${escapeHtml(char.sexual_preferences || '未知')}</p>
            </div>
        `).join('') || '<p>暂无角色数据</p>'}
    </div>

    <div class="section">
        <h2>关系一览</h2>
        ${(analysis.relationships || []).map(rel => `
            <p><strong>${escapeHtml(rel.from)}</strong> → <strong>${escapeHtml(rel.to)}</strong>: ${escapeHtml(rel.type)}</p>
            <p style="color: #a0a0b8; font-size: 14px;">${escapeHtml(rel.description || '')}</p>
        `).join('') || '<p>暂无关系数据</p>'}
    </div>

    <div class="section">
        <h2>首次亲密</h2>
        ${(analysis.first_sex_scenes || []).map(scene => `
            <div class="card">
                <p><strong>参与者:</strong> ${(scene.participants || []).map(p => escapeHtml(p)).join(' + ') || '?'}</p>
                <p><strong>章节:</strong> ${escapeHtml(scene.chapter)}</p>
                <p><strong>地点:</strong> ${escapeHtml(scene.location)}</p>
                <p>${escapeHtml(scene.description)}</p>
            </div>
        `).join('') || '<p>暂无数据</p>'}
    </div>

    <div class="section">
        <h2>亲密统计</h2>
        <p class="stat">${analysis.sex_scenes?.total_count || 0}</p>
        <p class="stat-label">次亲密接触</p>
        <table>
            <tr><th>次数</th><th>章节</th><th>参与者</th><th>地点</th></tr>
            ${(analysis.sex_scenes?.scenes || []).map((s, i) => `
                <tr><td>${i + 1}</td><td>${escapeHtml(s.chapter)}</td><td>${(s.participants || []).map(p => escapeHtml(p)).join(', ') || '?'}</td><td>${escapeHtml(s.location)}</td></tr>
            `).join('') || ''}
        </table>
    </div>

    <div class="section">
        <h2>关系发展</h2>
        ${(analysis.evolution || []).map(p => `
            <div class="card">
                <p><strong>${escapeHtml(p.stage)}</strong> (${escapeHtml(p.chapter)})</p>
                <p>${escapeHtml(p.description)}</p>
            </div>
        `).join('') || '<p>暂无数据</p>'}
    </div>

    <div class="section">
        <h2>总结</h2>
        <p>${escapeHtml(analysis.summary || '无')}</p>
    </div>
</body>
</html>
    `;

    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${sanitizeFilename(novelName).replace(/\\.txt$/i, '')}_分析报告.html`;
    a.click();
    URL.revokeObjectURL(url);
}
