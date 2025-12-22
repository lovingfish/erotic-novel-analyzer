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

function buildQuickStatsHtml({ sexCount, relationshipCount, characterCount, wordCountText } = {}) {
    const safeSexCount = escapeHtml(sexCount ?? 0);
    const safeRelationshipCount = escapeHtml(relationshipCount ?? 0);
    const safeCharacterCount = escapeHtml(characterCount ?? 0);
    const safeWordCountText = escapeHtml(wordCountText ?? '0');

    return `
        <div class="novel-meta-section">
            <div class="quick-stats-grid">
                <div class="quick-stat">
                    <div class="quick-stat-title">亲密次数</div>
                    <div class="quick-stat-value text-primary">${safeSexCount}</div>
                </div>
                <div class="quick-stat">
                    <div class="quick-stat-title">关系对</div>
                    <div class="quick-stat-value text-secondary">${safeRelationshipCount}</div>
                </div>
                <div class="quick-stat">
                    <div class="quick-stat-title">角色数</div>
                    <div class="quick-stat-value">${safeCharacterCount}</div>
                </div>
                <div class="quick-stat">
                    <div class="quick-stat-title">字数</div>
                    <div class="quick-stat-value">${safeWordCountText}</div>
                </div>
            </div>
        </div>
    `;
}

function renderQuickStats(container, stats) {
    if (!container) return;
    container.replaceChildren();
    container.insertAdjacentHTML('beforeend', buildQuickStatsHtml(stats));
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

function getExportThemeColors(isDark) {
    const dark = Boolean(isDark);
    return {
        primary: '#6366f1',
        secondary: '#ec4899',
        info: '#3b82f6',
        error: '#ef4444',
        bgBase: dark ? '#1c1c1e' : '#f2f2f7',
        textPrimary: dark ? '#ffffff' : '#000000',
        textSecondary: dark ? '#ffffffb3' : '#00000099'
    };
}

function buildRelationshipSvgHtml(data, { width = 1200, height = 800, isDark } = {}) {
    if (!data || (!data.characters && !data.relationships)) {
        return '<div class="empty-state"><div class="empty-icon">🔗</div><div class="empty-text">暂无关系数据</div></div>';
    }

    const allCharacters = Array.isArray(data.characters) ? data.characters : [];
    const relationships = Array.isArray(data.relationships) ? data.relationships : [];

    const charsInRelationships = new Set();
    relationships.forEach(rel => {
        if (typeof rel?.from === 'string') charsInRelationships.add(rel.from);
        if (typeof rel?.to === 'string') charsInRelationships.add(rel.to);
    });

    const characters = allCharacters.filter(c => c && typeof c.name === 'string' && charsInRelationships.has(c.name));

    if (characters.length === 0 && relationships.length === 0) {
        return '<div class="empty-state"><div class="empty-icon">🔗</div><div class="empty-text">暂无性关系数据</div></div>';
    }

    const displayWidth = Math.max(1, Math.round(Number(width) || 1200));
    const displayHeight = Math.max(1, Math.round(Number(height) || 800));
    const colors = getExportThemeColors(Boolean(isDark));

    const viewBoxWidth = 1200;
    const viewBoxHeight = 800;
    const centerX = viewBoxWidth / 2;
    const centerY = viewBoxHeight / 2;
    const radius = Math.min(viewBoxWidth, viewBoxHeight) * 0.35;

    const nodes = characters.map((char, i) => {
        const angle = (i / Math.max(1, characters.length)) * 2 * Math.PI - Math.PI / 2;
        const gender = char.gender;
        const nodeColor = gender === 'male' ? colors.info : (gender === 'female' ? colors.error : colors.primary);
        return {
            ...char,
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
            color: nodeColor
        };
    });

    const nodeByName = new Map(nodes.map(node => [node.name, node]));

    const edgesHtml = relationships.map((rel, relIndex) => {
        const source = typeof rel?.from === 'string' ? nodeByName.get(rel.from) : null;
        const target = typeof rel?.to === 'string' ? nodeByName.get(rel.to) : null;
        if (!source || !target) return '';

        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const perpX = -dy / len;
        const perpY = dx / len;

        const offsetAmount = (relIndex % 3 - 1) * 18;
        const labelX = midX + perpX * offsetAmount;
        const labelY = midY + perpY * offsetAmount;

        const labelText = String(rel?.type ?? '');
        const labelCharLen = Math.max(4, Array.from(labelText).length);
        const textLen = labelCharLen * 8;
        const rectX = labelX - textLen / 2 - 6;
        const rectY = labelY - 10;

        return `
            <line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="${colors.primary}" stroke-width="2" stroke-opacity="0.6"></line>
            <rect x="${rectX}" y="${rectY}" width="${textLen + 12}" height="16" fill="${colors.bgBase}" rx="4"></rect>
            <text x="${labelX}" y="${labelY + 3}" text-anchor="middle" fill="${colors.textSecondary}" font-size="10">${escapeHtml(labelText)}</text>
        `;
    }).join('');

    const nodesHtml = nodes.map(node => {
        const genderIcon = node.gender === 'male' ? 'M' : 'F';
        const titleText = [
            node.name,
            node.identity || '未知',
            node.personality || '未知',
            '',
            `性癖: ${node.sexual_preferences || '未知'}`
        ].join('\n');

        return `
            <g>
                <circle cx="${node.x}" cy="${node.y}" r="45" fill="${node.color}" fill-opacity="0.2"></circle>
                <circle cx="${node.x}" cy="${node.y}" r="35" fill="${node.color}" stroke="${colors.textPrimary}" stroke-width="2" stroke-opacity="0.3"></circle>
                <text x="${node.x}" y="${node.y + 5}" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">${escapeHtml(genderIcon)}</text>
                <text x="${node.x}" y="${node.y + 55}" text-anchor="middle" fill="${colors.textPrimary}" font-size="12">${escapeHtml(node.name)}</text>
                <title>${escapeHtml(titleText)}</title>
            </g>
        `;
    }).join('');

    return `
        <svg xmlns="http://www.w3.org/2000/svg" width="${displayWidth}" height="${displayHeight}" viewBox="0 0 1200 800">
            <rect x="0" y="0" width="1200" height="800" fill="${colors.bgBase}"></rect>
            ${edgesHtml}
            ${nodesHtml}
        </svg>
    `;
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

function buildCharactersHtml(data) {
    if (!data || !data.characters || data.characters.length === 0) {
        return '<div class="empty-state"><div class="empty-icon">👥</div><div class="empty-text">暂无角色数据</div></div>';
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

    return `
        <div class="multi-char-section">
            <h3>男性角色 (${males.length})</h3>
            <div class="char-grid">
                ${males.map(char => buildCharCardHtml(char, 'male')).join('')}
            </div>
        </div>
        <div class="multi-char-section">
            <h3>女性角色 (${females.length}) - 淫荡指数排行</h3>
            <div class="char-grid">
                ${females.map(char => buildCharCardHtml(char, 'female')).join('')}
            </div>
        </div>
    `;
}

function renderCharacters(data) {
    const container = document.getElementById('mainCharacters');
    if (!container) return;
    container.innerHTML = buildCharactersHtml(data);
}

function buildCharCardHtml(char, type) {
    if (!char) return '';
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

function renderCharCard(char, type) {
    return buildCharCardHtml(char, type);
}

function getLewdnessColor(score) {
    if (score >= 90) return '#ef4444';  // 红色 - 极度淫荡
    if (score >= 70) return '#f97316';  // 橙色 - 非常淫荡
    if (score >= 50) return '#eab308';  // 黄色 - 中等
    if (score >= 30) return '#22c55e';  // 绿色 - 较低
    return '#6366f1';  // 蓝色 - 纯洁
}

function buildFirstSexSceneHtml(data) {
    if (!data || !data.first_sex_scenes || data.first_sex_scenes.length === 0) {
        return '<div class="empty-state"><div class="empty-icon">💕</div><div class="empty-text">暂无首次亲密数据</div></div>';
    }

    return data.first_sex_scenes.map(scene => `
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

function renderFirstSexScene(data) {
    const container = document.getElementById('firstSexScene');
    if (!container) return;

    container.innerHTML = buildFirstSexSceneHtml(data);
}

function buildSexSceneCountHtml(data) {
    if (!data || !data.sex_scenes) {
        return '<div class="empty-state"><div class="empty-icon">📊</div><div class="empty-text">暂无统计数据</div></div>';
    }

    const scenes = data.sex_scenes;
    return `
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

function renderSexSceneCount(data) {
    const container = document.getElementById('sexSceneCount');
    if (!container) return;

    container.innerHTML = buildSexSceneCountHtml(data);
}

function buildRelationshipProgressHtml(data) {
    if (!data || !data.evolution || data.evolution.length === 0) {
        return '<div class="empty-state"><div class="empty-icon">📈</div><div class="empty-text">暂无发展数据</div></div>';
    }

    return `
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

function renderRelationshipProgress(data) {
    const container = document.getElementById('relationshipProgress');
    if (!container) return;

    container.innerHTML = buildRelationshipProgressHtml(data);
}

function buildRelationshipSummaryHtml(data) {
    if (!data) {
        return '<div class="empty-state"><div class="empty-icon">📖</div><div class="empty-text">暂无数据</div></div>';
    }

    const chars = data.characters || [];
    const males = chars.filter(c => c.gender === 'male');
    const females = chars.filter(c => c.gender === 'female');
    const novelInfo = data.novel_info || {};

    return `
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

function renderRelationshipSummary(data) {
    const container = document.getElementById('relationshipSummary');
    if (!container) return;

    container.innerHTML = buildRelationshipSummaryHtml(data);
}

function exportReport(analysis, novelName, opts = {}) {
    const safeNovelName = escapeHtml(novelName);
    const theme = document.documentElement.getAttribute('data-theme') || 'dark';
    const generatedAt = escapeHtml(opts?.generatedAt || new Date().toLocaleString());

    const toInt = (value, fallback = 0) => {
        const n = Number(value);
        return Number.isFinite(n) ? Math.trunc(n) : fallback;
    };

    const wordCount = toInt(opts?.wordCount, 0);
    const wordCountText = wordCount > 10000 ? (wordCount / 10000).toFixed(1) + '万' : String(wordCount);

    const sexCount = toInt(analysis?.sex_scenes?.total_count, 0);
    const relationshipCount = Array.isArray(analysis?.relationships) ? analysis.relationships.length : 0;
    const characterCount = Array.isArray(analysis?.characters) ? analysis.characters.length : 0;
    
    const styleCss = `
:root { --radius: 0.75rem; --border-color: oklch(var(--bc) / 0.1); }
[x-cloak] { display: none !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: oklch(var(--bc) / 0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: oklch(var(--bc) / 0.3); }
#relationshipChart { position: relative; overflow: hidden; }
#relationshipChart svg { width: 100%; height: 100%; }
.char-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; }
.char-card { background: oklch(var(--b2)); border: 1px solid var(--border-color); border-radius: var(--radius); padding: 1.25rem; transition: border-color 0.2s; }
.char-card:hover { border-color: oklch(var(--bc) / 0.2); }
.char-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.char-avatar { width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; color: white; }
.char-avatar.male { background: oklch(var(--in)); }
.char-avatar.female { background: oklch(var(--er)); }
.char-name { font-size: 1.125rem; font-weight: 600; }
.char-role { font-size: 0.75rem; opacity: 0.6; margin-top: 0.25rem; }
.char-details { display: flex; flex-direction: column; gap: 0.75rem; }
.char-detail { display: flex; flex-direction: column; gap: 0.25rem; }
.detail-label { font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; }
.detail-value { font-size: 0.875rem; opacity: 0.8; }
.char-detail.sexual { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid oklch(var(--bc) / 0.1); }
.char-detail.lewdness { margin: 0.75rem -1.25rem -1.25rem; padding: 1rem 1.25rem; background: oklch(var(--er) / 0.1); border-radius: 0 0 var(--radius) var(--radius); }
.lewdness-text { color: oklch(var(--er)); font-style: italic; }
.lewdness-badge { display: flex; flex-direction: column; align-items: center; padding: 0.5rem 0.75rem; border-radius: 0.5rem; margin-left: auto; }
.lewdness-rank { font-size: 0.7rem; opacity: 0.8; }
.lewdness-score { font-size: 1.25rem; font-weight: 700; color: white; }
.multi-char-section { margin-bottom: 1.5rem; }
.multi-char-section h3 { font-size: 0.875rem; font-weight: 600; opacity: 0.7; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid oklch(var(--bc) / 0.1); }
.sex-scene-card { background: oklch(var(--b2)); border: 1px solid var(--border-color); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1rem; }
.scene-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
.scene-badge { background: oklch(var(--er)); color: white; padding: 0.25rem 0.75rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 600; }
.scene-participants { color: oklch(var(--p)); font-weight: 600; }
.scene-chapter { font-weight: 600; }
.scene-location { font-size: 0.875rem; opacity: 0.7; margin-bottom: 0.75rem; }
.scene-description { font-size: 0.875rem; opacity: 0.8; line-height: 1.6; }
.count-display { text-align: center; padding: 2rem; background: oklch(var(--b2)); border: 1px solid var(--border-color); border-radius: var(--radius); margin-bottom: 1.5rem; }
.count-number { font-size: 4rem; font-weight: 700; background: linear-gradient(135deg, oklch(var(--p)), oklch(var(--s))); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.count-label { font-size: 1rem; opacity: 0.6; margin-top: 0.5rem; }
.scenes-timeline { display: flex; flex-direction: column; gap: 0.75rem; }
.scene-item { display: flex; align-items: center; padding: 0.75rem 1rem; background: oklch(var(--b2)); border: 1px solid var(--border-color); border-radius: calc(var(--radius) - 0.25rem); }
.scene-number { width: 2rem; height: 2rem; border-radius: 50%; background: oklch(var(--p)); color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; margin-right: 0.75rem; flex-shrink: 0; }
.scene-info { flex: 1; min-width: 0; }
.scene-participants-small { font-weight: 500; margin-bottom: 0.25rem; }
.scene-chapter-small { font-size: 0.875rem; }
.scene-location-small { font-size: 0.75rem; opacity: 0.6; }
.more-scenes { text-align: center; padding: 0.75rem; opacity: 0.6; font-size: 0.875rem; }
.progress-timeline { position: relative; padding-left: 1.5rem; }
.progress-timeline::before { content: ''; position: absolute; left: 0.5rem; top: 0; bottom: 0; width: 2px; background: oklch(var(--bc) / 0.1); }
.progress-item { position: relative; padding: 1rem; background: oklch(var(--b2)); border: 1px solid var(--border-color); border-radius: calc(var(--radius) - 0.25rem); margin-bottom: 0.75rem; }
.progress-dot { position: absolute; left: -1.5rem; top: 1.25rem; width: 0.75rem; height: 0.75rem; border-radius: 50%; background: oklch(var(--p)); border: 2px solid oklch(var(--b1)); }
.progress-dot.first { background: oklch(var(--er)); width: 1rem; height: 1rem; left: -1.625rem; }
.progress-chapter { font-size: 0.75rem; opacity: 0.6; margin-bottom: 0.25rem; }
.progress-stage { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
.progress-desc { font-size: 0.875rem; opacity: 0.8; line-height: 1.5; }
.summary-section { background: oklch(var(--b2)); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: var(--radius); margin-top: 1rem; }
.summary-title { font-size: 0.875rem; font-weight: 600; color: oklch(var(--p)); margin-bottom: 0.75rem; }
.summary-content { font-size: 0.9375rem; opacity: 0.8; line-height: 1.8; }
.novel-meta-section { background: oklch(var(--b2)); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: var(--radius); margin-bottom: 1rem; }
.novel-meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.75rem; }
.novel-meta-item { display: flex; flex-direction: column; gap: 0.25rem; }
.meta-label { font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; }
.meta-value { font-size: 0.875rem; opacity: 0.8; }
.meta-value.completed { color: oklch(var(--su)); }
.meta-value.ongoing { color: oklch(var(--wa)); }
.quick-stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
.quick-stat { display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
.quick-stat-title { font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; }
.quick-stat-value { font-size: 2.25rem; font-weight: 700; line-height: 1.1; }
.char-names-section { background: oklch(var(--b2)); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: var(--radius); margin-bottom: 1rem; }
.char-names-grid { display: flex; flex-direction: column; gap: 1rem; margin-top: 0.75rem; }
.char-names-group { display: flex; flex-direction: column; gap: 0.5rem; }
.char-group-label { font-size: 0.8125rem; opacity: 0.5; font-weight: 500; }
.char-names-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.char-name-tag { padding: 0.375rem 0.875rem; border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; background: oklch(var(--b3)); color: oklch(var(--bc)); transition: opacity 0.2s; }
.char-name-tag:hover { opacity: 0.8; }
.char-name-tag.male { background: oklch(var(--in) / 0.15); color: oklch(var(--in)); }
.char-name-tag.female { background: oklch(var(--er) / 0.15); color: oklch(var(--er)); }
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px; opacity: 0.5; }
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-text { font-size: 1rem; }
@media (max-width: 768px) {
    .char-grid { grid-template-columns: 1fr; }
    .novel-meta-grid { grid-template-columns: 1fr; }
    .char-names-grid { grid-template-columns: 1fr; }
    .quick-stats-grid { grid-template-columns: repeat(2, 1fr); }
    .quick-stat-value { font-size: 1.75rem; }
	}
    `;

    const quickStatsHtml = buildQuickStatsHtml({ sexCount, relationshipCount, characterCount, wordCountText });

    const summaryHtml = buildRelationshipSummaryHtml(analysis);
    const charactersHtml = buildCharactersHtml(analysis);
    const relationshipSvgHtml = buildRelationshipSvgHtml(analysis, { width: 1200, height: 800, isDark: theme === 'dark' });
    const firstSexSceneHtml = buildFirstSexSceneHtml(analysis);
    const sexSceneCountHtml = buildSexSceneCountHtml(analysis);
    const relationshipProgressHtml = buildRelationshipProgressHtml(analysis);

    const html = `<!DOCTYPE html>
<html lang="zh-CN" data-theme="${theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${safeNovelName} - 分析报告</title>
    <link href="https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
    <style>
        ${styleCss}
        .export-report #relationshipChart { height: 600px; overflow: hidden; position: relative; }
        .export-report #relationshipChart svg { width: 100%; height: 100%; }
    </style>
</head>
<body class="min-h-screen bg-base-100 export-report" x-data="{
    currentTab: 'summary',
    tabs: [
        { id: 'summary', name: '总结' },
        { id: 'characters', name: '主角' },
        { id: 'relationships', name: '关系图' },
        { id: 'firstsex', name: '首次' },
        { id: 'count', name: '统计' },
        { id: 'progress', name: '发展' }
    ]
}">
    <main class="min-h-screen py-8">
        <div class="max-w-6xl mx-auto px-6">
            <div class="mb-6">
                <div class="text-2xl font-semibold text-primary">${safeNovelName}</div>
                <div class="text-xs text-base-content/50 mt-1">导出时间：${generatedAt}</div>
            </div>

            <div class="border-b border-base-300 mb-6">
                <div class="flex gap-1">
                    <template x-for="tab in tabs" :key="tab.id">
                        <button class="px-4 py-2 text-sm font-medium transition-colors relative"
                            :class="currentTab === tab.id ? 'text-primary' : 'text-base-content/60 hover:text-base-content'"
                            @click="currentTab = tab.id">
                            <span x-text="tab.name"></span>
                            <span class="absolute bottom-0 left-0 right-0 h-0.5 bg-primary transition-transform origin-left"
                                :class="currentTab === tab.id ? 'scale-x-100' : 'scale-x-0'"></span>
                        </button>
                    </template>
                </div>
            </div>

            <div>
                <div x-show="currentTab === 'summary'" x-cloak>
                    <div id="quickStats">${quickStatsHtml}</div>
                    <div id="relationshipSummary">${summaryHtml}</div>
                </div>
                <div x-show="currentTab === 'characters'" x-cloak>
                    <div id="mainCharacters">${charactersHtml}</div>
                </div>
                <div x-show="currentTab === 'relationships'" x-cloak>
                    <div class="bg-base-200 border border-base-300 rounded-lg h-[600px]" id="relationshipChart">${relationshipSvgHtml}</div>
                </div>
                <div x-show="currentTab === 'firstsex'" x-cloak>
                    <div id="firstSexScene">${firstSexSceneHtml}</div>
                </div>
                <div x-show="currentTab === 'count'" x-cloak>
                    <div id="sexSceneCount">${sexSceneCountHtml}</div>
                </div>
                <div x-show="currentTab === 'progress'" x-cloak>
                    <div id="relationshipProgress">${relationshipProgressHtml}</div>
                </div>
            </div>
        </div>
    </main>
</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = sanitizeFilename(novelName).replace(/\.txt$/i, '') + '_分析报告.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
