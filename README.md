# 小说分析器

基于LLM的小说分析工具 - 剧情高潮、人物关系、情感曲线可视化

## 功能特性

- 自动扫描本地小说文件夹
- 支持嵌套子文件夹
- LLM智能分析：人物关系、剧情高潮、情感曲线
- 可视化展示：力导向关系图、情感曲线图、高潮时间线
- 支持配置任意OpenAI兼容API

## 使用方法

### 1. 安装依赖

```bash
# （可选）创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置（仅 `.env`）

为安全起见：网页端不提供任何配置入口，所有配置都从服务端 `.env` 读取（`.env` 不会进 Git）。

从 `.env.example` 复制一份为 `.env`，再按需修改：

```env
NOVEL_PATH=你的小说根目录
API_BASE_URL=https://your-api.com/v1
API_KEY=sk-your-api-key
MODEL_NAME=gpt-4o
HOST=127.0.0.1
PORT=6103
DEBUG=false
```

### 3. 启动服务

**Windows (使用提供的脚本):**

```bash
start.bat
```

`start.bat` 会优先使用 `venv\\Scripts\\python.exe`（存在则用），并从 `.env` 读取 `HOST/PORT/NOVEL_PATH` 等配置。
（准确说：`backend.py` 会自动读取 `.env`，脚本本身不解析 `.env`，避免覆盖配置。）

**手动启动:**

```bash
python backend.py
```

### 4. 使用

1. 打开浏览器访问 `http://127.0.0.1:6103`
2. 选择左侧小说文件
3. 点击"开始分析"
4. （可选）右上角“配置（只读）”里点击“测试连接”
5. 查看分析结果，支持多个Tab切换

## 配置说明

| 配置项       | 说明               | 示例                      |
| ------------ | ------------------ | ------------------------- |
| NOVEL_PATH   | 小说根目录         | X:\\Gallery\\h小说         |
| API_BASE_URL | OpenAI兼容API地址   | https://api.example.com/v1 |
| API_KEY      | API密钥             | sk-xxx                    |
| MODEL_NAME   | 模型名称            | gpt-4o                    |
| HOST         | 监听地址(默认本机)  | 127.0.0.1                 |
| PORT         | 端口                | 6103                      |
| DEBUG        | 显示LLM原始响应错误 | false                     |

## 安全提示

- 默认只监听 `127.0.0.1`；除非你知道风险，否则不要把 `HOST` 改成 `0.0.0.0`（会把本机文件列表/内容暴露给局域网）。
- API Key 仅存在于服务端 `.env`，网页不会读取/保存/发送 API Key。

## 目录结构

```
novel-analyzer/
├── backend.py          # FastAPI后端
├── .env                # API配置
├── .env.example        # 配置模板
├── requirements.txt    # 依赖列表
├── README.md           # 说明文档
├── templates/
│   └── index.html      # 前端页面
├── static/
│   ├── style.css       # 样式
│   └── chart-view.js   # 可视化逻辑
└── start.bat           # 启动脚本
```

## 依赖

- Python 3.8+
- fastapi
- uvicorn
- python-multipart
- jinja2
- python-dotenv
- requests
