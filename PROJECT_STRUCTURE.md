# 项目结构说明

## 目录结构

```
GRAG/
├── agent/                    # Agent 层
│   ├── __init__.py
│   └── grag_agent.py        # LangGraph Agent 实现
│
├── api/                      # API 层
│   ├── __init__.py
│   ├── main.py              # FastAPI 主应用
│   ├── routes.py            # RESTful API 路由
│   └── auth.py               # 认证模块
│
├── config/                   # 配置管理
│   ├── __init__.py
│   ├── config.yaml          # 配置文件（包含 TODO 标记的 prompt）
│   └── config_manager.py    # 配置管理器
│
├── crawler/                  # 数据采集层
│   ├── __init__.py
│   └── scraper.py           # Scrapy + Playwright 爬虫
│
├── frontend/                 # 前端应用
│   ├── __init__.py
│   └── app.py               # Streamlit 前端
│
├── knowledge/                # 知识存储层
│   ├── __init__.py
│   └── lightrag_wrapper.py  # LightRAG 封装
│
├── models/                   # 模型层
│   ├── __init__.py
│   └── model_manager.py     # 模型管理器（动态切换）
│
├── storage/                  # 存储层
│   ├── __init__.py
│   └── cache_manager.py     # PostgreSQL 缓存管理
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── logger.py            # 日志配置
│   └── monitoring.py        # 性能监控
│
├── logs/                     # 日志目录（自动创建）
│
├── docker-compose.yml       # Docker Compose 配置
├── Dockerfile               # 后端 Docker 镜像
├── Dockerfile.frontend      # 前端 Docker 镜像
├── requirements.txt         # Python 依赖
├── env.example              # 环境变量模板
├── .gitignore              # Git 忽略文件
├── README.md               # 项目说明
└── PROJECT_STRUCTURE.md    # 本文件
```

## 核心模块说明

### 1. Agent 层 (`agent/`)

- **grag_agent.py**: 使用 LangGraph 实现的 Agent
  - 意图识别：分析用户查询意图
  - 工具调用：调用爬虫、数据预处理等工具
  - 检索：使用 LightRAG 进行双层检索
  - 答案生成：基于检索结果生成答案

### 2. API 层 (`api/`)

- **main.py**: FastAPI 应用入口，初始化所有组件
- **routes.py**: RESTful API 路由
  - `POST /api/v1/query`: 问答查询
  - `POST /api/v1/query/stream`: 流式查询
  - `POST /api/v1/feedback`: 用户反馈
  - `POST /api/v1/model/switch`: 模型切换
  - `GET /api/v1/stats`: 系统统计
- **auth.py**: HTTP Basic 认证

### 3. 模型层 (`models/`)

- **model_manager.py**: 模型管理器
  - 支持 DeepSeek API 和本地模型（Ollama/vLLM）
  - 动态切换，优先使用本地模型
  - 自动回退机制

### 4. 知识存储层 (`knowledge/`)

- **lightrag_wrapper.py**: LightRAG 封装
  - 直接使用 LightRAG API
  - 自定义实体/关系抽取 prompt（TODO: 需要根据实际需求调整）
  - 双层检索（全局+局部）

### 5. 存储层 (`storage/`)

- **cache_manager.py**: PostgreSQL 缓存管理
  - 查询缓存表
  - LRU 清理策略（定时任务+容量限制）
  - 质量评分机制（基于用户反馈）

### 6. 数据采集层 (`crawler/`)

- **scraper.py**: 文档爬虫
  - Playwright 处理 JavaScript 渲染页面
  - 遵守 robots.txt
  - 控制爬取频率

### 7. 前端 (`frontend/`)

- **app.py**: Streamlit 前端界面
  - 聊天界面
  - 模型切换
  - 用户反馈
  - 系统统计

## 配置说明

### 配置文件 (`config/config.yaml`)

- **数据库配置**: PostgreSQL 连接信息
- **LightRAG 配置**: 
  - 实体抽取 prompt（TODO: 需要根据实际 LightRAG API 调整）
  - 关系抽取 prompt（TODO: 需要根据实际 LightRAG API 调整）
- **模型配置**: DeepSeek API 和本地模型配置
- **缓存配置**: LRU 清理策略和质量评分阈值
- **日志配置**: 日志级别和文件路径

### 环境变量 (`env.example`)

- `POSTGRES_*`: 数据库配置
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `API_USERNAME` / `API_PASSWORD`: API 认证信息
- `LOCAL_MODEL_URL` / `LOCAL_MODEL_NAME`: 本地模型配置

## TODO 清单

### 高优先级

1. **LightRAG Prompt 调整** (`config/config.yaml`)
   - 实体抽取 prompt 需要根据实际 LightRAG API 格式调整
   - 关系抽取 prompt 需要根据实际 LightRAG API 格式调整
   - 位置：`config/config.yaml` 第 22-69 行

2. **流式响应实现** (`api/routes.py`)
   - 当前流式响应为模拟实现
   - 需要集成模型的真实流式接口
   - 位置：`api/routes.py` 第 186-235 行

3. **Agent 意图识别** (`agent/grag_agent.py`)
   - 当前意图识别为简化实现
   - 需要完善 JSON 解析和意图分类
   - 位置：`agent/grag_agent.py` 第 111-166 行

4. **爬虫工具集成** (`agent/grag_agent.py`)
   - 需要实际调用爬虫模块
   - 位置：`agent/grag_agent.py` 第 188-211 行

### 中优先级

5. **LightRAG API 适配** (`knowledge/lightrag_wrapper.py`)
   - 需要根据实际 LightRAG API 调整初始化参数
   - 位置：`knowledge/lightrag_wrapper.py` 多处 TODO 标记

6. **Scrapy 集成** (`crawler/scraper.py`)
   - 当前只有 Playwright 实现
   - 需要集成 Scrapy 框架
   - 位置：`crawler/scraper.py` 第 185-205 行

## 部署说明

### Docker Compose 部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api
docker-compose logs -f frontend

# 停止服务
docker-compose down
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium

# 启动后端
uvicorn api.main:app --reload

# 启动前端（新终端）
streamlit run frontend/app.py
```

## 开发注意事项

1. **环境变量**: 确保 `.env` 文件已配置，敏感信息不要提交到 Git
2. **数据库初始化**: 首次运行会自动创建缓存表，无需手动初始化
3. **日志目录**: `logs/` 目录会自动创建，确保有写入权限
4. **模型切换**: 本地模型需要先启动 Ollama 或 vLLM 服务
5. **LightRAG**: 需要根据实际 LightRAG 版本调整 API 调用方式

## 测试建议

1. **单元测试**: 为各模块编写单元测试
2. **集成测试**: 测试 API 端到端流程
3. **性能测试**: 测试响应时间和并发能力
4. **缓存测试**: 测试缓存命中率和质量评分机制

