# GRAG 技术文档智能问答系统

基于知识图谱增强检索生成（GraphRAG）的技术文档智能问答系统，支持技术文档的智能问答、引用追踪和图谱可视化。

## 项目概述

本项目采用 LightRAG 框架实现低成本、高性能的文档问答能力，通过多层检索机制（全局+局部+BM25混合检索）提升检索效果，支持多种模型和存储后端的灵活切换。

## 核心特性

**检索能力**
- 双层检索机制：全局检索（实体/关系）+ 局部检索（文档块）
- BM25关键词检索：支持中文分词，与向量检索融合
- RRF融合算法：智能合并多种检索结果

**引用系统**
- 三层容错机制：Prompt引导 → 验证检查 → 自动修复
- 纯数字引用格式：[1], [2], [3] 简洁明了
- 完整元数据追溯：文档来源、chunk_id、检索方法

**存储方案**
- 存储后端支持多种选择，包括 PostgreSQL、Neo4j，亦可选用纯 JSON 文件。本地实验或小规模场景无需外部数据库，仅需使用 JSON 文件模式即可；大规模生产环境则建议接入数据库后端，满足性能和扩展性要求。**目前版本尚不保证高并发环境性能**
- 支持 Docker Compose 一键部署，灵活适配不同运行环境。

**知识构建**
- Zero-Crawler策略：仅使用GitHub API，避免爬虫复杂性
- 文件驱动管道：Fetch → Clean → Ingest 三步流程
- Pydantic数据契约：确保数据质量和可追溯性

**测试评估**
- 真实场景基准测试：27个场景覆盖部署/API/故障排查
- 100分质量评估：全面评价答案质量
- 回归测试框架：确保版本迭代不破坏已有功能

## 系统架构

### 核心架构哲学：Pipeline for Write, Service for Read

系统采用"写操作走管道，读操作走服务"的架构设计，明确拆分为两个解耦的子系统。

#### 子系统 A：知识构建管道（文件驱动，慢速，重 Debug）

```
GitHub API
    ↓
artifacts/01_raw/          # Raw Artifact（不可变源数据）
    ↓
清洗逻辑
    ↓
artifacts/02_clean/        # Clean Artifact（LightRAG 的黄金输入）
    ↓
LightRAG 实体抽取
    ↓
存储索引（PostgreSQL/Neo4j/JSON）
```

**特点**：
- 文件驱动，支持断点调试和人工介入
- 使用 Pydantic Schema 作为数据契约
- 开发阶段通过脚本串联，生产阶段可切换为 API

#### 子系统 B：在线问答服务（内存驱动，快速，只读）

```
用户查询
    ↓
应用层 (FastAPI + Streamlit)
    ↓
Agent层 (LangGraph) - 检索与答案生成
    ↓
模型层 (DeepSeek API / 本地模型)
    ↓
知识存储层 (LightRAG + 存储后端) - 只读查询
```

**特点**：
- 内存驱动，响应快速
- 基于已构建的索引回答问题
- 支持引用生成和质量评分

## 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **Agent 框架**：LangGraph
- **知识图谱**：LightRAG
- **检索增强**：BM25 (rank-bm25) + RRF融合
- **数据库**：PostgreSQL / Neo4j / JSON文件
- **模型**：DeepSeek API / Ollama / vLLM
- **数据采集**：GitHub API
- **部署**：Docker Compose
- **包管理**：uv

## 快速开始

### 1. 环境准备

- Python 3.11+
- Docker & Docker Compose（推荐）
- PostgreSQL 15+ / Neo4j（如果不用 Docker）

### 2. 克隆项目

```bash
git clone https://github.com/akaliyas/GRAG.git
cd GRAG
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
# 数据库配置（可选，JSON模式无需配置）
POSTGRES_PASSWORD=your_password
POSTGRES_DB=grag_db
POSTGRES_USER=grag_user

# API配置
DEEPSEEK_API_KEY=your_api_key
API_PASSWORD=your_api_password

# 存储模式（可选，默认postgresql）
STORAGE_MODE=json  # 可选：postgresql, neo4j, json
```

### 4. 安装依赖

**推荐使用 uv**：

```bash
# 安装 uv
pip install uv

# 同步依赖
uv sync
```

### 5. 启动服务

**方式一：Docker Compose（推荐）**

```bash
docker-compose up -d
```

**方式二：本地开发**

```bash
# 启动后端 API
uvicorn api.main:app --reload

# 启动前端（新终端）
streamlit run frontend/app.py
```

**方式三：纯JSON模式（无数据库）**

```bash
# 设置环境变量
export STORAGE_MODE=json
export LIGHTRAG_STORAGE_TYPE=file
export LIGHTRAG_GRAPH_STORAGE=NetworkXStorage

# 启动服务
uvicorn api.main:app --reload
```

### 6. 访问服务

- **API 文档**：http://localhost:8000/docs
- **前端界面**：http://localhost:8501
- **健康检查**：http://localhost:8000/health

## 使用说明

### 数据摄取管道

使用管道脚本摄取GitHub仓库文档：

```bash
# Step 1: 获取原始数据
uv run scripts/pipeline_fetch.py \
  --repo https://github.com/openai/openai-cookbook \
  --output artifacts/01_raw/openai_cookbook_raw.json

# Step 2: 清洗数据
uv run scripts/pipeline_clean.py \
  --input artifacts/01_raw/openai_cookbook_raw.json \
  --output artifacts/02_clean/openai_cookbook_clean.json

# Step 3: 导入知识库
uv run scripts/pipeline_ingest.py \
  --input artifacts/02_clean/openai_cookbook_clean.json
```

**JSON模式快速启动**：

```bash
# Windows
uv run scripts\ingest_with_json_mode.py

# Linux/macOS
STORAGE_MODE=json uv run scripts/pipeline_ingest.py --input artifacts/02_clean/data.json
```

### API 使用示例

**1. 问答查询**

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用 OpenAI Python SDK？",
    "mode": "hybrid_bm25",
    "use_cache": true
  }'
```

**2. 流式查询**

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置环境变量？",
    "stream": true
  }'
```

**3. 图谱查询**

```bash
curl -X POST "http://localhost:8000/api/v1/graph/query" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OpenAI API"
  }'
```

### 基准测试

运行真实场景基准测试：

```bash
# 运行OpenAI部署场景测试
uv run tests/run_openai_benchmark.py

# 运行完整基准测试套件
uv run tests/run_benchmark_tests.py --full

# 运行特定类别测试
uv run tests/run_benchmark_tests.py --category deployment
```

测试报告将保存到 `artifacts/test_results/benchmarks/`。

## 配置说明

主要配置文件：`config/config.yaml`

### 关键配置项

**存储配置**
```yaml
database:
  postgresql:
    enabled: true
  neo4j:
    enabled: false
  json:
    enabled: true  # 纯文件模式
```

**LightRAG配置**
```yaml
lightrag:
  chunk_size: 1200
  chunk_overlap: 100
  retrieval_mode: "hybrid"  # local, global, hybrid, hybrid_bm25
```

**BM25配置**
```yaml
lightrag:
  bm25:
    enabled: true
    k1: 1.5
    b: 0.75
    rrf_k: 60
```

**引用系统配置**
```yaml
citation:
  format_type: "number"  # number, file, mixed
  enable_fix: true  # 自动修复缺失引用
```

## 文档索引

### 项目文档

- **[基准测试指南](tests/BENCHMARKS.md)** - 真实场景测试框架说明
- **[CLAUDE.md](CLAUDE.md)** - Claude Code 协作开发指南

### 技术文档

- **系统架构**：参见本文档"系统架构"部分
- **API文档**：启动服务后访问 http://localhost:8000/docs
- **数据管道**：参见 scripts/ 目录下的脚本注释

## 开发指南

### 代码规范

- Python 3.11+
- 遵循 PEP 8
- 类型注解必需
- Google风格docstring

### 命名约定

- 模块名：小写 + 下划线（如 `grag_agent.py`）
- 类名：大驼峰（如 `GRAGAgent`）
- 函数/变量：小写 + 下划线（如 `process_query`）
- 常量：全大写 + 下划线（如 `MAX_RETRIES`）

### 核心模块

**API层** (`api/`)
- `main.py`: FastAPI应用入口
- `routes.py`: RESTful API路由
- `auth.py`: HTTP Basic认证

**Agent层** (`agent/`)
- `grag_agent.py`: LangGraph Agent实现
- 支持引用生成和质量评分

**知识层** (`knowledge/`)
- `lightrag_wrapper.py`: LightRAG封装
- `bm25_indexer.py`: BM25索引和搜索
- 支持多种存储后端

**模型层** (`models/`)
- `model_manager.py`: 模型管理和切换
- 支持DeepSeek API和本地模型

**存储层** (`storage/`)
- `cache_manager.py`: PostgreSQL缓存
- `json_knowledge_storage.py`: JSON文件存储
- `json_graph_storage.py`: JSON图谱存储

**工具层** (`utils/`)
- `citation.py`: 引用格式化和验证
- `schema.py`: Pydantic数据模型

### 测试

基准测试位于 `tests/benchmarks/`：

- `benchmark_base.py`: 基准测试基类
- `test_deployment.py`: 部署配置场景测试
- `test_api_usage.py`: API使用场景测试
- `test_troubleshooting.py`: 故障排查场景测试
- `test_openai_deployment.py`: OpenAI部署场景测试

## 已知限制

1. 流式响应实现需要完善
2. LightRAG实体抽取prompt需要根据实际需求调优
3. 仅支持GitHub仓库作为数据源
4. 本地模型（Ollama）配置中暂时禁用

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

项目负责人：akaliyas
