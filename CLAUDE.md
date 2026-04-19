# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 快速导航 (Quick Navigation)

- **详细架构**: [ARCHITECTURE.md](ARCHITECTURE.md) - 完整的模块列表和设计决策
- **开发指南**: [DEVELOPMENT.md](DEVELOPMENT.md) - 环境设置、验证流程、一致性清单
- **测试基准**: [tests/BENCHMARKS.md](tests/BENCHMARKS.md) - 真实场景测试套件
- **按需技能**: 使用 `/development`、`/testing`、`/deployment` 加载详细指南

## 项目概述 (Project Overview)

GRAG (Graph Retrieval Augmented Generation，图谱检索增强生成) 是一个基于知识图谱增强检索生成的技术文档智能问答系统。它基于 LightRAG 流程设计，实现低成本、高性能的文档问答能力，通过混合检索机制（global + local + BM25）提升检索效果。

**核心架构哲学**: "Pipeline for Write, Service for Read" - 两个解耦子系统：
- **知识构建管道**: 文件驱动、慢速、可调试 → GitHub API → artifacts/01_raw → 清洗 → artifacts/02_clean → LightRAG 摄取 → 存储
- **在线问答服务**: 内存驱动、快速、只读 → 用户查询 → FastAPI/Streamlit → LangGraph Agent → Model → Knowledge Storage

**关键特性**:
- 📊 混合检索: 向量搜索 + BM25 关键词搜索，使用 RRF 融合
- 🎯 引用系统: 自动来源归因，[1], [2] 格式
- 📈 性能监控: 内置指标收集和性能追踪
- 🔧 多部署模式: 本地 (JSON) 和 Docker (PostgreSQL/Neo4j) 模式
- 🎨 多页面 UI: Chat、Dashboard、Graph visualization、Pipeline management

## 核心命令 (Essential Commands)

### 包管理 (uses `uv`)
```bash
uv sync  # 安装依赖
```

### 管道脚本 (Pipeline Scripts)
```bash
# 数据摄取管道
uv run scripts/pipeline_fetch.py --repo <github-url> --output artifacts/01_raw/data.json
uv run scripts/pipeline_clean.py --input artifacts/01_raw/data.json --output artifacts/02_clean/data.json
uv run scripts/pipeline_ingest.py --input artifacts/02_clean/data.json
```

### 开发环境 (Development)
```bash
# 后端 (FastAPI)
uvicorn api.main:app --reload

# 前端 (Streamlit)
streamlit run frontend/app.py

# 一致性检查
uv run scripts/check_consistency.py --quick
```

### Docker 部署 (Docker Deployment)
```bash
docker-compose up -d    # 启动所有服务
docker-compose logs -f  # 查看日志
docker-compose down     # 停止服务
```

### 测试 (Testing)
```bash
# 基准测试
uv run tests/run_benchmark_tests.py --full

# 系统诊断
uv run scripts/test_diagnostics.py
```

## 系统架构 (System Architecture)

### 初始化顺序 (Initialization Order)
1. **StorageFactory** → 2. **ModelManager** → 3. **LightRAGWrapper** → 4. **GRAGAgent** → 5. **CacheManager**

### 架构概览
```
前端层 (Frontend) → API层 (API) → Agent层 (Agent) → 知识层 (Knowledge) → 存储层 (Storage) → 模型层 (Model)
```

**详细模块说明**: 见 [ARCHITECTURE.md](ARCHITECTURE.md)

## 核心开发原则 (Critical Development Principles)

1. **文件优先开发**: 管道步骤使用文件存储便于调试
   - `artifacts/01_raw/`: 原始数据
   - `artifacts/02_clean/`: 清洗数据
   - `artifacts/03_graph/`: 图谱数据

2. **幂等性**: 使用确定性 ID (文件路径 MD5)
   - `doc_id = MD5(source_url:path)` 确保跨仓库唯一性

3. **零爬虫策略**: 仅使用 GitHub API 作为数据源

4. **Pydantic 作为协议**: 所有数据契约都是 Pydantic 模型
   - 详见 [utils/schema.py](utils/schema.py)

5. **始终使用 `uv run`**: 执行任何管道脚本

## 数据流 (Data Flow)

### 查询请求流
```
用户查询 → FastAPI/Streamlit → 缓存检查 → LangGraph Agent → LightRAG 检索 → 模型 → 引用处理 → 响应生成 → 缓存存储 → 用户响应
```

### 知识构建流
```
GitHub 仓库 → GitHub API → 原始文档 → 清洗管道 → Context7 增强(可选) → LightRAG 摄取 → 存储
```

## 配置管理 (Configuration Management)

### 关键配置文件
- **`.env`**: 敏感配置 (API 密钥、密码)
- **`config/config.yaml`**: 非敏感配置，支持 `${VAR:default}` 语法
- **`config/presets/`**: 本地和 Docker 预设

### 存储模式切换 (Storage Mode Switching)
```bash
# 本地文件模式 (默认)
export STORAGE_MODE=json
export LIGHTRAG_STORAGE_TYPE=file

# PostgreSQL 模式
export STORAGE_MODE=postgresql
export LIGHTRAG_STORAGE_TYPE=postgresql
```

**完整配置说明**: 见 [config/config.yaml](config/config.yaml)

## 开发指南 (Development Guide)

### 一致性防呆清单
> 详细内容见 [DEVELOPMENT.md](DEVELOPMENT.md) - "一致性防呆清单"章节

**快速检查** (发版前必做):
```bash
uv run scripts/check_consistency.py          # 完整检查
uv run scripts/check_consistency.py --quick # 快速检查
```

**关键检查项**:
- ✅ 索引与管道同步
- ✅ 配置源单一性 (.env vs config.yaml)
- ✅ API 响应契约完整
- ✅ 缓存数据新鲜度

**常见问题**:
- 检索无结果 → 检查 `LIGHTRAG_WORKING_DIR` 与实际 ingest 路径
- 答案过期 → `uv run scripts/clear_cache.py`
- 引用错乱 → 检查 `context_metadata` 与答案一致性

### 开发工作流
1. 知识构建: 使用管道脚本 (fetch → clean → ingest)
2. 本地开发: 文件模式 + `uv run` 执行脚本
3. 测试验证: 使用基准测试套件
4. 发版前检查: 运行一致性检查

**详细开发指南**: 见 [DEVELOPMENT.md](DEVELOPMENT.md)

## 数据模式 (Data Schemas)

所有中间数据产品必须是 Pydantic 模型，详见 [utils/schema.py](utils/schema.py)

### 主要模型类别
- **管道模型**: `RawDoc`, `IngestionBatch`, `CleanDoc`, `CleanBatch`
- **API 模型**: `QueryRequest`, `QueryResponse`, `FeedbackRequest`
- **Agent 模型**: `AgentState`, `IntentType`
- **缓存模型**: `QueryCacheEntry`, `CacheStats`

## 已知限制与模型支持 (Known Limitations & Model Support)

### 已知限制
1. Ollama 本地模型已弃用 (使用 DeepSeek/Qwen API)
2. 仅支持 GitHub 仓库作为数据源
3. 实体/关系提取提示词可能需要优化

### 模型支持
- **DeepSeek API**: `deepseek-chat`, `deepseek-coder`
- **Qwen API**: `qwen-plus`, `qwen-turbo`, `qwen-max`

## TODO 优先级 (TODO Priorities)

### 阶段 1: 基础 (已完成)
- ✅ 完整管道测试
- ✅ LightRAG 验证
- ✅ 多模型支持
- ✅ 引用系统实现
- ✅ 基准测试框架

### 阶段 2: 优化 (进行中)
- 🔄 重构 Agent 层
- 🔄 BM25 性能调优
- 🔄 Context7 集成增强

### 阶段 3-4: 集成与生产 (未来)
- ⏳ 高级分析仪表板
- ⏳ 多仓库支持
- ⏳ API 驱动管道转换
