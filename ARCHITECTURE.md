# GRAG 系统架构

本文档提供 GRAG 系统的详细架构说明，包括各层设计、模块职责和关键决策。

## 架构概览

### 核心架构哲学

**"Pipeline for Write, Service for Read"** - 两个完全解耦的子系统：

1. **知识构建管道 (Knowledge Construction Pipeline)**
   - 文件驱动、慢速、可调试
   - 数据流向: GitHub API → artifacts/01_raw → 清洗 → artifacts/02_clean → LightRAG 摄取 → 存储
   - 支持断点调试和人工介入
   - 使用 Pydantic Schema 作为数据契约

2. **在线问答服务 (Online Q&A Service)**
   - 内存驱动、快速、只读
   - 数据流向: 用户查询 → FastAPI/Streamlit → LangGraph Agent → Model → Knowledge Storage
   - 基于已构建的索引回答问题
   - 支持引用生成和质量评分

### 系统分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端层 (Frontend Layer)                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Chat    │  │ Dashboard │  │  Graph   │  │ Pipeline │   │
│  │  Page    │  │   Page    │  │  Page    │  │  Page    │   │
│  └──────────┘  └───────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API 层 (API Layer)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI Routes: /query, /feedback, /model/switch   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent 层 (Agent Layer)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangGraph Agent: Intent → Retrieve → Generate       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  知识层 (Knowledge Layer)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │  LightRAG       │  │  BM25 Indexer   │  │  Citation  │ │
│  │  Wrapper        │  │  (Optional)     │  │  System    │ │
│  └─────────────────┘  └─────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 存储层 (Storage Layer)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  StorageFactory → Cache │ Graph │ Knowledge Storage  │   │
│  │  (JSON/PostgreSQL/Neo4j)                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 模型层 (Model Layer)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Manager: DeepSeek API (Ollama deprecated)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 初始化顺序 (Initialization Order)

系统启动时的组件初始化顺序 (api/main.py:100):

1. **StorageFactory** → 创建存储实例
2. **ModelManager** → 初始化模型客户端
3. **LightRAGWrapper** → 初始化 LightRAG
4. **GRAGAgent** → 初始化 Agent
5. **CacheManager** → 初始化缓存

## 核心模块详解

### 前端层 (Frontend Layer) [`frontend/`](frontend/)

#### 主要组件
- **app.py**: Streamlit 多页应用入口
- **pages/chat.py**: 问答聊天界面
  - 流式响应显示
  - 会话历史管理
  - 反馈收集
- **pages/dashboard.py**: 系统指标和统计
  - 系统健康监控
  - 性能指标展示
  - 缓存管理界面
- **pages/graph.py**: 知识图谱可视化
  - 实体关系图展示
  - 图谱统计数据
  - 图谱导出功能
- **pages/pipeline.py**: 管道管理界面
  - 数据摄取状态
  - 管道步骤监控

#### 配置
- **frontend/config.py**: 前端配置中心
  - API_BASE_URL: 动态构建
  - API_USERNAME/PASSWORD: 认证配置

### API 层 (API Layer) [`api/`](api/)

#### 主要组件
- **main.py**: FastAPI 应用入口
  - 系统启动/关闭生命周期
  - 依赖注入设置
  - CORS 中间件配置
- **routes.py**: RESTful API 端点
  - `POST /api/v1/query`: 标准查询
  - `POST /api/v1/query/stream`: 流式查询 (SSE)
  - `POST /api/v1/feedback`: 用户反馈
  - `POST /api/v1/model/switch`: 模型切换
  - `GET /api/v1/graph/*`: 图谱统计数据
  - `GET /api/v1/knowledge/*`: 知识库统计
- **auth.py**: HTTP Basic 认证
  - 用户名密码验证
  - 权限控制

#### 响应模型
- **QueryResponse**: 统一的查询响应格式
  - success: 成功状态
  - answer: 生成的答案
  - context_metadata: 带引用的完整元数据
  - citations: 引用列表
  - response_time: 响应时间
  - from_cache: 缓存命中标识

### Agent 层 (Agent Layer) [`agent/`](agent/)

#### 主要组件
- **grag_agent.py**: 基于 LangGraph 的 Agent
  - **重要**: Agent 仅处理检索和答案生成
  - 数据摄取在离线管道脚本中进行
  - 意图类型: 仅 `QUERY` 有效
  - 已弃用意图: `GITHUB_INGEST`, `PREPROCESS` (死代码路径)
  - 支持真正的 token 级别流式响应

#### Agent 状态 (AgentState)
- messages: 消息历史
- intent: 识别的意图
- query: 查询文本
- context_ids: 检索的上下文 ID
- context_metadata: 上下文元数据
- answer: 生成的答案
- citations: 引用
- error: 错误信息

### 知识层 (Knowledge Layer) [`knowledge/`](knowledge/)

#### 主要组件
- **lightrag_wrapper.py**: LightRAG 框架封装
  - 支持 PostgreSQL/Neo4j/JSON 存储
  - 混合检索: 全局 (实体/关系) + 局部 (文档块) + BM25
  - 自定义实体/关系提取提示词
  - Jupyter notebook 支持 (使用 nest_asyncio)
  - 存储后端可插拔设计

- **bm25_indexer.py**: BM25 关键词搜索
  - RRF (Reciprocal Rank Fusion) 混合结果融合
  - 可配置 k1, b, epsilon 参数
  - 支持中文分词
  - 索引持久化

#### 检索策略
- **全局检索**: 实体和关系搜索
- **局部检索**: 文档块搜索
- **BM25 检索**: 关键词搜索
- **混合检索**: RRF 融合多种检索结果

### 模型层 (Model Layer) [`models/`](models/))

#### 主要组件
- **model_manager.py**: 模型管理器
  - **DeepSeek API** (主要模型)
    - `deepseek-chat`: 通用问答
    - `deepseek-coder`: 代码特定任务
  - **Qwen API** (阿里云 DashScope)
    - `qwen-plus`: 平衡性能
    - `qwen-turbo`: 快速响应
    - `qwen-max`: 高质量
  - **Ollama 本地模型** (已弃用)
    - 代码保留但配置中已禁用
    - 建议使用 API 模型以获得更好性能

#### 模型切换策略
- 优先级: qwen (默认) / ds
- 健康检查和回退机制
- 自动消息格式转换

### 存储层 (Storage Layer) [`storage/`](storage/)

#### 主要组件
- **factory.py**: 存储工厂模式
  - 基于配置创建存储实例
  - 支持预设: `auto`, `local`, `docker`
  - 自动检测环境选择合适的存储后端

- **interface.py**: 存储接口定义
  - `ICacheStorage`: 缓存存储接口
  - `IGraphStorage`: 图存储接口
  - `IKnowledgeStorage`: 知识存储接口

- **cache_manager.py**: 缓存管理
  - LRU 清理策略
  - 质量评分机制
  - PostgreSQL/JSON 后端支持
  - 用户反馈集成

#### 存储实现
- **JSON 存储** (本地模式):
  - `json_cache_storage.py`: JSON 文件缓存
  - `json_graph_storage.py`: JSON 图谱存储
  - `json_knowledge_storage.py`: JSON 知识存储

- **PostgreSQL 存储** (Docker 模式):
  - 支持向量、图、KV 存储
  - 连接池管理
  - 事务支持

- **Neo4j 存储** (可选):
  - 图数据库专用存储
  - 支持复杂图查询

## 管道脚本 (Pipeline Scripts) [`scripts/`](scripts/)

### 数据管道 (Data Pipeline)
- **pipeline_fetch.py**: 从 GitHub 提取文档
  - 零爬虫策略，仅使用 GitHub API
  - 支持 Markdown 和 Jupyter Notebook
  - 提取 frontmatter 和元数据
  
- **pipeline_clean.py**: 清洗和预处理
  - 移除 HTML 标签
  - 清洗 Jupyter notebooks
  - 修复相对链接
  
- **pipeline_context7_enhance.py**: Context7 增强 (可选)
  - 元数据增强
  - 相关库发现
  - 文档上下文补充
  
- **pipeline_ingest.py**: 导入到 LightRAG
  - 实体提取
  - 关系提取
  - 向量化

### 测试脚本 (Test Scripts)
- **test_agent.py**: Agent 功能测试
- **test_agent_api.py**: Agent API 测试
- **test_diagnostics.py**: 系统诊断
- **test_ingest.py**: 摄取测试
- **test_streaming_api.py**: 流式 API 测试
- **test_context7_api.py**: Context7 API 测试
- **check_consistency.py**: 一致性检查

### 工具脚本 (Utility Scripts)
- **check_db.py**: 数据库连接检查
- **clear_cache.py**: 清除缓存
- **verify_config.py**: 配置验证
- **debug_prompts.py**: 调试提示词
- **extract_sample.py**: 提取样本数据

## 工具类 (Utilities) [`utils/`](utils/)

### 核心工具
- **schema.py**: Pydantic 数据模型
  - 所有数据契约的定义
  - 管道模型、API 模型、Agent 模型、缓存模型

- **citation.py**: 引用系统
  - 自动来源归因
  - 引用验证和修复
  - 格式: [1], [2], [3]

- **monitoring.py**: 性能监控
  - 指标收集
  - 性能追踪装饰器
  - API 调用统计

- **context7_client.py**: Context7 API 客户端
  - 元数据增强
  - 相关库发现

- **logger.py**: 日志配置
- **encoding.py**: UTF-8 编码工具

## 测试框架 (Testing Framework) [`tests/`](tests/)

### 基准测试 (Benchmarks)
- **test_deployment.py**: 部署配置场景
- **test_api_usage.py**: API 使用场景
- **test_troubleshooting.py**: 故障排查场景
- **test_openai_deployment.py**: OpenAI 特定场景

### 测试运行器
- **run_benchmark_tests.py**: 综合基准测试运行器
- **run_openai_benchmark.py**: OpenAI 基准测试运行器

## 设计决策

### 1. 写读分离架构
- **写操作**: 走离线管道，文件驱动，慢速但可调试
- **读操作**: 走在线服务，内存驱动，快速但只读
- **好处**: 解耦复杂度，优化各自性能

### 2. 多存储后端支持
- **JSON**: 本地开发，无需外部依赖
- **PostgreSQL**: 生产环境，支持向量搜索
- **Neo4j**: 图数据库专用，复杂图查询

### 3. 混合检索策略
- **向量检索**: 语义相似度
- **BM25 检索**: 关键词匹配
- **RRF 融合**: 倒数排名融合，兼顾两者优势

### 4. Pydantic 数据契约
- 所有中间数据使用 Pydantic 模型
- 类型安全和验证
- 自动序列化/反序列化
- IDE 支持自动完成

### 5. 零爬虫策略
- 仅使用 GitHub API 作为数据源
- 符合服务条款
- "Source Code is Truth" 原则

## 数据流详解

### 知识构建流程
```
GitHub 仓库
    ↓
GitHub API (零爬虫)
    ↓
原始文档 (artifacts/01_raw/*.json)
    ├─ RawDoc: 原始内容
    └─ IngestionBatch: 批次元数据
    ↓
清洗管道
    ├─ 移除 HTML 标签
    ├─ 清洗 Jupyter notebooks
    ├─ 提取 frontmatter
    └─ 修复相对链接
    ↓
清洗文档 (artifacts/02_clean/*.json)
    ├─ CleanDoc: 清洗后内容
    └─ CleanBatch: 批次元数据
    ↓
Context7 增强 (可选)
    ├─ 元数据增强
    ├─ 相关库发现
    └─ 文档上下文补充
    ↓
LightRAG 摄取
    ├─ 实体提取
    ├─ 关系提取
    └─ 分块
    ↓
PostgreSQL / Neo4j / JSON 存储
    ├─ 向量索引
    ├─ 图谱索引
    └─ 文档状态
```

### 查询请求流程
```
用户查询
    ↓
FastAPI / Streamlit
    ↓
缓存检查 (如果启用)
    ↓ (缓存未命中)
LangGraph Agent
    ↓
LightRAG 检索 (混合: 全局 + 局部 + BM25)
    ├─ 全局: 实体/关系搜索
    ├─ 局部: 文档块搜索
    └─ BM25: 关键词搜索 (可选)
    ↓
模型 (DeepSeek/Qwen API)
    ↓
引用处理
    ↓
响应生成
    ↓
缓存存储 (如果启用)
    ↓
用户响应
```

## 相关文档

- **CLAUDE.md**: 核心开发指南（精简版）
- **DEVELOPMENT.md**: 开发者完整指南
- **tests/BENCHMARKS.md**: 基准测试说明
- **README.md**: 项目概述和快速开始
