# GRAG 技术文档智能问答系统

基于知识图谱增强检索生成（GraphRAG）的技术文档智能问答系统，支持 OpenAI 官方文档、Deepseek API 文档等技术文档的智能问答。

## 项目概述

本项目采用 LightRAG 框架实现低成本、高性能的文档问答能力，通过双层检索机制（全局检索+局部检索）提升检索效果，支持 DeepSeek API 和本地模型（Ollama/vLLM）的动态切换。

## 核心特性

- 🚀 **双层检索机制**：结合全局检索和局部检索，提升检索效果
- 💰 **成本优化**：通过 LightRAG 大幅降低 API 调用成本（约 90% 节省）
- 🔄 **双模型方案**：支持 API 和本地模型，适应不同场景
- 📊 **智能缓存**：PostgreSQL 缓存表，支持 LRU 清理和质量评分
- 🔐 **安全认证**：账号密码认证，保护 API 安全
- 📈 **性能监控**：实时监控 API 调用次数、响应时间等指标
- 🔄 **流式响应**：支持流式输出，提升用户体验
- 🎯 **Zero-Crawler 策略**：完全去爬虫化，仅使用 GitHub API 作为数据源（Source Code is Truth）
- 📝 **原生结构化数据**：直接获取 Markdown 和 Jupyter Notebook 源码，自动清理 HTML 标签（包括 tfo-notebook-buttons 等导航元素），确保数据纯净

## 系统架构

### 核心架构哲学：Pipeline for Write, Service for Read

系统采用**"写操作走管道，读操作走服务"**的架构设计，明确拆分为两个解耦的子系统：

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
PostgreSQL 索引
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
Agent层 (LangGraph) - 仅负责检索
    ↓
模型层 (DeepSeek API / 本地模型)
    ↓
知识存储层 (LightRAG + PostgreSQL) - 只读查询
```

**特点**：
- 内存驱动，响应快速
- Agent 只负责检索，不涉及数据采集
- 基于已构建的索引回答问题

详细架构说明：参见 [Pipeline 架构文档](docs/PIPELINE_ARCHITECTURE.md)

## 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **Agent 框架**：LangGraph
- **知识图谱**：LightRAG
- **数据库**：PostgreSQL
- **模型**：DeepSeek API / Ollama / vLLM
- **数据采集**：GitHub API（完全去爬虫化，Zero-Crawler 策略）
- **部署**：Docker Compose

## 快速开始

### 1. 环境准备

- Python 3.11+
- Docker & Docker Compose（推荐）
- PostgreSQL 15+（如果不用 Docker）

### 2. 配置环境变量

复制 `example.env` 为 `.env` 并填写配置：

```bash
# Windows (PowerShell)
Copy-Item example.env .env

# Linux/macOS
cp example.env .env
```

编辑 `.env` 文件，填写必要的配置：

```env
POSTGRES_PASSWORD=your_password
DEEPSEEK_API_KEY=your_api_key
API_PASSWORD=your_api_password
```

### 3. 使用 Docker Compose 启动（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 4. 本地开发

**推荐使用 `uv` 进行依赖管理**（项目已配置 `uv.lock`）：

```bash
# 安装 uv（如果还没有安装）
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（uv 会自动创建虚拟环境并安装所有依赖）
uv sync

# 使用 uv 运行脚本（推荐，自动管理虚拟环境和依赖）
uv run scripts/pipeline_fetch.py --repo <repo-url> --output artifacts/01_raw/data.json
uv run scripts/pipeline_clean.py --input artifacts/01_raw/data.json --output artifacts/02_clean/data.json
uv run scripts/pipeline_ingest.py --input artifacts/02_clean/data.json
```

**或者使用传统 pip 方式**（需要手动安装依赖）：

```bash
# 注意：requirements.txt 仅包含基础工具，实际依赖需要通过 uv 管理
# 如果使用 pip，需要手动安装所有依赖包
pip install fastapi uvicorn streamlit langgraph lightrag-hku[api] PyGithub python-dotenv pydantic asyncpg
```

# 启动 PostgreSQL（如果本地没有）
docker run -d \
  --name grag_postgres \
  -e POSTGRES_DB=grag_db \
  -e POSTGRES_USER=grag_user \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:15-alpine

# 启动后端 API
uvicorn api.main:app --reload

# 启动前端（新终端）
streamlit run frontend/app.py
```

### 5. 访问服务

- **API 文档**：http://localhost:8000/docs
- **前端界面**：http://localhost:8501
- **健康检查**：http://localhost:8000/health

## 使用说明

### API 使用示例

#### 1. 问答查询

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用 OpenAI API 进行文本生成？",
    "use_cache": true,
    "stream": false
  }'
```

#### 2. 流式查询

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用 OpenAI API 进行文本生成？",
    "use_cache": true,
    "stream": true
  }'
```

#### 3. 提交反馈

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用 OpenAI API 进行文本生成？",
    "is_positive": true
  }'
```

#### 4. 切换模型

```bash
curl -X POST "http://localhost:8000/api/v1/model/switch" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "local"
  }'
```

#### 5. 获取统计信息

```bash
curl -X GET "http://localhost:8000/api/v1/stats" \
  -H "Authorization: Basic $(echo -n 'admin:password' | base64)"
```

## 配置说明

主要配置文件：`config/config.yaml`

### 关键配置项

- **数据库配置**：PostgreSQL 连接信息
- **LightRAG 配置**：实体/关系抽取 prompt（TODO: 需要根据实际需求调整）
- **模型配置**：DeepSeek API 和本地模型配置
- **缓存配置**：LRU 清理策略和质量评分阈值
- **日志配置**：日志级别和文件路径

## 文档索引

### 核心模块文档

- **[Pipeline 架构文档](docs/PIPELINE_ARCHITECTURE.md)** ⭐ - 核心架构设计：Pipeline for Write, Service for Read
- **[开发规则文档](docs/DEVELOPMENT_RULES.md)** ⭐ - 开发约定：uv 使用、脚本执行、代码风格
- **[API 层文档](docs/API_LAYER.md)** - API 接口、路由、认证机制
- **[Agent 层文档](docs/AGENT_LAYER.md)** - LangGraph Agent、工作流、意图识别（已简化，仅负责检索）
- **[模型层文档](docs/MODEL_LAYER.md)** - 模型管理、动态切换、健康检查
- **[知识存储层文档](docs/KNOWLEDGE_LAYER.md)** - LightRAG 封装、双层检索、文档管理
- **[存储层文档](docs/STORAGE_LAYER.md)** - 缓存管理、LRU 清理、质量评分
- **[工具层文档](docs/TOOLS_LAYER.md)** - GitHub 提取工具、Zero-Crawler 策略、Pydantic Schema
- **[配置管理文档](docs/CONFIG_MANAGEMENT.md)** - 配置加载、环境变量解析
- **[前端文档](docs/FRONTEND.md)** - Streamlit 界面、API 集成

### 专项文档

- **[API 认证文档](docs/API_AUTHENTICATION.md)** - HTTP Basic 认证、安全配置
- **[PostgreSQL 配置文档](docs/POSTGRESQL_CONFIG.md)** - 数据库配置、存储后端

## 多智能体协作规范

> **本文档作为开发协作的统一信息源（Single Source of Truth）**  
> AI Agent 在协助开发时，应基于此文档理解项目状态、架构设计、模块接口和开发约定。

### 模块接口规范

#### 1. API 层 (`api/`)

**主要文件：**
- `api/main.py`: FastAPI 应用入口，负责系统初始化
- `api/routes.py`: RESTful API 路由定义
- `api/auth.py`: HTTP Basic 认证

**关键接口：**

```python
# 查询接口
POST /api/v1/query
Request: {
    "query": str,           # 用户查询文本
    "use_cache": bool,      # 是否使用缓存（默认 True）
    "stream": bool          # 是否流式响应（默认 False）
}
Response: {
    "success": bool,
    "answer": str,          # 生成的答案
    "context_ids": List[str],  # 检索到的上下文 ID
    "response_time": float,
    "model_type": str,      # 使用的模型类型
    "from_cache": bool,     # 是否来自缓存
    "error": Optional[str]
}

# 反馈接口
POST /api/v1/feedback
Request: {
    "query": str,
    "is_positive": bool     # True 为正面反馈
}

# 模型切换接口
POST /api/v1/model/switch
Request: {
    "model_type": str       # "local" 或 "deepseek"
}
```

**初始化顺序：**
1. `ModelManager` → 2. `LightRAGWrapper` → 3. `GRAGAgent` → 4. `CacheManager`

#### 2. Agent 层 (`agent/`)

**主要文件：**
- `agent/grag_agent.py`: LangGraph Agent 实现

**核心接口：**

```python
class GRAGAgent:
    def __init__(
        self,
        model_manager: ModelManager,
        lightrag_wrapper: LightRAGWrapper
    )
    
    async def process_query(
        self,
        query: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        处理用户查询
        
        Returns:
            {
                "answer": str,
                "context_ids": List[str],
                "model_type": str
            }
        """
```

**Agent 工作流（LangGraph）：**
```
用户查询 → 检索 (LightRAG) → 生成答案
```

**设计变更**：
- ⚠️ **已废弃**：同步的 `GITHUB_INGEST` 意图路由（数据采集应在离线管道中完成）
- ⚠️ **已废弃**：同步的 `PREPROCESS` 意图路由（数据预处理应在离线管道中完成）
- ✅ **当前设计**：Agent 仅负责检索和答案生成，基于已构建的索引

**意图类型：**
- `QUERY`: 直接查询（唯一支持的意图）
- ~~`GITHUB_INGEST`~~: **已废弃** - 数据采集应在离线管道中完成
- ~~`PREPROCESS`~~: **已废弃** - 数据预处理应在离线管道中完成

**GitHub 提取工具 (`agent/tools/github_ingestor.py`)：**
- 支持 Markdown 和 Jupyter Notebook 文件
- 自动清洗 Notebook（仅保留 Markdown 和 Code 输入，丢弃 Output）
- **HTML 标签清理**：完全移除 `tfo-notebook-buttons` 等导航元素，移除所有 HTML 标签保留纯文本
- 提取 Markdown Frontmatter
- 修复相对链接为 GitHub Raw URL
- 保护 Markdown 代码块，避免误删代码块内的内容

#### 3. 模型层 (`models/`)

**主要文件：**
- `models/model_manager.py`: 模型管理器

**核心接口：**

```python
class ModelManager:
    def get_llm(self) -> BaseLLM:
        """获取当前 LLM 实例（优先本地，失败时回退到 API）"""
    
    def switch_model(self, model_type: str) -> bool:
        """切换模型类型：'local' 或 'deepseek'"""
    
    def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Any:
        """统一的聊天补全接口"""
```

**模型优先级：**
1. 本地模型（Ollama/vLLM）- 优先使用
2. DeepSeek API - 本地模型不可用时自动回退

#### 4. 知识存储层 (`knowledge/`)

**主要文件：**
- `knowledge/lightrag_wrapper.py`: LightRAG 封装

**核心接口：**

```python
class LightRAGWrapper:
    def __init__(self, model_manager: ModelManager)
    
    def add_documents(
        self, 
        documents: List[str],
        file_paths: Optional[List[str]] = None
    ) -> None:
        """
        添加文档到知识库
        
        Args:
            documents: 文档文本列表
            file_paths: 文件路径列表（用于引文功能）
        """
    
    def ingest_from_json_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        从 GitHubIngestor 输出的 JSON 文件导入文档到 LightRAG
        
        Args:
            json_file_path: JSON 文件路径
            
        Returns:
            导入结果统计信息（包含成功状态、文档数量、数据源信息等）
        """
    
    def query(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        查询知识库
        
        Returns:
            {
                "answer": str,
                "context_ids": List[str],  # 检索到的实体/文档 ID
                "entities": List[str],     # 相关实体
                "relations": List[str]     # 相关关系
            }
        """
```

**存储类型：**
- PostgreSQL（默认）
- Neo4j（可选）

#### 5. 存储层 (`storage/`)

**主要文件：**
- `storage/cache_manager.py`: PostgreSQL 缓存管理

**核心接口：**

```python
class CacheManager:
    def get_cache(self, query: str) -> Optional[Dict]:
        """获取缓存（如果存在）"""
    
    def set_cache(
        self,
        query: str,
        answer: str,
        context_ids: List[str],
        model_type: str,
        response_time: float
    ) -> None:
        """设置缓存"""
    
    def update_feedback(
        self,
        query: str,
        is_positive: bool
    ) -> None:
        """更新反馈，影响质量评分"""
```

**缓存策略：**
- LRU 清理：定时任务 + 容量限制
- 质量评分：基于用户反馈（正面反馈提升，负面反馈降低）

### 数据流图

#### 查询请求完整流程（在线服务）

```
用户请求
    ↓
API 层 (api/routes.py)
    ├─ 认证验证 (api/auth.py)
    ├─ 缓存检查 (storage/cache_manager.py)
    │   └─ [命中] → 直接返回缓存结果
    └─ [未命中] → Agent 层
        ↓
Agent 层 (agent/grag_agent.py)
    └─ 检索 (knowledge/lightrag_wrapper.py)
        ↓
LightRAG 双层检索（只读）
    ├─ 全局检索（实体/关系）
    └─ 局部检索（文档块）
        ↓
模型层 (models/model_manager.py)
    ├─ 优先：本地模型 (Ollama/vLLM)
    └─ 回退：DeepSeek API
        ↓
生成答案
    ↓
缓存存储 (storage/cache_manager.py)
    ↓
返回响应
```

#### 知识构建流程（离线管道）

```
GitHub API
    ↓
scripts/pipeline_fetch.py
    ↓
artifacts/01_raw/{repo}_raw.json (Raw Artifact)
    ↓
scripts/pipeline_clean.py
    ↓
artifacts/02_clean/{repo}_clean.json (Clean Artifact)
    ↓
scripts/pipeline_ingest.py
    ↓
LightRAG 实体抽取
    ↓
PostgreSQL 索引
```

**关键点**：
- 知识构建是**离线管道**，不在用户请求时执行
- 每个步骤的输入输出都是文件，支持断点调试和人工介入
- 使用 Pydantic Schema 作为数据契约

#### 数据格式转换

```
用户查询 (str)
    ↓
QueryRequest (Pydantic Model)
    ↓
AgentState (LangGraph State)
    ↓
LightRAG QueryParam
    ↓
LLM Messages (List[Dict])
    ↓
答案 (str)
    ↓
QueryResponse (Pydantic Model)
```

### 开发约定

#### 代码风格

- **Python 版本**：Python 3.11+
- **代码格式**：遵循 PEP 8
- **类型注解**：尽可能使用类型提示
- **文档字符串**：所有公共函数/类必须有 docstring（Google 风格）

#### 命名约定

- **模块名**：小写 + 下划线（如 `grag_agent.py`）
- **类名**：大驼峰（如 `GRAGAgent`）
- **函数/变量名**：小写 + 下划线（如 `process_query`）
- **常量**：全大写 + 下划线（如 `MAX_RETRIES`）

#### 错误处理

- **API 层**：使用 FastAPI 的 `HTTPException`
- **业务层**：抛出自定义异常，由上层捕获
- **日志记录**：所有错误必须记录日志（使用 `logger.error`）

#### 测试要求

- **单元测试**：每个模块应有对应的测试文件
- **集成测试**：关键流程应有端到端测试
- **测试文件位置**：`tests/` 目录

#### 配置管理

- **敏感信息**：使用环境变量（`.env` 文件）
- **非敏感配置**：使用 `config/config.yaml`
- **配置加载**：统一通过 `config/config_manager.py`

### 设计决策记录

#### 1. 为什么选择 LightRAG？

- **成本优势**：相比传统 RAG，API 调用成本降低约 90%
- **双层检索**：全局检索（实体/关系）+ 局部检索（文档块），提升检索精度
- **知识图谱**：支持实体和关系的结构化存储，便于复杂查询

#### 2. 为什么使用双层检索？

- **全局检索**：基于知识图谱的实体和关系检索，适合概念性查询
- **局部检索**：基于向量相似度的文档块检索，适合具体细节查询
- **互补优势**：两种检索方式结合，覆盖不同查询场景

#### 3. 为什么优先使用本地模型？

- **成本控制**：本地模型无 API 调用费用
- **隐私保护**：数据不出本地
- **响应速度**：本地推理延迟更低（如果硬件支持）

#### 4. 为什么使用 PostgreSQL 作为缓存？

- **结构化存储**：支持复杂查询（质量评分、访问统计等）
- **性能优化**：索引支持快速查询
- **数据持久化**：服务重启后缓存不丢失

#### 5. 为什么使用 LangGraph 作为 Agent 框架？

- **状态管理**：内置状态机，便于管理复杂工作流
- **工具调用**：原生支持工具节点，易于扩展
- **条件路由**：支持基于意图的条件分支

#### 6. 为什么采用 Zero-Crawler 策略？

- **数据纯净**：直接获取源码（Source Code is Truth），避免 HTML 噪音
- **成本为零**：GitHub API 免费额度充足（5000次/时），无需 SaaS 服务费用
- **工程简化**：移除爬虫依赖，降低系统复杂度
- **质量提升**：结构化数据（Markdown/Notebook）更利于 LightRAG 实体抽取
- **自动清洗**：内置 HTML 标签清理机制，完全移除导航元素和 HTML 标签，确保 RAG 检索精度

### 已知限制与边界情况

#### 当前限制

1. **流式响应**：当前为模拟实现，需要完善
2. **LightRAG Prompt**：实体/关系抽取 prompt 需要根据实际需求调整（标记为 TODO）
3. **数据源限制**：仅支持 GitHub 仓库作为数据源（Zero-Crawler 策略）
4. **错误重试**：部分模块缺少完整的重试机制

#### 错误处理策略

- **模型调用失败**：自动回退到备用模型
- **数据库连接失败**：记录错误，返回友好提示
- **LightRAG 查询失败**：记录错误，返回默认响应

#### 降级方案

- **缓存不可用**：跳过缓存，直接查询
- **本地模型不可用**：自动切换到 API 模型
- **LightRAG 不可用**：返回错误信息，不生成答案

### AI Agent 协作指南

#### 开发新功能时

1. **先查阅本文档**：了解相关模块的接口和约定
2. **遵循现有模式**：参考相似功能的实现方式
3. **更新文档**：如有接口变更，同步更新本文档
4. **添加测试**：新功能必须包含测试用例

#### 修改现有代码时

1. **保持接口兼容**：尽量不破坏现有接口
2. **更新类型注解**：如有类型变更，同步更新
3. **检查依赖关系**：确保不影响其他模块
4. **运行测试**：确保所有测试通过

#### 遇到问题时

1. **查看日志**：检查 `logs/` 目录下的日志文件
2. **检查配置**：确认 `config/config.yaml` 和 `.env` 配置正确
3. **查阅文档**：参考本文档和相关代码注释
4. **记录问题**：在 TODO 清单中记录待解决的问题

## 开发计划（快速原型开发路线）

### Phase 1：知识构建管道开发（当前阶段）

**目标**：建立文件驱动的数据流，支持可调试的知识构建

- [x] GitHub 数据提取工具（`GitHubIngestor`）
- [x] 初步测试通过
- [ ] **定义 Pydantic Schema**（`utils/schema.py`）
  - `RawDoc`、`IngestionBatch`
  - `CleanDoc`、`CleanBatch`
- [ ] **重构 `GitHubIngestor`**：返回 `IngestionBatch` 对象，支持文件读写
- [ ] **创建管道脚本**：
  - `scripts/pipeline_fetch.py`：Step 1 - 提取原始数据到 `artifacts/01_raw/`
  - `scripts/pipeline_clean.py`：Step 2 - 清洗数据到 `artifacts/02_clean/`
  - `scripts/pipeline_ingest.py`：Step 3 - 导入到 LightRAG
- [ ] **LightRAG 测试开发**：验证实体抽取和索引构建

**设计原则**：
- 每个步骤的输入输出都是文件，可随时检查
- 支持断点、回放、人工介入
- 使用 Pydantic Schema 作为数据契约
- **使用 `uv run` 执行脚本**（自动管理虚拟环境和依赖）

**执行示例**：
```bash
# Step 1: Fetch
uv run scripts/pipeline_fetch.py --repo https://github.com/openai/openai-python --output artifacts/01_raw/openai_v1_raw.json

# Step 2: Clean
uv run scripts/pipeline_clean.py --input artifacts/01_raw/openai_v1_raw.json --output artifacts/02_clean/openai_v1_clean.json

# Step 3: Ingest
uv run scripts/pipeline_ingest.py --input artifacts/02_clean/openai_v1_clean.json
```

### Phase 2：在线问答服务开发

**目标**：基于已构建的索引提供快速问答服务

- [ ] **简化 Agent 层**：移除同步的 `GITHUB_INGEST` 意图路由
- [ ] **Agent 只负责检索**：基于 LightRAG 进行查询和答案生成
- [ ] **API 服务开发**：提供 RESTful API 接口
- [ ] **前端界面开发**：Streamlit 界面集成

**设计原则**：
- Agent 非常轻量，只负责读
- 响应快速，不涉及数据采集
- 基于已构建的索引回答问题

### Phase 3：系统集成和优化

**目标**：完善系统功能，优化性能和准确度

- [ ] 双层检索实现和优化
- [ ] 缓存机制完善
- [ ] 用户反馈功能
- [ ] 系统集成测试
- [ ] 性能优化和准确度优化

### Phase 4：生产化改造（可选）

**目标**：将文件驱动模式切换为 API 驱动模式

- [ ] FastAPI 数据导入接口（复用核心逻辑）
- [ ] 后台任务队列（异步处理数据导入）
- [ ] 监控和日志完善
- [ ] 文档编写

**设计原则**：
- 核心逻辑不变，只改变数据流（文件 → 内存/队列）
- 使用 Pydantic Schema 作为统一接口

## TODO 清单

### Phase 1：知识构建管道开发（当前阶段）

#### 1. 数据模型定义
- [x] **定义 Pydantic Schema**（`utils/schema.py`）
  - [x] `RawDoc`：原始文档模型
  - [x] `IngestionBatch`：批量导入批次（Raw Artifact），支持 `save_to_file()` 和 `load_from_file()`
  - [x] `CleanDoc`：清洗后的文档模型
  - [x] `CleanBatch`：清洗批次（Clean Artifact），支持 `save_to_file()` 和 `load_from_file()`

#### 2. 工具重构
- [x] **重构 `GitHubIngestor`**（`agent/tools/github_ingestor.py`）
  - [x] 添加 `extract_repo_docs()` 返回 `IngestionBatch` 对象
  - [x] 保留 `download_and_clean()` 方法（向后兼容）
  - [x] 确保返回对象支持文件读写

#### 3. 管道脚本开发
- [x] **创建 `scripts/pipeline_fetch.py`**
  - [x] Step 1：提取原始数据到 `artifacts/01_raw/`
  - [x] 调用 `GitHubIngestor.extract_repo_docs()`
  - [x] 保存为 `IngestionBatch` JSON 文件
- [x] **创建 `scripts/pipeline_clean.py`**
  - [x] Step 2：清洗数据到 `artifacts/02_clean/`
  - [x] 读取 Raw Artifact
  - [x] 应用清洗逻辑（HTML 标签、Notebook 清理、链接修复）
  - [x] 保存为 `CleanBatch` JSON 文件
- [x] **创建 `scripts/pipeline_ingest.py`**
  - [x] Step 3：导入到 LightRAG
  - [x] 读取 Clean Artifact
  - [x] 调用 `LightRAGWrapper.ingest_batch()`
  - [x] 验证导入结果

#### 4. LightRAG 集成
- [x] **重构 `LightRAGWrapper.ingest_batch()`**（`knowledge/lightrag_wrapper.py`）
  - [x] 接收 `CleanBatch` Pydantic 对象
  - [x] 提取文档内容和元数据
  - [x] 调用 LightRAG 的 `insert()` 方法
  - [x] 添加 `ingest_from_file()` 辅助方法
- [ ] **LightRAG 测试开发**
  - [ ] 验证实体抽取功能
  - [ ] 验证索引构建功能
  - [ ] 测试端到端流程（Fetch → Clean → Ingest）
  - [ ] 验证 PostgreSQL 索引写入

#### 5. 优化和完善
- [ ] 完善 LightRAG 的 prompt（实体抽取、关系抽取）
- [ ] 创建 `artifacts/` 目录结构（01_raw、02_clean、03_graph）
- [ ] 添加 `.gitignore` 规则（忽略 artifacts 目录）

### Phase 2：在线问答服务开发

- [ ] **简化 Agent 层**：移除同步的 `GITHUB_INGEST` 意图路由
- [ ] **Agent 只负责检索**：基于 LightRAG 进行查询和答案生成
- [ ] **API 服务开发**：提供 RESTful API 接口
- [ ] **前端界面开发**：Streamlit 界面集成

### Phase 3：系统集成和优化

- [ ] 双层检索实现和优化
- [ ] 缓存机制完善
- [ ] 用户反馈功能
- [ ] 系统集成测试
- [ ] 性能优化和准确度优化
- [ ] 实现完整的流式响应（当前为模拟）
- [ ] 优化 GitHub API 数据提取性能
- [ ] 添加单元测试和集成测试
- [ ] 完善错误处理和重试机制
- [ ] 添加 API 限流和防护
- [ ] 优化缓存策略

### 已完成的任务 ✅

- [x] 实现 HTML 标签清理（包括 tfo-notebook-buttons 等导航元素）
- [x] 打通 LightRAG 入库接口（`ingest_from_json_file` 方法）
- [x] GitHub 数据提取初步测试通过

## 技术指标

- **响应时间**：平均响应时间 < 2秒
- **成本控制**：API 调用成本 < 2.2元（200-400页文档）

## 许可证

### 项目许可证

本项目采用 **MIT License** 开源许可证。

### 依赖项目许可证

本项目使用了以下开源项目，它们的许可证与 MIT License 完全兼容：

| 依赖项目 | 许可证 | 兼容性 |
|---------|--------|--------|
| [LightRAG](https://github.com/HKUDS/LightRAG) | MIT License | ✅ 完全兼容 |
| [FastAPI](https://github.com/tiangolo/fastapi) | MIT License | ✅ 完全兼容 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT License | ✅ 完全兼容 |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT License | ✅ 完全兼容 |
| [PyGithub](https://github.com/PyGithub/PyGithub) | LGPL-3.0 | ✅ 兼容 |
| [Streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 | ✅ 兼容 |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | MIT License | ✅ 完全兼容 |
| [nbformat](https://github.com/jupyter/nbformat) | BSD License | ✅ 兼容 |
| [PostgreSQL](https://www.postgresql.org/) | PostgreSQL License | ✅ 兼容 |

### 许可证兼容性说明

MIT License 是兼容性最好的开源许可证之一，可以：
- ✅ 与 MIT、Apache 2.0、BSD 等宽松许可证兼容
- ✅ 允许商业使用、修改和分发
- ✅ 只需保留原始版权声明

所有依赖项目的许可证都与 MIT License 兼容，可以安全地在本项目中使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

项目负责人：akaliyas

