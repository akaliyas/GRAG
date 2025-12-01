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

## 系统架构

```
应用层 (FastAPI + Streamlit)
    ↓
Agent层 (LangGraph)
    ↓
模型层 (DeepSeek API / 本地模型)
    ↓
知识存储层 (LightRAG + PostgreSQL)
    ↓
数据处理层 (LightRAG)
    ↓
数据采集层 (SaaS 服务: Firecrawl/Jina Reader)
```

## 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **Agent 框架**：LangGraph
- **知识图谱**：LightRAG
- **数据库**：PostgreSQL
- **模型**：DeepSeek API / Ollama / vLLM
- **数据采集**：Firecrawl / Jina Reader API（SaaS 服务）
- **部署**：Docker Compose

## 快速开始

### 1. 环境准备

- Python 3.11+
- Docker & Docker Compose（推荐）
- PostgreSQL 15+（如果不用 Docker）

### 2. 配置环境变量

复制 `env.example` 为 `.env` 并填写配置：

```bash
cp env.example .env
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

```bash
# 安装依赖
pip install -r requirements.txt

# 使用 （可选，无需手动激活虚拟环境）
# 方式1：创建隔离环境
#   uv venv
#   uv pip sync requirements.txt
# 方式2：直接运行脚本（自动解析 requirements）
#   uv run scripts/extract_sample.py --repo <repo-url> --output samples/data.json

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

## 开发计划

### Phase 1：基础搭建（Week 1-2）
- [ ] 环境配置（Docker Compose）
- [ ] 数据采集模块开发
- [ ] LightRAG 集成和测试

### Phase 2：核心功能（Week 2-3）
- [ ] 知识图谱构建
- [ ] 双层检索实现
- [ ] API 服务开发

### Phase 3：应用开发（Week 3-4）
- [ ] 前端界面开发
- [ ] 用户反馈功能
- [ ] 系统集成测试

### Phase 4：优化完善（Week 4）
- [ ] 性能优化
- [ ] 准确度优化
- [ ] 文档编写

## TODO 清单

- [ ] 完善 LightRAG 的 prompt（实体抽取、关系抽取）
- [ ] 实现完整的流式响应（当前为模拟）
- [ ] 完善爬虫功能（Scrapy 集成）
- [ ] 实现 Agent 层的完整工具调用
- [ ] 添加单元测试和集成测试
- [ ] 完善错误处理和重试机制
- [ ] 添加 API 限流和防护
- [ ] 优化缓存策略

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
| [Playwright](https://github.com/microsoft/playwright) | Apache 2.0 | ✅ 兼容 |
| [Streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 | ✅ 兼容 |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | MIT License | ✅ 完全兼容 |
| [Scrapy](https://github.com/scrapy/scrapy) | BSD License | ✅ 兼容 |
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

