# GRAG 开发与验证指南

本文档面向**人类开发者**，说明如何在本地搭建环境、运行服务、做功能验证与回归测试。架构背景与 AI 协作约定见根目录 [`CLAUDE.md`](./CLAUDE.md)；产品功能与用户使用见 [`README.md`](./README.md)。

---

## 1. 前置条件（Prerequisites）

| 项 | 说明 |
|----|------|
| Python | **3.11+** |
| 包管理 | 推荐使用 [**uv**](https://github.com/astral-sh/uv)（与 `uv.lock` 一致） |
| 密钥 | 对话模型（如 **DeepSeek**）、嵌入（如 **SiliconFlow**）的 API Key；API **HTTP Basic** 密码 |
| 可选 | **Docker** / **Docker Compose**（数据库模式）；**GitHub Token**（管道拉取私有仓库或提高 API 限额） |

安装 uv（若尚未安装）：

```bash
pip install uv
```

同步依赖（在项目根目录）：

```bash
uv sync
```

> 若仓库中包含 `pyproject.toml` 但未出现在你的检出中，请从上游同步；Docker 构建会依赖 `uv.lock` 与 `pyproject.toml`。

---

## 2. 首次配置

### 2.1 环境变量

1. 复制模板：将 [`example.env`](./example.env) 复制为项目根目录下的 **`.env`**。
2. 至少填写（按你实际使用的提供商调整）：
   - **`DEEPSEEK_API_KEY`**：对话模型（或按 `config/config.yaml` 中 `models` 配置使用其他 API）。
   - **`EMBEDDING_API_KEY`** / **`EMBEDDING_PROVIDER`** / **`EMBEDDING_MODEL`**：与嵌入服务一致（参见 `example.env` 内注释）。
   - **`API_USERNAME`** / **`API_PASSWORD`**：REST API 的 **HTTP Basic** 认证；未配置 `API_PASSWORD` 时，受保护接口会返回 500 提示。
3. **`.env` 中建议不要**在键名两侧加空格（例如应写 `GITHUB_TOKEN=...`，避免 `KEY = value` 导致解析异常）。

敏感项仅放在 `.env` 或本机密钥管理；**不要提交 `.env`**。

### 2.2 配置文件

非敏感项在 [`config/config.yaml`](./config/config.yaml)，支持 `${ENV:default}`。本地与 Docker 预设见 `config/presets/`。

---

## 3. 本地开发：推荐工作流

### 3.1 纯文件存储模式（无 PostgreSQL，适合日常开发）

与 Windows 下 [`start_backend.bat`](./start_backend.bat) 对齐的典型变量：

**PowerShell：**

```powershell
$env:DEPLOYMENT_MODE = "local"
$env:LIGHTRAG_STORAGE_TYPE = "file"
$env:STORAGE_MODE = "file"
uv run python scripts/verify_config.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Bash：**

```bash
export DEPLOYMENT_MODE=local
export LIGHTRAG_STORAGE_TYPE=file
export STORAGE_MODE=file
uv run python scripts/verify_config.py
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

另开终端启动 **Streamlit** 前端：

```bash
uv run streamlit run frontend/app.py
```

常用地址：

| 服务 | URL |
|------|-----|
| API 文档（Swagger） | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |
| 前端 | http://localhost:8501（默认端口以 Streamlit 输出为准） |

### 3.2 使用仓库自带启动脚本

- Windows：[`start_backend.bat`](./start_backend.bat)、[`start_frontend.bat`](./start_frontend.bat)（会先尝试 `uv run python scripts/verify_config.py`）。
- Linux/macOS：[`start_backend.sh`](./start_backend.sh)、[`start_frontend.sh`](./start_frontend.sh)。

---

## 4. 验证与诊断（建议每次大改或发版前跑一遍）

按**由浅入深**顺序执行。

### 4.1 配置与存储

```bash
uv run python scripts/verify_config.py
```

### 4.2 环境与依赖诊断

```bash
uv run scripts/test_diagnostics.py
```

可选子检查（见脚本内说明）：

```bash
uv run scripts/test_diagnostics.py --check env
uv run scripts/test_diagnostics.py --check db
uv run scripts/test_diagnostics.py --check lightrag
uv run scripts/test_diagnostics.py --check model
```

### 4.3 数据库（若使用 PostgreSQL 模式）

```bash
uv run scripts/check_db.py
```

### 4.4 HTTP API 冒烟（需后端已启动且已配置 Basic 认证）

将 `admin` / `你的API_PASSWORD` 换成 `.env` 中实际值。

**PowerShell：**

```powershell
$cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:你的API_PASSWORD"))
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query" -Method Post -Headers @{ Authorization = "Basic $cred" } -ContentType "application/json" -Body '{"query":"你好","use_cache":false,"stream":false}'
```

**curl（Linux/macOS/Git Bash）：**

```bash
curl -s http://localhost:8000/health
curl -s -X POST "http://localhost:8000/api/v1/query" \
  -u "admin:你的API_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"query":"你好","use_cache":false,"stream":false}'
```

> 根目录 [`test_client.py`](./test_client.py) 为快速客户端示例；若接口启用了 **Basic Auth**，需在请求中自行携带 `Authorization` 头，否则会 **401**。

---

## 5. 功能与集成测试脚本（`scripts/`）

**约定：脚本一律用 `uv run` 执行**，以保证依赖与虚拟环境一致。

| 脚本 | 用途 |
|------|------|
| [`scripts/test_agent.py`](./scripts/test_agent.py) | Agent 行为 |
| [`scripts/test_agent_api.py`](./scripts/test_agent_api.py) | 含直接调用 Agent、工作流等 |
| [`scripts/test_diagnostics.py`](./scripts/test_diagnostics.py) | 环境 / DB / LightRAG / 模型诊断 |
| [`scripts/test_ingest.py`](./scripts/test_ingest.py) | 数据写入 / 摄取相关 |
| [`scripts/test_streaming_api.py`](./scripts/test_streaming_api.py) | 流式 API |
| [`scripts/test_context7_api.py`](./scripts/test_context7_api.py) | Context7（可选增强） |
| [`scripts/test_pipeline_ragflow.py`](./scripts/test_pipeline_ragflow.py) | 与 RAGFlow 管道相关的测试 |

其他维护用脚本：`clear_cache.py`、`debug_prompts.py`、`extract_sample.py` 等，按需使用。

---

## 6. 知识管道（Pipeline）与问答联调

在线服务**默认不完成「从 GitHub 到索引」的完整摄取**；构建知识库应走离线管道（与 README / CLAUDE 中 **Pipeline for Write** 一致）：

1. **Fetch**：`uv run scripts/pipeline_fetch.py --repo <GitHub URL> --output artifacts/01_raw/...json`
2. **Clean**：`uv run scripts/pipeline_clean.py --input ... --output artifacts/02_clean/...json`
3. **（可选）Context7**：`uv run scripts/pipeline_context7_enhance.py`（参见脚本参数）
4. **Ingest**：`uv run scripts/pipeline_ingest.py --input artifacts/02_clean/...json`

JSON 模式快速路径可参考 [`scripts/ingest_with_json_mode.py`](./scripts/ingest_with_json_mode.py) 与 README 中的说明。

完成摄取后，再用 **4.4** 或前端 **Chat** 页面对同一知识库做问答验证。

---

## 7. 基准测试与回归（`tests/`）

- **说明文档**：[tests/BENCHMARKS.md](./tests/BENCHMARKS.md)（场景分类、难度、报告路径等）。
- **全量或分类运行**（示例）：

```bash
uv run tests/run_benchmark_tests.py --full
uv run tests/run_benchmark_tests.py --category deployment
uv run tests/run_openai_benchmark.py
```

基准报告输出位置以 `tests/BENCHMARKS.md` 为准（通常为 `artifacts/test_results/` 下相关目录）。

---

## 8. 一致性防呆清单 (Consistency Checklist)

> **为何需要**：GRAG 采用了"写走管道、读走服务"的架构，涉及多存储、混合检索、缓存等复杂环节，容易出现**索引不同步、配置冲突、缓存过期**等一致性问题。本清单帮助预防常见的一致性漏洞。

### 8.1 🔴 发布前必查 (Critical - 每次发版或大改动前执行)

| 检查项 | 操作命令/方法 | 失败后果 |
|--------|---------------|----------|
| **索引与管道同步** | `uv run scripts/test_diagnostics.py --check lightrag` | 用户查询到旧数据或无结果 |
| **嵌入模型版本一致** | 检查 `config.yaml` 中 `embedding_model` 与已嵌入向量是否匹配 | 新旧向量混存，语义空间不一致 |
| **配置源单一性** | `uv run python scripts/verify_config.py` | 多配置源冲突导致意外行为 |
| **API 响应契约** | `uv run scripts/test_agent_api.py` | 前端字段解析错误或 NPE |
| **缓存数据新鲜度** | 检查 `data/cache/` 最近修改时间，知识库更新后执行 `uv run scripts/clear_cache.py` | 返回过期答案 |

### 8.2 🟡 日常开发自检 (Routine - 每次修改相关代码后执行)

#### 数据与索引层
- [ ] **修改嵌入模型后**：删除 `rag_storage/vector_store_*.json` 并重新 ingest
- [ ] **修改 BM25 参数后**：删除 `rag_storage/bm25/` 索引目录并重建
- [ ] **管道产物更新后**：确认 `artifacts/02_clean/` 已被最新 ingest 消费
- [ ] **多路索引对齐**：向量、图、KV 存储的文档数量应一致

#### 检索与生成层
- [ ] **引用编号验证**：检查答案中 `[1][2]` 与 `context_metadata` 的对应关系
- [ ] **混合检索结果**：确认 RRF 融合的各路结果来自同一数据源版本
- [ ] **上下文窗口一致性**：Top-K 变更后验证模型实际接收的上下文长度

#### 配置与环境层
- [ ] **存储模式确认**：`STORAGE_MODE` / `LIGHTRAG_STORAGE_TYPE` 与预期一致
- [ ] **环境变量优先级**：`.env` 中值不应被硬编码覆盖
- [ ] **多环境隔离**：开发机与 Docker 环境的存储路径不冲突

#### 缓存与性能层
- [ ] **缓存失效策略**：知识库更新后相关缓存已清除
- [ ] **质量评分阈值**：`cache.quality.low_threshold` / `high_threshold` 设置合理
- [ ] **并发安全性**：多进程/多实例下缓存写入无竞态条件

### 8.3 🟢 可选增强检查 (Optional - 用于质量保证)

#### Agent 与 API 契约
- [ ] **Pydantic 模型同步**：`AgentState` / `QueryResponse` 字段与实际使用一致
- [ ] **意图路由验证**：废弃的意图分支（如 `GITHUB_INGEST`）已正确屏蔽
- [ ] **错误处理完整性**：所有异常分支返回完整的错误信息

#### 管道数据契约
- [ ] **Schema 版本兼容**：旧 artifact 文件能正确反序列化
- [ ] **doc_id 规则稳定**：`MD5(source_url:path)` 规则未变更
- [ ] **批次追踪**：`IngestionBatch` / `CleanBatch` 元数据完整

#### 可观测性
- [ ] **日志一致性**：关键操作的日志级别和格式统一
- [ ] **指标对齐**：Dashboard 指标与 API 日志一致
- [ ] **会话状态**：Streamlit 多页面切换后状态正确

### 8.4 📋 快速检查脚本 (One-liner Checks)

```bash
# 完整一致性检查（推荐发版前执行）
uv run scripts/test_diagnostics.py --check env && \
uv run scripts/test_diagnostics.py --check lightrag && \
uv run scripts/test_agent_api.py && \
uv run python scripts/verify_config.py

# 仅检查索引和配置
uv run scripts/test_diagnostics.py --check lightrag && \
uv run python scripts/verify_config.py

# 检查缓存一致性
ls -lt data/cache/ | head -5  # 查看最新缓存时间
```

### 8.5 ⚠️ 一致性问题应急预案

| 问题现象 | 紧急处理 | 根本解决 |
|----------|----------|----------|
| 检索无结果但 ingest 成功 | 检查 `LIGHTRAG_WORKING_DIR` 路径 | 统一配置源 |
| 答案包含过期信息 | `uv run scripts/clear_cache.py` | 添加版本化缓存键 |
| 引用编号错乱 | 检查 `citation.py` 修复逻辑 | 完整的引用验证流程 |
| 前端显示错误后端正常 | 检查 `QueryResponse` 字段契约 | API 响应模式统一 |
| 混合检索结果异常 | 验证向量/BM25/图索引同步 | 添加索引一致性检查 |

### 8.6 🛠️ 自动化检查工具

项目提供了一致性检查脚本 [`scripts/check_consistency.py`](./scripts/check_consistency.py)，可快速验证系统一致性：

```bash
# 运行完整检查（推荐发版前执行）
uv run scripts/check_consistency.py

# 仅检查特定类别
uv run scripts/check_consistency.py --category data    # 数据与索引层
uv run scripts/check_consistency.py --category config  # 配置一致性
uv run scripts/check_consistency.py --category cache   # 缓存一致性
uv run scripts/check_consistency.py --category api     # API 契约

# 快速检查（仅关键项）
uv run scripts/check_consistency.py --quick
```

**集成到 CI/CD**：建议在发版流程中自动运行完整检查，确保一致性状态。

---

## 9. 开发时注意点（简表）

| 主题 | 建议 |
|------|------|
| 数据契约 | 管道与 API 共用 [`utils/schema.py`](./utils/schema.py) 中的 **Pydantic** 模型，修改字段时同步调用方 |
| 日志与指标 | API 层使用 [`utils/monitoring.py`](./utils/monitoring.py) 的装饰器；关注启动失败时的堆栈与 `logs`（若项目有配置） |
| 关闭逻辑 | 修改 `api/main.py` 中 **shutdown** 时，确认是否应操作**启动时注入的** `CacheManager` 实例，避免新建空实例 |
| Windows 编码 | 入口已调用 `utils.encoding.ensure_utf8_encoding()`；管道输出中文时终端需支持 UTF-8 |

---

## 10. 文档索引

| 文档 | 用途 |
|------|------|
| [README.md](./README.md) | 产品说明、用户向快速开始 |
| [CLAUDE.md](./CLAUDE.md) | 架构、模块边界、给 AI 助手的约定 |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | 本文：人类开发与验证 |
| [tests/BENCHMARKS.md](./tests/BENCHMARKS.md) | 基准测试套件说明 |
| [example.env](./example.env) | 环境变量模板 |

---

## 11. 故障排查速查

| 现象 | 可能原因 |
|------|----------|
| 启动报 API 密码未配置 | `.env` 未设置 **`API_PASSWORD`** 或与 `config` 中 `auth` 不一致 |
| 嵌入 / 对话 429 或超时 | API Key、网络、代理；诊断脚本看 `HTTP_PROXY` / `HTTPS_PROXY` |
| 检索无结果 | 未完成 **ingest**，或 `rag_storage` / 工作目录与当前 `LIGHTRAG_WORKING_DIR` 不一致 |
| PostgreSQL 连接失败 | 服务未起、`.env` 中主机端口密码、或先用 **file** 模式隔离问题 |

若以上无法解决，把 **`uv run scripts/test_diagnostics.py`** 的完整输出与最小复现步骤一并提交 **Issue**，便于定位。
