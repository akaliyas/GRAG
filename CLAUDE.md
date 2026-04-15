# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRAG (Graph Retrieval Augmented Generation) is a technical document intelligent Q&A system based on knowledge graph enhanced retrieval generation. It uses LightRAG framework for low-cost, high-performance document Q&A with hybrid retrieval mechanism (global + local + BM25).

**Core Architecture Philosophy**: "Pipeline for Write, Service for Read" - two decoupled subsystems:
- **Knowledge Construction Pipeline** (file-driven, slow, debuggable): GitHub API → artifacts/01_raw → cleaning → artifacts/02_clean → LightRAG ingestion → PostgreSQL/Neo4j indexing
- **Online Q&A Service** (memory-driven, fast, read-only): User query → FastAPI/Streamlit → LangGraph Agent → Model → Knowledge Storage

**Key Features**:
- 📊 Hybrid retrieval: Vector search + BM25 keyword search with RRF fusion
- 🎯 Citation system: Automatic source attribution with [1], [2] format
- 📈 Performance monitoring: Built-in metrics collection and performance tracking
- 🔧 Multi-deployment: Local (JSON) and Docker (PostgreSQL/Neo4j) modes
- 🎨 Multi-page UI: Chat, Dashboard, Graph visualization, and Pipeline management

## Essential Commands

### Package Management (uses `uv`)
```bash
# Install dependencies
uv sync

# Run pipeline scripts (ALWAYS use uv run)
uv run scripts/pipeline_fetch.py --repo <github-url> --output artifacts/01_raw/data.json
uv run scripts/pipeline_clean.py --input artifacts/01_raw/data.json --output artifacts/02_clean/data.json
uv run scripts/pipeline_ingest.py --input artifacts/02_clean/data.json
```

### Development
```bash
# Backend (FastAPI)
uvicorn api.main:app --reload

# Frontend (Streamlit)
streamlit run frontend/app.py

# Database test
uv run scripts/check_db.py
```

### Docker Deployment
```bash
docker-compose up -d    # Start all services
docker-compose logs -f  # View logs
docker-compose down     # Stop services
```

### Testing
```bash
uv run scripts/test_agent.py         # Test agent functionality
uv run scripts/test_diagnostics.py  # System diagnostics
uv run scripts/test_query.py         # Test query functionality
uv run scripts/test_streaming.py     # Test streaming responses
uv run scripts/check_db.py           # Database connection test
```

## Architecture & Key Components

### Initialization Order (api/main.py:100)
1. **StorageFactory** → 2. **ModelManager** → 3. **LightRAGWrapper** → 4. **GRAGAgent** → 5. **CacheManager**

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Chat    │  │ Dashboard │  │  Graph   │  │ Pipeline │   │
│  │  Page    │  │   Page    │  │  Page    │  │  Page    │   │
│  └──────────┘  └───────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI Routes: /query, /feedback, /model/switch   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Agent Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangGraph Agent: Intent → Retrieve → Generate       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Knowledge Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │  LightRAG       │  │  BM25 Indexer   │  │  Citation  │ │
│  │  Wrapper        │  │  (Optional)     │  │  System    │ │
│  └─────────────────┘  └─────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Storage Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  StorageFactory → Cache │ Graph │ Knowledge Storage  │   │
│  │  (JSON/PostgreSQL/Neo4j)                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Model Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Manager: DeepSeek API (Ollama deprecated)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Modules

**Frontend Layer** ([`frontend/`](d:\Project\Python\GRAG\frontend\))
- [`app.py`](d:\Project\Python\GRAG\frontend\app.py): Streamlit multi-page application
- [`pages/chat.py`](d:\Project\Python\GRAG\frontend\pages\chat.py): Q&A chat interface
- [`pages/dashboard.py`](d:\Project\Python\GRAG\frontend\pages\dashboard.py): System metrics and statistics
- [`pages/graph.py`](d:\Project\Python\GRAG\frontend\pages\graph.py): Knowledge graph visualization
- [`pages/pipeline.py`](d:\Project\Python\GRAG\frontend\pages\pipeline.py): Pipeline management interface
- [`storage/session_manager.py`](d:\Project\Python\GRAG\frontend\storage\session_manager.py): Session state management

**API Layer** ([`api/`](d:\Project\Python\GRAG\api\))
- [`main.py`](d:\Project\Python\GRAG\api\main.py): FastAPI entry point, system startup/shutdown
- [`routes.py`](d:\Project\Python\GRAG\api\routes.py): RESTful API endpoints
  - `POST /api/v1/query`: Standard query
  - `POST /api/v1/query/stream`: Streaming query (SSE)
  - `POST /api/v1/feedback`: User feedback
  - `POST /api/v1/model/switch`: Model switching
  - `GET /api/v1/graph/*`: Graph statistics and data
  - `GET /api/v1/knowledge/*`: Knowledge base stats
- [`auth.py`](d:\Project\Python\GRAG\api\auth.py): HTTP Basic authentication

**Agent Layer** ([`agent/`](d:\Project\Python\GRAG\agent\))
- [`grag_agent.py`](d:\Project\Python\GRAG\agent\grag_agent.py): LangGraph-based agent
  - **IMPORTANT**: Agent only handles retrieval and answer generation
  - Data ingestion happens in offline pipeline scripts
  - Intent types: `QUERY` only (deprecated: `GITHUB_INGEST`, `PREPROCESS`)
  - Supports streaming responses with true token-level streaming
- [`tools/github_ingestor.py`](d:\Project\Python\GRAG\agent\tools\github_ingestor.py): GitHub data extraction (zero-crawler, API only)

**Knowledge Layer** ([`knowledge/`](d:\Project\Python\GRAG\knowledge\))
- [`lightrag_wrapper.py`](d:\Project\Python\GRAG\knowledge\lightrag_wrapper.py): LightRAG wrapper
  - Supports PostgreSQL/Neo4j/JSON storage
  - Hybrid retrieval: global (entities/relations) + local (chunks) + BM25
  - Custom entity/relation extraction prompts
  - Jupyter notebook support with nest_asyncio
- [`bm25_indexer.py`](d:\Project\Python\GRAG\knowledge\bm25_indexer.py): BM25 keyword search
  - RRF (Reciprocal Rank Fusion) for hybrid results
  - Configurable k1, b, epsilon parameters

**Model Layer** ([`models/`](d:\Project\Python\GRAG\models\))
- [`model_manager.py`](d:\Project\Python\GRAG\models\model_manager.py): Model management
  - **DeepSeek API** (primary)
  - **Ollama local models** (deprecated, code retained but disabled)
  - Automatic message format conversion
  - Health check and fallback mechanisms

**Storage Layer** ([`storage/`](d:\Project\Python\GRAG\storage\))
- [`factory.py`](d:\Project\Python\GRAG\storage\factory.py): Storage factory pattern
  - Creates storage instances based on configuration
  - Supports presets: `auto`, `local`, `docker`
- [`interface.py`](d:\Project\Python\GRAG\storage\interface.py): Storage interfaces
  - `ICacheStorage`, `IGraphStorage`, `IKnowledgeStorage`
- [`cache_manager.py`](d:\Project\Python\GRAG\storage\cache_manager.py): Cache management
  - LRU cleanup with quality scoring
  - PostgreSQL/JSON backends
  - User feedback integration
- Implementations:
  - [`json_cache_storage.py`](d:\Project\Python\GRAG\storage\json_cache_storage.py): JSON file cache
  - [`json_graph_storage.py`](d:\Project\Python\GRAG\storage\json_graph_storage.py): JSON graph storage
  - [`json_knowledge_storage.py`](d:\Project\Python\GRAG\storage\json_knowledge_storage.py): JSON knowledge storage

**Pipeline Scripts** ([`scripts/`](d:\Project\Python\GRAG\scripts\))
- **Pipeline**:
  - [`pipeline_fetch.py`](d:\Project\Python\GRAG\scripts\pipeline_fetch.py): Extract from GitHub
  - [`pipeline_clean.py`](d:\Project\Python\GRAG\scripts\pipeline_clean.py): Clean and preprocess
  - [`pipeline_ingest.py`](d:\Project\Python\GRAG\scripts\pipeline_ingest.py): Import to LightRAG
  - [`pipeline_context7_enhance.py`](d:\Project\Python\GRAG\scripts\pipeline_context7_enhance.py): Context7 enhancement
- **Testing**:
  - [`test_agent.py`](d:\Project\Python\GRAG\scripts\test_agent.py): Agent functionality test
  - [`test_query.py`](d:\Project\Python\GRAG\scripts\test_query.py): Query test
  - [`test_streaming.py`](d:\Project\Python\GRAG\scripts\test_streaming.py): Streaming test
  - [`test_diagnostics.py`](d:\Project\Python\GRAG\scripts\test_diagnostics.py): System diagnostics
  - [`check_db.py`](d:\Project\Python\GRAG\scripts\check_db.py): Database check

**Utilities** ([`utils/`](d:\Project\Python\GRAG\utils\))
- [`schema.py`](d:\Project\Python\GRAG\utils\schema.py): Pydantic data models
  - Pipeline: `RawDoc`, `IngestionBatch`, `CleanDoc`, `CleanBatch`
  - API: `QueryRequest`, `QueryResponse`, `FeedbackRequest`
  - Agent: `AgentState`, `IntentType`
  - Cache: `QueryCacheEntry`, `CacheStats`
  - Metadata: `DocMetadata`
- [`citation.py`](d:\Project\Python\GRAG\utils\citation.py): Citation system
  - Automatic source attribution
  - Citation validation and repair
  - Format: [1], [2], [3]
- [`monitoring.py`](d:\Project\Python\GRAG\utils\monitoring.py): Performance monitoring
  - Metrics collection
  - Performance tracking decorators
  - API call statistics
- [`context7_client.py`](d:\Project\Python\GRAG\utils\context7_client.py): Context7 API client
- [`logger.py`](d:\Project\Python\GRAG\utils\logger.py): Logging configuration
- [`encoding.py`](d:\Project\Python\GRAG\utils\encoding.py): UTF-8 encoding utilities

## Data Schemas (Pydantic Protocol)

All intermediate data products must be Pydantic models in [`utils/schema.py`](d:\Project\Python\GRAG\utils\schema.py):

### Pipeline Models
- **`RawDoc`**: Raw artifact from GitHub API
  - `path`: File path in repository
  - `content`: Original content (unmodified)
  - `source_url`: GitHub raw URL
  - `file_type`: markdown/notebook/text
  - `metadata`: Frontmatter and other metadata
  - `doc_id`: Deterministic ID (MD5 of source_url:path)
  - Methods: `save_to_file()`, `load_from_file()`

- **`IngestionBatch`**: Batch of raw documents
  - `batch_id`: Batch identifier
  - `repo_url`: Source repository
  - `docs`: List of `RawDoc`
  - Methods: `save_to_file()`, `load_from_file()`

- **`CleanDoc`**: Cleaned artifact for LightRAG
  - `doc_id`: Inherited from RawDoc
  - `file_path`: File path
  - `content`: Cleaned content
  - `source_url`: Original URL
  - `file_type`: File type
  - `metadata`: Enhanced metadata
  - Methods: `save_to_file()`, `load_from_file()`

- **`CleanBatch`**: Batch of cleaned documents
  - `batch_id`: Batch identifier
  - `source_url`: Source repository
  - `docs`: List of `CleanDoc`
  - `cleaned_at`: Cleaning timestamp
  - Methods: `save_to_file()`, `load_from_file()`

### API Models
- **`QueryRequest`**: Query request
  - `query`: Question text
  - `use_cache`: Whether to use cache
  - `stream`: Whether to stream response

- **`QueryResponse`**: Query response
  - `success`: Success status
  - `answer`: Generated answer
  - `context_ids`: Referenced document IDs
  - `context_metadata`: Full metadata with citations
  - `citations`: Citation list
  - `citation_info`: Citation statistics
  - `response_time`: Response time in seconds
  - `model_type`: Model used
  - `from_cache`: Whether from cache
  - `error`: Error message if failed

### Agent Models
- **`AgentState`**: LangGraph state
  - `messages`: Message history
  - `intent`: Recognized intent
  - `query`: Query text
  - `context_ids`: Retrieved context IDs
  - `context_metadata`: Context metadata
  - `answer`: Generated answer
  - `citations`: Citations
  - `error`: Error information

### Cache Models
- **`QueryCacheEntry`**: Cache entry structure
- **`CacheStats`**: Cache statistics

### Metadata Models
- **`DocMetadata`**: Standardized document metadata
  - `type`: File type
  - `url`: Source URL
  - `frontmatter`: Markdown frontmatter
  - `tags`: Document tags
  - `version`: Metadata version

## Configuration Management

- **Sensitive config**: `.env` file (API keys, passwords)
- **Non-sensitive config**: [`config/config.yaml`](d:\Project\Python\GRAG\config\config.yaml)
- **Environment variables**: Supported with `${VAR_NAME:default}` syntax
- **Presets**: `config/presets/local.yaml`, `config/presets/docker.yaml`
- **Access via**: `from config.config_manager import get_config`

### Key Config Sections
- `database.postgresql`: PostgreSQL connection settings
- `database.neo4j`: Neo4j connection settings
- `lightrag`: LightRAG settings
  - Entity/relation extraction prompts
  - Embedding model and provider (SiliconFlow/OpenAI)
  - BM25 configuration (enabled/disabled, k1, b, epsilon)
  - Storage backend (file/postgresql/neo4j)
- `models.api`: DeepSeek API configuration
- `models.qwen`: Qwen API configuration (Alibaba DashScope)
- `models.local`: Ollama configuration (deprecated)
- `model_switch`: Model priority and fallback settings
- `api`: FastAPI server settings
- `cache`: Cache configuration (LRU settings, quality thresholds)
- `context7`: Context7 API settings (optional enhancement)
- `logging`: Log level and output settings

## Code Style Standards

- **Python version**: 3.11+
- **Type annotations**: Required for all functions
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Imports order**: standard library → third-party → local modules

## Critical Development Principles

1. **File-First Development**: Pipeline steps use file storage for debugging
   - [`artifacts/01_raw/`](d:\Project\Python\GRAG\artifacts\01_raw\): Raw data from GitHub API
   - [`artifacts/02_clean/`](d:\Project\Python\GRAG\artifacts\02_clean\): Cleaned data for LightRAG
   - [`artifacts/03_graph/`](d:\Project\Python\GRAG\artifacts\03_graph\): Graph data (future use)

2. **Idempotency**: Use deterministic IDs (file path MD5) for repeatable operations
   - `doc_id = MD5(source_url:path)` ensures cross-repository uniqueness

3. **Zero-Crawler Strategy**: Only GitHub API as data source (no web scraping)
   - Complies with terms of service
   - Reliable data source
   - "Source Code is Truth"

4. **Pydantic as Protocol**: All data contracts are Pydantic models
   - Type safety and validation
   - Automatic serialization/deserialization
   - IDE support for auto-completion

5. **Always use `uv run`**: For executing any pipeline script
   - Ensures consistent environment
   - Automatic dependency management

## Common Patterns

### Storage Factory Pattern
```python
from storage.factory import create_storage_factory

# Auto-detect environment
factory = create_storage_factory(preset='auto')
cache_storage = factory.create_cache_storage()
graph_storage = factory.create_graph_storage()
```

### Pipeline Script Template
```python
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def ensure_uv_environment():
    """Check if running in uv environment"""
    if os.getenv('UV_PROJECT_ENVIRONMENT'):
        return True
    logger.warning("Use 'uv run' to execute this script")
```

### Error Handling
- **API Layer**: `raise HTTPException(status_code=..., detail=...)`
- **Business Layer**: Custom exceptions with logging
- **Always log errors**: `logger.error(...)` or `logger.exception(...)`

### Async in Jupyter
```python
if _is_jupyter():
    import nest_asyncio
    nest_asyncio.apply()
```

### Citation System
```python
from utils.citation import build_citation_prompt, post_process_answer

# Build prompt with citations
prompt = build_citation_prompt(context_metadata, query, format_type="number")

# Post-process answer to validate/fix citations
result = post_process_answer(answer, context_metadata, format_type="number", enable_fix=True)
```

### Performance Monitoring
```python
from utils.monitoring import track_performance

@track_performance("operation_name")
def my_function():
    # Function implementation
    pass
```

## Data Flow

### Query Request Flow
```
User Query
    ↓
FastAPI / Streamlit
    ↓
Cache Check (if enabled)
    ↓ (cache miss)
LangGraph Agent
    ↓
LightRAG Retrieval (hybrid: global + local + BM25)
    ├─ Global: Entity/relationship search
    ├─ Local: Document chunk search
    └─ BM25: Keyword search (optional)
    ↓
Model (DeepSeek/Qwen API)
    ↓
Citation Processing
    ↓
Response Generation
    ↓
Cache Storage (if enabled)
    ↓
User Response
```

### Knowledge Construction Flow
```
GitHub Repository
    ↓
GitHub API (zero-crawler)
    ↓
Raw Artifact (artifacts/01_raw/*.json)
    ├─ RawDoc: Original content
    └─ IngestionBatch: Batch metadata
    ↓
Cleaning Pipeline
    ├─ Remove HTML tags
    ├─ Clean Jupyter notebooks
    ├─ Extract frontmatter
    └─ Fix relative links
    ↓
Clean Artifact (artifacts/02_clean/*.json)
    ├─ CleanDoc: Cleaned content
    └─ CleanBatch: Batch metadata
    ↓
LightRAG Ingestion
    ├─ Entity extraction
    ├─ Relationship extraction
    └─ Chunking
    ↓
PostgreSQL / Neo4j / JSON Storage
    ├─ Vector index
    ├─ Graph index
    └─ Document status
```

## Data Artifacts

### Artifacts Directory Structure
```
artifacts/
├── 01_raw/              # Raw data from GitHub API
│   ├── {repo}_raw.json
│   └── test_*_raw.json
├── 02_clean/            # Cleaned data for LightRAG
│   ├── {repo}_clean.json
│   └── test_*_clean.json
├── 03_graph/            # Graph data (reserved for future use)
└── test_results/        # Pipeline test results
    └── benchmarks/
```

### RAG Storage Directory
```
rag_storage/
├── kv_store_*.json      # Key-value storage
├── graph_store_*.json   # Graph data
├── vector_store_*.json  # Vector embeddings
├── chunk_index_*.json   # Document chunks
└── lightrag_log.txt     # Processing logs
```

## Known Limitations

1. **Model Support**: Ollama local models deprecated (use DeepSeek/Qwen API)
2. **Data Sources**: Only GitHub repos supported (no web scraping)
3. **Prompt Tuning**: Entity/relation extraction prompts may need optimization
4. **Streaming**: Token-level streaming implemented but may need testing
5. **Graph Storage**: Neo4j support available but not extensively tested

## Model Support

### Currently Supported
- **DeepSeek API** ([`models/model_manager.py`](d:\Project\Python\GRAG\models\model_manager.py)): Primary model
  - `deepseek-chat`: General Q&A
  - `deepseek-coder`: Code-specific tasks

- **Qwen API** ([`models/model_manager.py`](d:\Project\Python\GRAG\models\model_manager.py)): Alibaba DashScope
  - `qwen-plus`: Balanced performance
  - `qwen-turbo`: Fast responses
  - `qwen-max`: High quality
  - `qwen-max-latest`: Latest version

### Deprecated
- **Ollama Local Models**: Code retained but disabled in config
  - Use API models for better performance
  - Code kept for potential future re-enabling

## Deployment Modes

### Local Mode (Development)
- **Storage**: JSON files in `rag_storage/` and `data/cache/`
- **Config Preset**: `config/presets/local.yaml`
- **Environment**: No additional setup required
- **Use Case**: Development, testing, small-scale deployments

### Docker Mode (Production)
- **Storage**: PostgreSQL + Neo4j
- **Config Preset**: `config/presets/docker.yaml`
- **Environment**: Requires Docker and Docker Compose
- **Use Case**: Production deployments, multi-user scenarios

### Auto Mode (Default)
- Automatically detects environment
- Uses Docker mode if `DOCKER` environment variable set
- Falls back to Local mode otherwise

## TODO Priorities

### Phase 1: Foundation (Current)
- ✅ Complete pipeline testing
- ✅ LightRAG validation
- ✅ Multi-model support (DeepSeek + Qwen)
- ✅ Citation system implementation

### Phase 2: Optimization (Next)
- 🔄 Refactor Agent layer (remove deprecated code)
- 🔄 BM25 performance tuning
- 🔄 Entity/relation extraction prompt optimization
- ⏳ Comprehensive test coverage

### Phase 3: Integration (Future)
- ⏳ Context7 integration
- ⏳ Advanced analytics dashboard
- ⏳ Multi-repository support
- ⏳ Real-time collaboration features

### Phase 4: Production (Future)
- ⏳ File-driven → API-driven pipeline transformation
- ⏳ Distributed caching (Redis)
- ⏳ Advanced monitoring and alerting
- ⏳ Multi-tenant support
