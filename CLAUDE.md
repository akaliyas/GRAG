# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRAG (Graph Retrieval Augmented Generation) is a technical document intelligent Q&A system based on knowledge graph enhanced retrieval generation. It uses LightRAG framework for low-cost, high-performance document Q&A with dual-retrieval mechanism (global + local retrieval).

**Core Architecture Philosophy**: "Pipeline for Write, Service for Read" - two decoupled subsystems:
- **Knowledge Construction Pipeline** (file-driven, slow, debuggable): GitHub API → artifacts/01_raw → cleaning → artifacts/02_clean → LightRAG ingestion → PostgreSQL indexing
- **Online Q&A Service** (memory-driven, fast, read-only): User query → FastAPI/Streamlit → LangGraph Agent → Model → Knowledge Storage

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
```

## Architecture & Key Components

### Initialization Order (api/main.py:52)
1. ModelManager → 2. LightRAGWrapper → 3. GRAGAgent → 4. CacheManager

### Core Layers

**API Layer** (`api/`)
- `main.py`: FastAPI entry point, handles system startup/shutdown
- `routes.py`: RESTful API endpoints (`/api/v1/query`, `/api/v1/feedback`, `/api/v1/model/switch`)
- `auth.py`: HTTP Basic authentication

**Agent Layer** (`agent/`)
- `grag_agent.py`: LangGraph-based agent for query processing
- **IMPORTANT**: Agent only handles retrieval and answer generation. Data ingestion happens in offline pipeline scripts.
- Intent types: `QUERY` only (deprecated: `GITHUB_INGEST`, `PREPROCESS`)

**Knowledge Layer** (`knowledge/`)
- `lightrag_wrapper.py`: LightRAG wrapper supporting PostgreSQL/Neo4j storage
- Dual retrieval: global (entities/relations) + local (document chunks)

**Model Layer** (`models/`)
- `model_manager.py`: Handles model switching (API/local) with fallback
- Supports DeepSeek API and local models (Ollama/vLLM)

**Storage Layer** (`storage/`)
- `cache_manager.py`: PostgreSQL caching with LRU cleanup + quality scoring

**Pipeline Scripts** (`scripts/`)
- `pipeline_fetch.py`: Extract docs from GitHub repo
- `pipeline_clean.py`: Clean and preprocess data
- `pipeline_ingest.py`: Import data into LightRAG

## Data Schemas (Pydantic Protocol)

All intermediate data products must be Pydantic models in `utils/schema.py`:
- `RawDoc` / `IngestionBatch`: Raw artifact from GitHub API
- `CleanDoc` / `CleanBatch`: Cleaned artifact for LightRAG input

**Required methods**: `save_to_file(path: str)` and `load_from_file(path: str)`

## Configuration Management

- **Sensitive config**: `.env` file (API keys, passwords)
- **Non-sensitive config**: `config/config.yaml`
- **Environment variables**: Supported with `${VAR_NAME:default}` syntax
- **Access via**: `from config.config_manager import get_config`

### Key Config Sections
- `database.postgresql`: PostgreSQL connection
- `lightrag`: LightRAG settings including entity/relation extraction prompts
- `models.api`: DeepSeek API configuration
- `api`: FastAPI server settings

## Code Style Standards

- **Python version**: 3.11+
- **Type annotations**: Required for all functions
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Imports order**: standard library → third-party → local modules

## Critical Development Principles

1. **File-First Development**: Pipeline steps use file storage for debugging
2. **Idempotency**: Use deterministic IDs (file path MD5) for repeatable operations
3. **Zero-Crawler Strategy**: Only GitHub API as data source (no web scraping)
4. **Pydantic as Protocol**: All data contracts are Pydantic models
5. **Always use `uv run`**: For executing any pipeline script

## Common Patterns

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
- API Layer: `raise HTTPException(status_code=..., detail=...)`
- Business Layer: Custom exceptions with logging
- Always log errors: `logger.error(...)` or `logger.exception(...)`

### Async in Jupyter
```python
if _is_jupyter():
    import nest_asyncio
    nest_asyncio.apply()
```

## Data Flow

**Query Request**:
User → API → Cache Check → Agent (LangGraph) → LightRAG Retrieval → Model → Response → Cache Storage

**Knowledge Construction**:
GitHub API → Raw Artifact → Cleaning → Clean Artifact → LightRAG Ingestion → PostgreSQL

## Known Limitations

1. Stream response is currently simulated (needs implementation)
2. LightRAG entity/relation extraction prompts need tuning
3. Only GitHub repos supported as data source
4. Local model (Ollama) temporarily disabled in config

## TODO Priorities

From README.md:750-815
- Phase 1: Complete pipeline testing and LightRAG validation
- Phase 2: Simplify Agent layer (remove deprecated sync routes)
- Phase 3: System integration and optimization
- Phase 4: Production transformation (file-driven → API-driven)
