# Handoff - GRAG Agent层数据协议问题

## ✅ 已修复：context_metadata数据流

### 当前状态（已验证）
数据流完整：`LightRAGWrapper` → `Agent` → `API` → `QueryResponse`

### 缓存管理
**工具**: `scripts/clear_cache.py`

**清除LLM缓存**:
```bash
uv run scripts/clear_cache.py --llm
```

**效果**:
- 清除条目: 2064
- 释放空间: 27.88 MB
- 备份位置: `rag_storage/kv_store_llm_response_cache.json.backup_<timestamp>`

**缓存文件**: `rag_storage/kv_store_llm_response_cache.json`

### 各层实现

#### 1. LightRAGWrapper层 (`knowledge/lightrag_wrapper.py`)

**位置1**: `_query_lightrag()` 方法 (行1015-1025)
```python
context_metadata.append({
    'index': idx,
    'chunk_id': chunk_id,
    'content': content,
    'source': {
        'file_path': chunk.get('file_path', ''),
        'source_url': chunk.get('source_url', ''),
        'doc_id': chunk.get('doc_id', ''),
        'title': chunk.get('title', '')
    }
})
```

**位置2**: `_query_hybrid_bm25()` 方法 (行1156-1165)
```python
context_metadata.append({
    'index': idx,
    'chunk_id': r.get('doc_id', ''),
    'content': r.get('content', ''),
    'source': {
        'file_path': r.get('file_path', r.get('metadata', {}).get('file_path', '')),
        ...
    }
})
```

**返回** (行1058):
```python
return {
    'answer': answer,
    'contexts': contexts,
    'context_metadata': context_metadata,  # ✅ 包含
    ...
}
```

#### 2. Agent层 (`agent/grag_agent.py`)

**位置**: `query()` 方法 (行637-645)
```python
response_data = {
    "success": True,
    "answer": final_state.get("answer", ""),
    "context_ids": final_state.get("context_ids", []),
    "context_metadata": final_state.get("context_metadata", []),  # ✅ 新增
    ...
}

# Pydantic 验证 (行648-663)
validated = QueryResponse(
    success=response_data["success"],
    answer=response_data["answer"],
    context_ids=response_data["context_ids"],
    context_metadata=response_data["context_metadata"],  # ✅ 传递
    ...
)
```

#### 3. API层 (`api/routes.py`)

**位置**: `/query` 端点 (行132-142)
```python
return QueryResponse(
    success=True,
    answer=result["answer"],
    context_ids=result.get("context_ids", []),
    context_metadata=result.get("context_metadata", []),  # ✅ 包含
    ...
)
```

#### 4. Schema层 (`utils/schema.py`)

**位置**: `QueryResponse` (行266-278)
```python
class QueryResponse(BaseModel):
    success: bool
    answer: str
    context_ids: List[str]
    context_metadata: List[Dict[str, Any]]  # ✅ 已定义
    response_time: float
    model_type: str
    from_cache: bool
    ...
```

## ⚠️ 已知问题

### 1. LightRAG chunks字段不完整
- **实际返回**: `content`, `chunk_id`, `reference_id`, `file_path`
- **缺失字段**: `source_url`, `doc_id`, `title`（为空字符串）
- **影响**: `context_metadata.source` 部分字段为空，但不影响基本功能
- **原因**: LightRAG的 `query_data()` 未返回完整元数据

### 2. Windows 代理配置导致 API 超时
- **症状**: 查询请求超时（60秒），Health 端点正常但 Query 端点超时
- **日志**: `ConnectTimeout` 或 `Failed to connect to 127.0.0.1:7897`
- **根本原因**: Windows 注册表中的 `ProxyServer=127.0.0.1:7897`，即使 `ProxyEnable=0x0`（禁用），Python httpx 库仍会尝试使用
- **诊断命令**:
  ```bash
  reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" | findstr -i "proxy"
  ```
- **解决方案**:
  - 方案 A: 清除注册表 `reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer /f`
  - 方案 B: 在 .env 中添加 `NO_PROXY=*`
- **详细说明**: 参见 `.rules/grag_rules.md` 第13节

### 3. 缓存管理
**已解决**:
- ✅ 创建 `scripts/clear_cache.py` 清除工具
- ✅ 支持清除LLM响应缓存（28 MB → 2 bytes）
- ✅ 自动备份缓存文件

**使用方法**:
```bash
# 清除LLM缓存
uv run scripts/clear_cache.py --llm

# 清除所有缓存
uv run scripts/clear_cache.py --all
```

**注意**: API服务内存缓存仍需重启服务清空

## 📋 待办

1. **完善查询缓存清除** - 扩展`clear_cache.py`支持PostgreSQL查询缓存
2. **优化缓存策略** - 仅在用户反馈"有帮助"时启用缓存
3. **完善元数据** - 研究如何从LightRAG获取完整的`doc_id`, `title`等字段

## 🔍 验证方法

### context_metadata验证
```python
# 1. 直接测试LightRAGWrapper
wrapper.query(query, mode="hybrid", top_k=5)
# 预期: context_metadata = List[20]

# 2. API测试（新查询）
requests.post("/api/v1/query", json={"query": "新问题", "use_cache": False})
# 预期: context_metadata = List[20]

# 3. 检查Pydantic验证
# 查看日志: "Agent 响应已通过 Pydantic 验证"
```

### 缓存清除验证
```bash
# 清除前
ls -lh rag_storage/kv_store_llm_response_cache.json
# 输出: -rw-r--r-- 1 Admin 197121 28M ...

# 清除后
uv run scripts/clear_cache.py --llm

# 验证
ls -lh rag_storage/kv_store_llm_response_cache.json
# 输出: -rw-r--r-- 1 Admin 197121 2  ...
```
