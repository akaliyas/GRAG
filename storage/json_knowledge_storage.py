"""
JSON 知识库存储实现 - 提供知识库统计和文档管理

从 LightRAG 的 JSON 存储文件读取元数据，
提供文档、块、实体、关系统计信息。
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from storage.interface import IKnowledgeStorage

logger = logging.getLogger(__name__)


class JSONKnowledgeStorage(IKnowledgeStorage):
    """
    JSON 知识库存储实现

    从 LightRAG 的 JSON 文件读取知识库元数据，
    提供统计信息和文档管理功能。
    适用于本地文件系统部署模式。
    """

    def __init__(self, data_dir: str = "./rag_storage", namespace: str = "default"):
        """
        初始化 JSON 知识库存储

        Args:
            data_dir: LightRAG 工作目录
            namespace: 命名空间（用于多租户隔离）
        """
        self.data_dir = Path(data_dir)
        self.namespace = namespace

        # LightRAG 存储文件路径（根据实际文件名）
        self._kv_store_file = self.data_dir / "kv_store_llm_response_cache.json"
        self._full_docs_file = self.data_dir / "kv_store_full_docs.json"
        self._doc_status_file = self.data_dir / "kv_store_doc_status.json"
        self._chunk_store_file = self.data_dir / "kv_store_text_chunks.json"
        self._entity_chunks_file = self.data_dir / "kv_store_entity_chunks.json"
        self._relation_chunks_file = self.data_dir / "kv_store_relation_chunks.json"
        self._full_entities_file = self.data_dir / "kv_store_full_entities.json"
        self._full_relations_file = self.data_dir / "kv_store_full_relations.json"
        self._graphml_file = self.data_dir / "graph_chunk_entity_relation.graphml"

        logger.info(f"JSON 知识库存储初始化: {self.data_dir}")

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """加载 JSON 文件"""
        if not file_path.exists():
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载 JSON 文件失败 {file_path}: {e}")
            return {}

    def get_doc_count(self) -> int:
        """获取已导入文档数量"""
        data = self._load_json_file(self._doc_status_file)
        # 统计文档数量（去重）
        docs = set()
        for value in data.values():
            if isinstance(value, dict) and "source_id" in value:
                docs.add(value["source_id"])
        return len(docs)

    def get_chunk_count(self) -> int:
        """获取文档块数量"""
        data = self._load_json_file(self._chunk_store_file)
        return len(data)

    def get_entity_count(self) -> int:
        """获取实体数量"""
        data = self._load_json_file(self._full_entities_file)
        return len(data)

    def get_relation_count(self) -> int:
        """获取关系数量"""
        data = self._load_json_file(self._full_relations_file)
        return len(data)

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出所有文档

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            文档列表，每个包含 doc_id, title, source, created_at 等
        """
        data = self._load_json_file(self._doc_status_file)

        documents = []
        seen_sources = {}  # 用于去重，记录每个 source 的最新文档

        for doc_id, doc_data in data.items():
            if not isinstance(doc_data, dict):
                continue

            source_id = doc_data.get("source_id", "")
            if not source_id:
                continue

            # 记录或更新文档信息
            if source_id not in seen_sources:
                seen_sources[source_id] = {
                    "doc_id": doc_id,
                    "source": source_id,
                    "title": doc_data.get("title", source_id),
                    "status": doc_data.get("status", "unknown"),
                    "created_at": doc_data.get("created_at", ""),
                    "updated_at": doc_data.get("updated_at", ""),
                }

        # 应用分页
        all_docs = list(seen_sources.values())
        paginated_docs = all_docs[offset:offset + limit]

        return paginated_docs

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        获取知识库综合统计信息

        Returns:
            统计信息字典
        """
        return {
            "doc_count": self.get_doc_count(),
            "chunk_count": self.get_chunk_count(),
            "entity_count": self.get_entity_count(),
            "relation_count": self.get_relation_count(),
            "storage_dir": str(self.data_dir),
            "namespace": self.namespace,
        }

    def search_documents(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        搜索文档（按标题或路径匹配）

        Args:
            keyword: 搜索关键词
            limit: 返回数量限制

        Returns:
            匹配的文档列表
        """
        all_docs = self.list_documents(limit=1000)
        keyword_lower = keyword.lower()

        matched = [
            doc for doc in all_docs
            if keyword_lower in doc.get("title", "").lower()
            or keyword_lower in doc.get("source", "").lower()
        ]

        return matched[:limit]

    def get_source_summary(self) -> str:
        """
        获取知识库来源摘要

        用于RAG注入，让LLM能够回答"知识库包含什么"这类问题。

        Returns:
            格式化的文本描述
        """
        try:
            from collections import Counter
            import re

            # 读取完整文档数据（包含source_url）
            full_docs_data = self._load_json_file(self._full_docs_file)

            if not full_docs_data:
                return "当前知识库为空，尚未导入任何文档。"

            # 统计来源
            sources = Counter()
            total_docs = 0

            for doc_id, doc_data in full_docs_data.items():
                if not isinstance(doc_data, dict):
                    continue

                total_docs += 1

                # 获取source_url
                source_url = doc_data.get("source_url", "")

                if source_url and "github.com" in source_url:
                    # 提取仓库名
                    match = re.search(r'github\.com/([^/]+/[^/]+)', source_url)
                    if match:
                        repo = f"github.com/{match.group(1)}"
                        sources[repo] += 1
                elif source_url:
                    # 非GitHub来源，使用域名
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(source_url)
                        domain = parsed.netloc
                        if domain:
                            sources[domain] += 1
                    except Exception:
                        sources[source_url[:50]] += 1

            if not sources:
                return f"当前知识库包含 {total_docs} 个文档，但缺少来源信息。"

            # 生成自然语言描述
            summary_parts = [f"当前知识库包含 {len(sources)} 个来源，共 {total_docs} 个文档：\n"]

            for source, count in sources.most_common():
                summary_parts.append(f"- {source} ({count} 个文档)")

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"生成来源摘要失败: {e}")
            return "无法获取知识库来源信息。"
