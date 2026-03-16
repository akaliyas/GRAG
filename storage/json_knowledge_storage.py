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

        # LightRAG 存储文件路径
        self._kv_store_file = self.data_dir / f"kv_store_{namespace}.json"
        self._doc_status_file = self.data_dir / f"kv_store_full_docs_{namespace}.json"
        self._chunk_store_file = self.data_dir / f"kv_store_chunked_entity_{namespace}.json"
        self._graphml_file = self.data_dir / f"graph_{namespace}.graphml"

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
        try:
            import networkx as nx
            if self._graphml_file.exists():
                graph = nx.read_graphml(str(self._graphml_file))
                return graph.number_of_nodes()
        except Exception as e:
            logger.error(f"读取图数据失败: {e}")
        return 0

    def get_relation_count(self) -> int:
        """获取关系数量"""
        try:
            import networkx as nx
            if self._graphml_file.exists():
                graph = nx.read_graphml(str(self._graphml_file))
                return graph.number_of_edges()
        except Exception as e:
            logger.error(f"读取图数据失败: {e}")
        return 0

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
