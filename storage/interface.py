"""
存储抽象层 - 定义存储接口

此模块定义了所有存储后端的通用接口，使核心业务逻辑
与具体存储实现解耦，支持多种部署方式。

支持的存储实现：
- JSON 文件存储（本地应用）
- PostgreSQL 存储（Docker/云部署）
- Redis 存储（高性能场景）
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime


class ICacheStorage(ABC):
    """
    缓存存储接口

    提供查询缓存的基本 CRUD 操作，支持 LRU 清理和质量评分。
    """

    @abstractmethod
    def get_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取查询缓存

        Args:
            query: 查询文本

        Returns:
            缓存数据字典，包含 answer, context_ids 等；未找到返回 None
        """
        pass

    @abstractmethod
    def set_cache(
        self,
        query: str,
        answer: str,
        context_ids: List[str],
        model_type: str,
        response_time: float,
        context_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        设置查询缓存

        Args:
            query: 查询文本
            answer: 答案内容
            context_ids: 上下文 ID 列表
            model_type: 模型类型
            response_time: 响应时间（秒）
            context_metadata: 完整上下文元数据（可选）

        Returns:
            设置成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def update_feedback(self, query: str, is_positive: bool) -> bool:
        """
        更新用户反馈

        Args:
            query: 查询文本
            is_positive: 是否为正面反馈

        Returns:
            更新成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def cleanup_lru(self, max_size: int, batch_size: int) -> int:
        """
        LRU 清理：删除最久未访问的缓存条目

        Args:
            max_size: 最大缓存条目数
            batch_size: 每次清理的条目数

        Returns:
            清理的条目数
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典，包含 total_count, total_size 等
        """
        pass

    @abstractmethod
    def clear_all(self) -> bool:
        """
        清空所有缓存

        Returns:
            清空成功返回 True，失败返回 False
        """
        pass


class IGraphStorage(ABC):
    """
    图存储接口（用于图可视化）

    提供图数据的导出功能，支持多种可视化工具。
    """

    @abstractmethod
    def export_graph_gml(self) -> str:
        """
        导出图数据为 GML 格式

        Returns:
            GML 格式的图数据字符串
        """
        pass

    @abstractmethod
    def export_graph_json(self) -> Dict[str, Any]:
        """
        导出图数据为 JSON 格式（用于前端可视化）

        Returns:
            包含 nodes 和 edges 的字典
        """
        pass

    @abstractmethod
    def get_entities(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取实体列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            实体字典列表，每个包含 entity_name, entity_type, description 等
        """
        pass

    @abstractmethod
    def get_relationships(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取关系列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            关系字典列表，每个包含 source, target, relation_type 等
        """
        pass

    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图统计信息

        Returns:
            统计信息字典，包含 entity_count, relation_count 等
        """
        pass


class IKnowledgeStorage(ABC):
    """
    知识库存储接口（LightRAG 抽象）

    虽然 LightRAG 内部已有存储抽象，但此接口用于
    获取知识库的元数据和统计信息。
    """

    @abstractmethod
    def get_doc_count(self) -> int:
        """获取已导入文档数量"""
        pass

    @abstractmethod
    def get_chunk_count(self) -> int:
        """获取文档块数量"""
        pass

    @abstractmethod
    def get_entity_count(self) -> int:
        """获取实体数量"""
        pass

    @abstractmethod
    def get_relation_count(self) -> int:
        """获取关系数量"""
        pass

    @abstractmethod
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出所有文档

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            文档列表，每个包含 doc_id, title, source, created_at 等
        """
        pass

    @abstractmethod
    def get_source_summary(self) -> str:
        """
        获取知识库来源摘要

        用于RAG注入，让LLM能够回答"知识库包含什么"这类问题。

        Returns:
            格式化的文本描述，例如：
            "当前知识库包含 2 个来源：
            1. github.com/user/repo-a (125 documents)
            2. github.com/user/repo-b (89 documents)"
        """
        pass


class StorageBackend:
    """存储后端类型枚举"""
    JSON = "json"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    NEO4J = "neo4j"
