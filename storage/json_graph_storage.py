"""
JSON 图存储实现 - 用于图可视化

从 NetworkX 生成的 GraphML 文件读取图数据，
并将其转换为 JSON 格式供前端可视化使用。
"""
import json
import logging
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.interface import IGraphStorage

logger = logging.getLogger(__name__)


class JSONGraphStorage(IGraphStorage):
    """
    JSON 图存储实现

    从 LightRAG 的 NetworkX 图文件读取数据，提供 JSON 格式导出。
    适用于本地文件系统部署模式。
    """

    def __init__(self, data_dir: str = "./rag_storage", namespace: str = "default"):
        """
        初始化 JSON 图存储

        Args:
            data_dir: LightRAG 工作目录
            namespace: 命名空间（用于多租户隔离）
        """
        self.data_dir = Path(data_dir)
        self.namespace = namespace
        self._graphml_file = self.data_dir / f"graph_{namespace}.graphml"
        self._graph: Optional[nx.Graph] = None

        logger.info(f"JSON 图存储初始化: {self._graphml_file}")

        # 加载图数据
        self._load_graph()

    def _load_graph(self) -> None:
        """从 GraphML 文件加载图数据"""
        if self._graphml_file.exists():
            try:
                self._graph = nx.read_graphml(str(self._graphml_file))
                logger.info(f"成功加载图数据: {self._graph.number_of_nodes()} 个节点, "
                           f"{self._graph.number_of_edges()} 条边")
            except Exception as e:
                logger.error(f"加载图数据失败: {e}")
                self._graph = nx.Graph()
        else:
            logger.warning(f"图文件不存在: {self._graphml_file}")
            self._graph = nx.Graph()

    def export_graph_gml(self) -> str:
        """
        导出图数据为 GML 格式

        Returns:
            GML 格式的图数据字符串
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return ""

        try:
            # 使用 NetworkX 的 generate_gml
            return nx.generate_gml(self._graph)
        except Exception as e:
            logger.error(f"导出 GML 格式失败: {e}")
            return ""

    def export_graph_json(self) -> Dict[str, Any]:
        """
        导出图数据为 JSON 格式（用于前端可视化）

        Returns:
            包含 nodes 和 edges 的字典，格式兼容 D3.js、ECharts 等可视化库
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return {"nodes": [], "edges": []}

        try:
            nodes = []
            edges = []

            # 导出节点
            for node_id, node_data in self._graph.nodes(data=True):
                node = {
                    "id": str(node_id),
                    "name": str(node_id),
                    # 尝试提取常见属性
                    "entity_type": node_data.get("entity_type", "unknown"),
                    "description": node_data.get("description", ""),
                    "source_id": node_data.get("source_id", ""),
                    # 额外属性
                    "degree": self._graph.degree(node_id),
                }
                nodes.append(node)

            # 导出边
            for source, target, edge_data in self._graph.edges(data=True):
                edge = {
                    "source": str(source),
                    "target": str(target),
                    "relation": edge_data.get("relation", "related"),
                    "weight": edge_data.get("weight", 1.0),
                    "keywords": edge_data.get("keywords", ""),
                }
                edges.append(edge)

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                }
            }

        except Exception as e:
            logger.error(f"导出 JSON 格式失败: {e}")
            return {"nodes": [], "edges": []}

    def get_entities(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取实体列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            实体字典列表
        """
        if self._graph is None:
            return []

        entities = []
        for i, (node_id, node_data) in enumerate(self._graph.nodes(data=True)):
            if i < offset:
                continue
            if len(entities) >= limit:
                break

            entity = {
                "entity_id": str(node_id),
                "entity_name": str(node_id),
                "entity_type": node_data.get("entity_type", "unknown"),
                "description": node_data.get("description", ""),
                "source_id": node_data.get("source_id", ""),
                "degree": self._graph.degree(node_id),
            }
            entities.append(entity)

        return entities

    def get_relationships(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取关系列表

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            关系字典列表
        """
        if self._graph is None:
            return []

        relationships = []
        for i, (source, target, edge_data) in enumerate(self._graph.edges(data=True)):
            if i < offset:
                continue
            if len(relationships) >= limit:
                break

            relation = {
                "source": str(source),
                "target": str(target),
                "relation_type": edge_data.get("relation", "related"),
                "weight": edge_data.get("weight", 1.0),
                "keywords": edge_data.get("keywords", ""),
            }
            relationships.append(relation)

        return relationships

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图统计信息

        Returns:
            统计信息字典
        """
        if self._graph is None:
            return {
                "entity_count": 0,
                "relation_count": 0,
                "node_count": 0,
                "edge_count": 0,
            }

        return {
            "entity_count": self._graph.number_of_nodes(),
            "relation_count": self._graph.number_of_edges(),
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            # 图密度
            "density": nx.density(self._graph) if self._graph.number_of_nodes() > 0 else 0,
            # 是否连通
            "is_connected": nx.is_connected(self._graph) if self._graph.number_of_nodes() > 0 else True,
        }

    def reload(self) -> None:
        """重新加载图数据（用于数据更新后刷新）"""
        self._load_graph()
