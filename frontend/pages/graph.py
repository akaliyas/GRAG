"""
知识图谱可视化页面

展示 LightRAG 构建的知识图谱，支持：
- 实体和关系可视化
- 图统计信息
- 实体搜索
- 图导出
"""
import logging
import streamlit as st
import requests
import json
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# API 配置
API_BASE_URL = "http://localhost:8000/api/v1"

# 安全获取 secrets
try:
    API_USERNAME = st.secrets["API_USERNAME"]
except (FileNotFoundError, KeyError):
    API_USERNAME = "admin"

try:
    API_PASSWORD = st.secrets["API_PASSWORD"]
except (FileNotFoundError, KeyError):
    API_PASSWORD = ""

logger = logging.getLogger(__name__)


def get_auth_headers():
    """获取认证头"""
    import base64
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def fetch_graph_stats() -> Optional[Dict[str, Any]]:
    """获取图统计信息"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/graph/stats",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("data")
    except Exception as e:
        logger.error(f"获取图统计失败: {e}")
    return None


def fetch_entities(limit: int = 100, keyword: str = None) -> List[Dict[str, Any]]:
    """获取实体列表"""
    try:
        params = {"limit": limit}
        if keyword:
            params["keyword"] = keyword

        response = requests.get(
            f"{API_BASE_URL}/graph/entities",
            params=params,
            headers=get_auth_headers(),
            timeout=10
        )
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("entities", [])
    except Exception as e:
        logger.error(f"获取实体列表失败: {e}")
    return []


def fetch_relationships(limit: int = 100) -> List[Dict[str, Any]]:
    """获取关系列表"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/graph/relationships",
            params={"limit": limit},
            headers=get_auth_headers(),
            timeout=10
        )
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("relationships", [])
    except Exception as e:
        logger.error(f"获取关系列表失败: {e}")
    return []


def fetch_graph_data(format: str = "json") -> Optional[Dict[str, Any]]:
    """导出图数据"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/graph/export",
            params={"format": format},
            headers=get_auth_headers(),
            timeout=30
        )
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("data")
    except Exception as e:
        logger.error(f"导出图数据失败: {e}")
    return None


def create_network_graph(entities: List[Dict], relationships: List[Dict], max_nodes: int = 50):
    """
    使用 Plotly 创建交互式网络图

    Args:
        entities: 实体列表
        relationships: 关系列表
        max_nodes: 最大显示节点数
    """
    if not entities or not relationships:
        st.info("暂无图数据可显示")
        return

    # 限制节点数量
    entities = entities[:max_nodes]

    # 创建节点映射
    entity_ids = {e["entity_id"] for e in entities}
    entity_names = {e["entity_id"]: e["entity_name"] for e in entities}

    # 过滤有效关系
    valid_relationships = [
        r for r in relationships
        if r.get("source") in entity_ids and r.get("target") in entity_ids
    ][:max_nodes * 2]

    if not valid_relationships:
        st.info("没有可显示的关系")
        return

    # 创建 NetworkX 图
    G = nx.Graph()

    # 添加节点
    for entity in entities:
        G.add_node(
            entity["entity_id"],
            label=entity["entity_name"],
            type=entity.get("entity_type", "unknown"),
            description=entity.get("description", "")
        )

    # 添加边
    for rel in valid_relationships:
        G.add_edge(
            rel["source"],
            rel["target"],
            relation=rel.get("relation_type", "related"),
            weight=rel.get("weight", 1.0)
        )

    # 计算布局
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except Exception as e:
        logger.error(f"计算布局失败: {e}")
        st.error("无法生成图布局")
        return

    # 提取边和节点信息
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get("relation", "related"))

    node_x = []
    node_y = []
    node_text = []
    node_info = []

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[1].get("label", node[0]))
        node_info.append(
            f"**{node[1].get('label')}**<br>"
            f"类型: {node[1].get('type')}<br>"
            f"描述: {node[1].get('description', 'N/A')[:100]}"
        )

    # 创建边轨迹
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 创建节点轨迹
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        hovertext=node_info,
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )

    # 创建图
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='知识图谱可视化',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[
                            dict(
                                text="知识图谱 - LightRAG",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(size=12, color='#888')
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    st.plotly_chart(fig, use_container_width=True)


def show():
    """显示知识图谱页面"""

    st.title("🕸️ 知识图谱")

    # 加载图统计
    with st.spinner("加载图数据..."):
        stats = fetch_graph_stats()

        if stats is None:
            st.error("无法加载图数据，请确认后端服务正在运行")
            return

    # 显示统计信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("实体数", stats.get("entity_count", 0))
    with col2:
        st.metric("关系数", stats.get("relation_count", 0))
    with col3:
        st.metric("图密度", f"{stats.get('density', 0):.3f}")
    with col4:
        st.metric("连通性", "是" if stats.get("is_connected", False) else "否")

    st.markdown("---")

    # 控制面板
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        max_nodes = st.slider("最大节点数", 10, 200, 50, 10)

    with col2:
        search_keyword = st.text_input("搜索实体", placeholder="输入关键词...")

    with col3:
        export_format = st.selectbox("导出格式", ["json", "gml"])

    # 搜索和过滤
    if search_keyword:
        entities = fetch_entities(limit=max_nodes, keyword=search_keyword)
    else:
        entities = fetch_entities(limit=max_nodes)

    relationships = fetch_relationships(limit=max_nodes * 2)

    # 数据导出按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 导出图数据", use_container_width=True):
            data = fetch_graph_data(format=export_format)
            if data:
                st.download_button(
                    label=f"下载 {export_format.upper()} 文件",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name=f"knowledge_graph.{export_format}",
                    mime="application/json"
                )
            else:
                st.error("导出失败")

    with col2:
        if st.button("🔄 刷新数据", use_container_width=True):
        st.rerun()

    st.markdown("---")

    # 图可视化
    st.subheader("图谱可视化")

    if entities and relationships:
        create_network_graph(entities, relationships, max_nodes)

        # 显示实体和关系表格
        with st.expander("📋 实体列表"):
            st.dataframe(
                entities,
                column_config={
                    "entity_id": "ID",
                    "entity_name": "名称",
                    "entity_type": "类型",
                    "description": "描述",
                    "degree": "连接数"
                },
                use_container_width=True
            )

        with st.expander("🔗 关系列表"):
            st.dataframe(
                relationships,
                column_config={
                    "source": "源实体",
                    "target": "目标实体",
                    "relation_type": "关系类型",
                    "weight": "权重"
                },
                use_container_width=True
            )
    else:
        st.info("暂无数据，请先运行数据管道导入文档")


# Streamlit 多页面应用会自动执行此文件
show()
