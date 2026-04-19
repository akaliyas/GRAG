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

# 导入配置
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.config import API_BASE_URL, API_USERNAME, API_PASSWORD

logger = logging.getLogger(__name__)


def get_auth_headers():
    """获取认证头"""
    import base64
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def fetch_graph_stats() -> Optional[Dict[str, Any]]:
    """获取图统计信息（带重试机制）"""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
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
                else:
                    logger.warning(f"API返回失败: {data.get('error', 'Unknown error')}")
                    return None
            else:
                logger.warning(f"API请求失败: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
        except requests.exceptions.ConnectionError:
            logger.warning(f"连接失败 (尝试 {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.error(f"请求异常: {e}")
        except Exception as e:
            logger.error(f"获取图统计失败: {e}")
            break

        if attempt < max_retries - 1:
            import time
            time.sleep(retry_delay)

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
    # 输入验证
    if not entities:
        st.info("暂无实体数据可显示")
        return

    if not relationships:
        st.info("暂无关系数据可显示")
        # 仍然显示实体节点
        relationships = []

    # 限制节点数量并验证实体数据结构
    valid_entities = []
    seen_ids = set()

    for entity in entities[:max_nodes]:
        if not isinstance(entity, dict):
            logger.warning(f"跳过无效实体: {entity}")
            continue

        entity_id = entity.get("entity_id")
        if not entity_id:
            logger.warning(f"跳过缺少 entity_id 的实体: {entity}")
            continue

        if entity_id in seen_ids:
            logger.warning(f"跳过重复实体 ID: {entity_id}")
            continue

        seen_ids.add(entity_id)
        valid_entities.append(entity)

    if not valid_entities:
        st.warning("没有有效的实体数据")
        return

    entities = valid_entities

    # 创建节点映射
    entity_ids = {e["entity_id"] for e in entities}
    entity_names = {e["entity_id"]: e.get("entity_name", e["entity_id"]) for e in entities}

    # 过滤有效关系
    valid_relationships = []
    seen_relations = set()

    for rel in relationships[:max_nodes * 2]:
        if not isinstance(rel, dict):
            logger.warning(f"跳过无效关系: {rel}")
            continue

        source = rel.get("source")
        target = rel.get("target")

        if not source or not target:
            logger.warning(f"跳过缺少 source/target 的关系: {rel}")
            continue

        if source not in entity_ids or target not in entity_ids:
            continue

        # 避免重复关系
        relation_key = (source, target)
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)

        valid_relationships.append(rel)

    # 创建 NetworkX 图
    G = nx.Graph()

    # 添加节点（带数据验证）
    for entity in entities:
        try:
            entity_id = entity["entity_id"]
            entity_name = entity.get("entity_name", entity_id)

            # 安全截断描述
            description = entity.get("description", "")
            if description and len(description) > 200:
                description = description[:200] + "..."

            G.add_node(
                entity_id,
                label=entity_name,
                type=str(entity.get("entity_type", "unknown")),
                description=description
            )
        except Exception as e:
            logger.error(f"添加节点失败: {e}")
            continue

    # 添加边（带类型安全保护）
    for rel in valid_relationships:
        try:
            source = rel["source"]
            target = rel["target"]

            # 安全获取weight，确保是float类型
            weight = rel.get("weight", 1.0)
            if not isinstance(weight, (int, float)):
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    weight = 1.0

            # 限制weight范围
            weight = max(0.1, min(weight, 10.0))

            G.add_edge(
                source,
                target,
                relation=str(rel.get("relation_type", "related")),
                weight=weight
            )
        except Exception as e:
            logger.error(f"添加边失败: {e}")
            continue

    # 检查图是否为空
    if G.number_of_nodes() == 0:
        st.warning("图数据为空")
        return

    # 计算布局（根据图大小选择策略）
    try:
        num_nodes = G.number_of_nodes()

        if num_nodes == 1:
            # 单节点特殊处理
            pos = {list(G.nodes())[0]: (0.5, 0.5)}
        elif num_nodes <= 20:
            # 小图使用spring layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif num_nodes <= 100:
            # 中等图调整参数
            pos = nx.spring_layout(G, k=1, iterations=30, seed=42)
        else:
            # 大图使用快速布局
            pos = nx.spring_layout(G, k=0.5, iterations=20, seed=42)

    except Exception as e:
        logger.error(f"计算布局失败: {e}")
        st.error("无法生成图布局")
        return

    # 提取边和节点信息（带错误处理）
    edge_x = []
    edge_y = []
    edge_text = []

    try:
        for edge in G.edges(data=True):
            if edge[0] not in pos or edge[1] not in pos:
                logger.warning(f"边 {edge} 的节点位置缺失，跳过")
                continue
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(str(edge[2].get("relation", "related")))
    except Exception as e:
        logger.error(f"处理边数据失败: {e}")

    node_x = []
    node_y = []
    node_text = []
    node_info = []

    try:
        for node in G.nodes(data=True):
            if node[0] not in pos:
                logger.warning(f"节点 {node[0]} 位置缺失，跳过")
                continue
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node[1].get("label", node[0])))
            description = node[1].get('description', '')
            # 安全截断描述
            if description and len(description) > 100:
                description = description[:100] + "..."
            node_info.append(
                f"**{node[1].get('label', node[0])}**<br>"
                f"类型: {node[1].get('type', 'unknown')}<br>"
                f"描述: {description or 'N/A'}"
            )
    except Exception as e:
        logger.error(f"处理节点数据失败: {e}")

    # 检查是否有有效数据
    if not node_x or not node_y:
        st.warning("无法生成可视化：没有有效的节点数据")
        logger.warning("节点数据为空，跳过可视化")
        return

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
                        title=dict(
                            text='知识图谱可视化',
                            font=dict(size=16)
                        ),
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

    # 显示统计信息（带类型安全保护）
    try:
        entity_count = int(stats.get("entity_count", 0))
        relation_count = int(stats.get("relation_count", 0))
        # 安全处理density：转换为float，失败则使用0
        try:
            density = float(stats.get("density", 0))
        except (ValueError, TypeError):
            density = 0.0
        is_connected = bool(stats.get("is_connected", False))
    except Exception as e:
        logger.error(f"解析统计数据失败: {e}")
        entity_count = 0
        relation_count = 0
        density = 0.0
        is_connected = False

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("实体数", entity_count)
    with col2:
        st.metric("关系数", relation_count)
    with col3:
        st.metric("图密度", f"{density:.3f}")
    with col4:
        st.metric("连通性", "是" if is_connected else "否")

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
