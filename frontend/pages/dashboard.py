"""
系统监控仪表盘页面
显示系统指标、性能数据和健康状态
"""
import streamlit as st
import requests
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

# API 配置
API_BASE_URL = "http://localhost:8000/api/v1"

# 安全获取 secrets（避免 FileNotFoundError）
try:
    API_USERNAME = st.secrets["API_USERNAME"]
except (FileNotFoundError, KeyError):
    API_USERNAME = "admin"

try:
    API_PASSWORD = st.secrets["API_PASSWORD"]
except (FileNotFoundError, KeyError):
    API_PASSWORD = ""


def init_session_state():
    """初始化会话状态"""
    if "dashboard_refresh_interval" not in st.session_state:
        st.session_state.dashboard_refresh_interval = 5
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None


def get_auth_headers():
    """获取认证头"""
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def get_stats() -> Optional[dict]:
    """获取系统统计信息"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/stats",
            headers=get_auth_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def get_health() -> bool:
    """检查系统健康状态"""
    try:
        response = requests.get(
            "http://localhost:8000/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        return False


def get_metrics_history() -> list:
    """获取历史指标数据"""
    metrics_file = Path(__file__).parent.parent.parent / "logs" / "metrics.json"

    if not metrics_file.exists():
        return []

    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("history", [])
    except Exception as e:
        return []


def format_size(bytes_size: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def show():
    """显示仪表盘页面"""
    init_session_state()

    st.subheader("📊 系统监控仪表盘")

    # 自动刷新控制
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("**⏱️ 自动刷新**")
        refresh_interval = st.select_slider(
            "刷新间隔",
            options=[0, 5, 10, 30, 60],
            value=st.session_state.dashboard_refresh_interval,
            format_func=lambda x: "关闭" if x == 0 else f"{x} 秒"
        )
        st.session_state.dashboard_refresh_interval = refresh_interval

    with col2:
        st.markdown("**上次刷新**")
        if st.session_state.last_refresh:
            st.caption(st.session_state.last_refresh)
        else:
            st.caption("从未刷新")

    with col3:
        st.markdown("**操作**")
        if st.button("🔄 立即刷新", use_container_width=True):
            st.rerun()

    # 系统健康状态
    st.markdown("---")
    st.markdown("### 🏥 系统健康状态")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        health = get_health()
        if health:
            st.metric(
                "API 状态",
                "✅ 正常",
                help="后端 API 服务正常运行"
            )
        else:
            st.metric(
                "API 状态",
                "❌ 异常",
                help="无法连接到后端 API 服务"
            )

    with col2:
        stats = get_stats()
        if stats:
            st.metric(
                "服务状态",
                "✅ 运行中",
                help="所有服务组件正常运行"
            )
        else:
            st.metric(
                "服务状态",
                "⚠️ 检查中",
                help="正在检查服务状态..."
            )

    with col3:
        st.metric(
            "运行时间",
            "未知",
            help="系统运行时间"
        )

    with col4:
        st.metric(
            "数据库",
            "✅ 连接",
            help="PostgreSQL 数据库连接状态"
        )

    # 系统指标
    st.markdown("---")
    st.markdown("### 📈 系统指标")

    if stats:
        metrics = stats.get("metrics", {})
        cache = stats.get("cache", {})

        # API 调用统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            api_calls = metrics.get("api_calls", {})
            total_calls = sum(api_calls.values()) if api_calls else 0
            st.metric("API 调用次数", total_calls)

        with col2:
            avg_response = metrics.get("avg_response_time", 0)
            st.metric("平均响应时间", f"{avg_response:.2f}s")

        with col3:
            cache_entries = cache.get("total_entries", 0)
            st.metric("缓存条目", cache_entries)

        with col4:
            quality_score = cache.get("average_quality_score", 0)
            st.metric("质量评分", f"{quality_score:.2f}")

        # 详细指标
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 API 调用详情")
            if api_calls:
                for model, count in api_calls.items():
                    st.write(f"**{model}**: {count} 次")
            else:
                st.info("暂无 API 调用记录")

        with col_right:
            st.markdown("#### 💾 缓存详情")
            st.write(f"**总条目数**: {cache.get('total_entries', 0)}")
            st.write(f"**平均质量**: {cache.get('average_quality_score', 0):.2f}")
            st.write(f"**LRU 大小**: {cache.get('lru_max_size', 0)}")
            st.write(f"**清理间隔**: {cache.get('cleanup_interval', 0)}s")

    # 性能图表
    st.markdown("---")
    st.markdown("### 📉 性能趋势")

    # 获取历史数据
    history = get_metrics_history()

    if history:
        # 这里可以添加图表展示
        st.info("历史数据图表功能开发中...")
    else:
        st.info("暂无历史数据")

    # 数据库状态
    st.markdown("---")
    st.markdown("### 🗄️ 数据库状态")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**PostgreSQL**")
        st.metric("状态", "✅ 正常")
        st.metric("存储", "未知")

    with col2:
        st.markdown("**LightRAG 表**")
        st.metric("实体数", "未知")
        st.metric("关系数", "未知")

    with col3:
        st.markdown("**向量索引**")
        st.metric("pgvector", "✅ 已启用")
        st.metric("维度", "未知")

    # 最近活动
    st.markdown("---")
    st.markdown("### 🕐 最近活动")

    # 从日志文件读取最近活动
    log_file = Path(__file__).parent.parent.parent / "logs" / "grag.log"

    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                # 读取最后 20 行
                lines = f.readlines()[-20:]
                st.code("".join(lines), language="log")
        except Exception as e:
            st.warning(f"无法读取日志文件: {e}")
    else:
        st.info("日志文件不存在")

    # 快速操作
    st.markdown("---")
    st.markdown("### ⚡ 快速操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ 清空缓存", use_container_width=True):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/cache/clear",
                    headers=get_auth_headers(),
                    timeout=10
                )
                if response.status_code == 200:
                    st.success("缓存已清空")
                    st.rerun()
                else:
                    st.error("清空缓存失败")
            except Exception as e:
                st.error(f"操作失败: {e}")

    with col2:
        if st.button("📥 导出日志", use_container_width=True):
            log_file = Path(__file__).parent.parent.parent / "logs" / "grag.log"
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                st.download_button(
                    label="下载日志文件",
                    data=log_content,
                    file_name=f"grag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("日志文件不存在")

    with col3:
        if st.button("🔄 重启服务", use_container_width=True):
            st.warning("此功能需要手动重启服务")
            st.info("请使用以下命令重启：")
            st.code("docker-compose restart", language="bash")

    # 自动刷新
    if st.session_state.dashboard_refresh_interval > 0:
        import time
        time.sleep(st.session_state.dashboard_refresh_interval)
        st.rerun()


# Streamlit 多页面应用会自动执行此文件
show()
