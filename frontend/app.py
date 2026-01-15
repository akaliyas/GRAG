"""
GRAG 技术文档智能问答系统 - Streamlit 多页面应用
"""
import streamlit as st
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 页面配置
st.set_page_config(
    page_title="GRAG 技术文档智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏系统信息
with st.sidebar:
    st.markdown("### 🤖 GRAG System")
    st.markdown("---")
    st.markdown("**项目**: GRAG")
    st.markdown("**版本**: 1.0.0")
    st.markdown("---")
    st.info("💡 使用左侧导航菜单切换页面")

# 主标题（仅首页显示）
st.markdown("""
<div class="main-header">
    <h1>🤖 GRAG 技术文档智能问答系统</h1>
    <p>基于知识图谱增强检索生成的智能问答平台</p>
</div>
""", unsafe_allow_html=True)

# 首页欢迎内容
st.markdown("### 欢迎使用 GRAG 系统")
st.markdown("""
请使用左侧导航菜单访问以下功能：

- **💬 聊天**: 智能问答对话
- **🔄 数据管道**: 数据采集和处理流程管理
- **📊 仪表盘**: 系统监控和性能指标
""")

