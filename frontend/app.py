"""
Streamlit 前端应用
"""
import streamlit as st
import requests
import time
from typing import Optional

# 页面配置
st.set_page_config(
    page_title="GRAG 技术文档智能问答系统",
    page_icon="🤖",
    layout="wide"
)

# API 配置
API_BASE_URL = "http://localhost:8000/api/v1"
API_USERNAME = st.secrets.get("API_USERNAME", "admin")
API_PASSWORD = st.secrets.get("API_PASSWORD", "")

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "unknown"


def get_auth_headers():
    """获取认证头"""
    import base64
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def query_api(query: str, stream: bool = False) -> Optional[dict]:
    """
    调用 API 进行查询
    
    Args:
        query: 查询文本
        stream: 是否流式返回
        
    Returns:
        查询结果字典
    """
    try:
        url = f"{API_BASE_URL}/query/stream" if stream else f"{API_BASE_URL}/query"
        
        response = requests.post(
            url,
            json={"query": query, "use_cache": True, "stream": stream},
            headers=get_auth_headers(),
            stream=stream,
            timeout=60
        )
        response.raise_for_status()
        
        if stream:
            return {"stream": True, "response": response}
        else:
            return response.json()
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


def submit_feedback(query: str, is_positive: bool):
    """
    提交用户反馈
    
    Args:
        query: 查询文本
        is_positive: 是否为正面反馈
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"query": query, "is_positive": is_positive},
            headers=get_auth_headers(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"反馈提交失败: {e}")
        return None


def get_stats():
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
        st.error(f"获取统计信息失败: {e}")
        return None


def switch_model(model_type: str) -> bool:
    """
    切换模型
    
    Args:
        model_type: 模型类型（"api" 或 "local"，local 暂时禁用）
        
    Returns:
        是否切换成功
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/model/switch",
            json={"model_type": model_type},
            headers=get_auth_headers(),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            st.session_state.current_model = result.get("current_model", "unknown")
        return result.get("success", False)
    except Exception as e:
        st.error(f"模型切换失败: {e}")
        return False


# 主界面
st.title("🤖 GRAG 技术文档智能问答系统")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 模型切换
    st.subheader("模型选择")
    # ========================================================================
    # Ollama 本地模型支持已弃用
    # ========================================================================
    # model_type = st.selectbox(
    #     "当前模型",
    #     ["local", "deepseek"],
    #     index=0 if st.session_state.current_model == "local" else 1
    # )
    
    model_type = st.selectbox(
        "当前模型",
        ["api"],  # local 暂时禁用
        index=0
    )
    
    if st.button("切换模型"):
        if switch_model(model_type):
            st.success(f"已切换到 {model_type}")
        else:
            st.error("模型切换失败")
    
    st.divider()
    
    # 系统统计
    st.subheader("📊 系统统计")
    if st.button("刷新统计"):
        stats = get_stats()
        if stats:
            metrics = stats.get("metrics", {})
            cache = stats.get("cache", {})
            
            st.metric("API 调用次数", sum(metrics.get("api_calls", {}).values()))
            st.metric("缓存条目数", cache.get("total_entries", 0))
            st.metric("平均质量评分", f"{cache.get('average_quality_score', 0):.2f}")
    
    st.divider()
    
    # 清空对话
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 主聊天界面
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("query") and message.get("answer"):
            # 反馈按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"positive_{message['id']}"):
                    submit_feedback(message["query"], True)
                    st.success("感谢您的反馈！")
            with col2:
                if st.button("👎", key=f"negative_{message['id']}"):
                    submit_feedback(message["query"], False)
                    st.info("感谢您的反馈，我们会改进！")

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "id": len(st.session_state.messages)
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示助手回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 调用 API
        result = query_api(prompt, stream=False)
        
        if result and result.get("success"):
            answer = result.get("answer", "")
            full_response = answer
            message_placeholder.markdown(full_response)
            
            # 显示额外信息
            with st.expander("📋 详细信息"):
                st.write(f"**响应时间**: {result.get('response_time', 0):.2f} 秒")
                st.write(f"**模型类型**: {result.get('model_type', 'unknown')}")
                st.write(f"**来自缓存**: {'是' if result.get('from_cache') else '否'}")
                if result.get("context_ids"):
                    st.write(f"**上下文数量**: {len(result.get('context_ids', []))}")
        else:
            error_msg = result.get("error", "查询失败") if result else "API 调用失败"
            message_placeholder.error(f"❌ {error_msg}")
            full_response = f"错误: {error_msg}"
        
        # 添加助手消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "query": prompt,
            "answer": full_response,
            "id": len(st.session_state.messages)
        })

