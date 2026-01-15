"""
聊天页面 - 增强版
支持历史记录、搜索、导出等功能
"""
import streamlit as st
import requests
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

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
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "export_format" not in st.session_state:
        st.session_state.export_format = "json"


def get_auth_headers():
    """获取认证头"""
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def query_api(query: str, stream: bool = False) -> Optional[dict]:
    """调用 API 进行查询"""
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
        return response.json() if not stream else {"stream": True, "response": response}
    except Exception as e:
        return None


def submit_feedback(query: str, is_positive: bool):
    """提交用户反馈"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"query": query, "is_positive": is_positive},
            headers=get_auth_headers(),
            timeout=10
        )
        return response.json()
    except Exception as e:
        return None


def save_chat_history(messages: List[Dict], filepath: str):
    """保存聊天历史到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_chat_history(filepath: str) -> List[Dict]:
    """从文件加载聊天历史"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return []


def export_chat(messages: List[Dict], format_type: str = "json") -> str:
    """导出聊天记录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format_type == "json":
        filename = f"chat_history_{timestamp}.json"
        content = json.dumps(messages, ensure_ascii=False, indent=2)
        return filename, content, "application/json"

    elif format_type == "txt":
        filename = f"chat_history_{timestamp}.txt"
        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"[{role}] {msg.get('content', '')}")
        content = "\n\n".join(lines)
        return filename, content, "text/plain"

    elif format_type == "md":
        filename = f"chat_history_{timestamp}.md"
        lines = ["# 聊天历史\n\n"]
        for msg in messages:
            role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
            lines.append(f"## {role}\n\n{msg.get('content', '')}\n\n---\n")
        content = "".join(lines)
        return filename, content, "text/markdown"


def show():
    """显示聊天页面"""
    init_session_state()

    st.subheader("💬 智能对话")

    # 显示聊天消息
    for idx, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示反馈按钮（仅助手消息）
            if message["role"] == "assistant" and message.get("query"):
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("👍 有帮助", key=f"thumb_up_{idx}"):
                        submit_feedback(message["query"], True)
                        st.success("感谢反馈！")
                        st.rerun()
                with col_btn2:
                    if st.button("👎 无帮助", key=f"thumb_down_{idx}"):
                        submit_feedback(message["query"], False)
                        st.info("感谢反馈，我们会改进！")
                        st.rerun()

                # 显示详细信息
                with st.expander("📋 详细信息"):
                    if "response_time" in message:
                        st.write(f"**响应时间**: {message['response_time']:.2f} 秒")
                    if "model_type" in message:
                        st.write(f"**模型**: {message['model_type']}")
                    if "from_cache" in message:
                        cache_status = "是" if message['from_cache'] else "否"
                        st.write(f"**来自缓存**: {cache_status}")

    st.markdown("---")

    # 用户输入（必须在顶层，不能在容器内）
    prompt = st.chat_input("请输入您的问题...")

    # 处理用户输入
    if prompt:
        # 添加用户消息
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # 调用 API
        with st.spinner("正在思考..."):
            result = query_api(prompt)

        # 显示助手回复
        if result and result.get("success"):
            answer = result.get("answer", "")

            # 保存助手消息
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "query": prompt,
                "answer": answer,
                "response_time": result.get("response_time", 0),
                "model_type": result.get("model_type", "unknown"),
                "from_cache": result.get("from_cache", False),
                "timestamp": datetime.now().isoformat()
            })
        else:
            error_msg = result.get("error", "查询失败") if result else "API 调用失败"
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"❌ {error_msg}",
                "timestamp": datetime.now().isoformat()
            })

        st.rerun()

    # 底部工具栏（展开式）
    with st.expander("🔍 搜索和导出", expanded=False):
        col_search, col_export = st.columns(2)

        with col_search:
            st.markdown("**搜索历史**")
            search_query = st.text_input("输入关键词", placeholder="搜索聊天内容...")

            if search_query:
                filtered_messages = [
                    msg for msg in st.session_state.chat_messages
                    if search_query.lower() in msg.get("content", "").lower()
                ]

                if filtered_messages:
                    st.write(f"找到 {len(filtered_messages)} 条结果")
                    for msg in filtered_messages:
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        st.caption(f"{role_icon} {msg.get('content', '')[:100]}...")
                else:
                    st.info("未找到匹配结果")

        with col_export:
            st.markdown("**导出聊天**")
            export_format = st.selectbox(
                "选择格式",
                ["json", "txt", "md"],
                label_visibility="collapsed"
            )

            if st.button("导出记录"):
                if st.session_state.chat_messages:
                    filename, content, mime_type = export_chat(
                        st.session_state.chat_messages,
                        export_format
                    )
                    st.download_button(
                        label=f"下载 {filename}",
                        data=content,
                        file_name=filename,
                        mime=mime_type
                    )

            if st.button("🗑️ 清空历史"):
                st.session_state.chat_messages = []
                st.success("已清空聊天历史")
                st.rerun()

        # 统计信息
        st.markdown("---")
        user_msgs = sum(1 for m in st.session_state.chat_messages if m["role"] == "user")
        asst_msgs = sum(1 for m in st.session_state.chat_messages if m["role"] == "assistant")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("用户消息", user_msgs)
        with col_stat2:
            st.metric("助手回复", asst_msgs)


# Streamlit 多页面应用会自动执行此文件
show()
