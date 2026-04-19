"""
聊天页面 - 增强版
支持历史记录、搜索、导出、会话持久化等功能
"""
import logging
import streamlit as st
import requests
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Import session manager and config
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.storage.session_manager import FileSessionManager, generate_session_title
from frontend.config import API_BASE_URL, API_USERNAME, API_PASSWORD

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session manager (文件系统存储，持久化)
session_manager = FileSessionManager()
logger.info("文件会话管理器已初始化")


def init_session_state():
    """
    初始化会话状态

    自动加载最新的会话历史，或创建新会话。
    会话数据持久化到文件系统，刷新或重启浏览器后保留。
    """
    # Initialize basic session state variables
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "export_format" not in st.session_state:
        st.session_state.export_format = "json"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize debouncing state to prevent multiple session creation
    if "creating_session" not in st.session_state:
        st.session_state.creating_session = False

    # Initialize session creation counter to prevent duplicate creations
    if "session_creation_counter" not in st.session_state:
        st.session_state.session_creation_counter = 0

    # Session persistence - load or create session
    if "current_session_id" not in st.session_state:
        # Try to load the latest session
        latest_session = session_manager.get_latest_session()

        if latest_session:
            # Load existing session
            st.session_state.current_session_id = latest_session["session_id"]
            st.session_state.chat_messages = latest_session.get("messages", [])
            logger.info(f"Loaded existing session: {latest_session['session_id']}")
        else:
            # Create new session
            session_id = session_manager.create_session()
            st.session_state.current_session_id = session_id
            st.session_state.chat_messages = []
            logger.info(f"Created new session: {session_id}")

    # Ensure chat_messages exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def get_auth_headers():
    """获取认证头"""
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def save_current_session() -> bool:
    """
    保存当前会话到文件系统

    在每次发送消息后自动调用，将对话历史持久化到 JSON 文件。
    刷新或重启浏览器后，会话数据仍然保留。

    Returns:
        bool: 保存是否成功
    """
    try:
        if "current_session_id" not in st.session_state:
            logger.warning("没有会话 ID 可保存")
            return False

        # 从第一条用户消息生成标题
        title = generate_session_title(st.session_state.chat_messages)

        # 准备元数据
        metadata = {
            "model_type": "unknown",
            "saved_at": datetime.now().isoformat(),
        }

        # 尝试从最后一条助手消息中获取模型类型
        for msg in reversed(st.session_state.chat_messages):
            if msg.get("role") == "assistant" and "model_type" in msg:
                metadata["model_type"] = msg["model_type"]
                break

        # 保存会话
        success = session_manager.save_session(
            session_id=st.session_state.current_session_id,
            messages=st.session_state.chat_messages,
            title=title,
            metadata=metadata,
        )

        if success:
            logger.debug(f"会话保存成功: {st.session_state.current_session_id}")

        return success

    except Exception as e:
        logger.error(f"保存会话时出错: {e}")
        return False


def format_timestamp(timestamp_str: str) -> str:
    """
    格式化时间戳为易读格式

    Args:
        timestamp_str: ISO 格式的时间戳字符串

    Returns:
        格式化后的时间字符串，如 "02-24 10:30"
    """
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return timestamp_str


def switch_session(session_id: str) -> bool:
    """
    切换到指定会话

    Args:
        session_id: 要切换到的会话 ID

    Returns:
        切换成功返回 True，失败返回 False
    """
    try:
        # 先保存当前会话
        save_current_session()

        # 加载目标会话
        session_data = session_manager.load_session(session_id)
        if not session_data:
            logger.error(f"无法加载会话: {session_id}")
            return False

        # 更新会话状态
        st.session_state.current_session_id = session_id
        st.session_state.chat_messages = session_data.get("messages", [])

        logger.info(f"切换到会话: {session_id}")
        return True

    except Exception as e:
        logger.error(f"切换会话失败: {e}")
        return False


def create_new_session() -> bool:
    """
    创建新会话

    保存当前会话后，创建一个全新的空会话。
    强制更新 session_manager 的当前会话状态以确保 UI 同步。

    Returns:
        创建成功返回 True，失败返回 False
    """
    try:
        # 先保存当前会话
        save_current_session()

        # 创建新会话
        new_session_id = session_manager.create_session()

        # 切换到新会话
        st.session_state.current_session_id = new_session_id
        st.session_state.chat_messages = []

        # 强制更新 session_manager 的当前会话
        session_manager.set_current_session(new_session_id)

        logger.info(f"创建新会话: {new_session_id}")
        return True

    except Exception as e:
        logger.error(f"创建新会话失败: {e}")
        return False


def delete_session_with_confirmation(session_id: str, session_title: str) -> bool:
    """
    删除会话（带确认）

    Args:
        session_id: 要删除的会话 ID
        session_title: 会话标题（用于确认提示）

    Returns:
        删除成功返回 True，取消或失败返回 False
    """
    try:
        # 不允许删除当前会话
        if session_id == st.session_state.get("current_session_id"):
            st.error("无法删除当前正在使用的会话")
            return False

        # 删除会话
        success = session_manager.delete_session(session_id)
        if success:
            logger.info(f"删除会话成功: {session_id}")
            return True
        else:
            st.error("删除失败")
            return False

    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        st.error(f"删除失败: {e}")
        return False


def export_all_sessions() -> tuple[str, str]:
    """
    导出所有会话为 JSON

    Returns:
        (filename, content) 文件名和内容
    """
    try:
        sessions = session_manager.list_sessions()

        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "sessions": []
        }

        for session_meta in sessions:
            session_id = session_meta.get("session_id")
            if session_id:
                session_data = session_manager.load_session(session_id)
                if session_data:
                    export_data["sessions"].append(session_data)

        content = json.dumps(export_data, ensure_ascii=False, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grag_sessions_backup_{timestamp}.json"

        return filename, content

    except Exception as e:
        logger.error(f"导出会话失败: {e}")
        raise


def render_session_sidebar():
    """
    渲染会话管理侧边栏

    显示所有历史会话，支持切换、创建、删除操作。
    """
    try:
        with st.sidebar:
            st.markdown("### 💬 我的对话")

            # 新建对话按钮 - 使用计数器防止重复点击
            # 获取当前计数器值（作为唯一标识符）
            current_counter = st.session_state.get("session_creation_counter", 0)

            if st.button(
                "➕ 新建对话",
                use_container_width=True,
                type="primary",
                key=f"create_session_btn_{current_counter}",
                disabled=st.session_state.get("creating_session", False)
            ):
                # 防止重复执行：检查是否已在创建中
                if not st.session_state.get("creating_session", False):
                    # 标记开始创建
                    st.session_state.creating_session = True

                    # 执行创建
                    success = create_new_session()

                    # 增加计数器，使按钮的 key 在下次 rerun 时改变
                    # 这会导致旧按钮的点击状态被清除，防止重复触发
                    st.session_state.session_creation_counter = current_counter + 1

                    # 重置创建标志
                    st.session_state.creating_session = False

                    # 刷新 UI
                    if success:
                        st.rerun()
                    else:
                        st.error("创建会话失败")

            st.markdown("---")

            # 获取会话列表（带错误处理）
            try:
                sessions = session_manager.list_sessions()
            except Exception as e:
                logger.error(f"获取会话列表失败: {e}")
                st.error(f"⚠️ 无法加载会话列表")
                sessions = []

            current_session_id = st.session_state.get("current_session_id", "")

            # 显示会话列表
            if sessions:
                st.markdown("#### 📚 历史对话")

                for idx, session in enumerate(sessions):
                    try:
                        session_id = session.get("session_id", "")
                        title = session.get("title", "新对话")
                        updated_at = session.get("updated_at", "")
                        message_count = session.get("message_count", 0)

                        # 检查是否是当前会话
                        is_current = (session_id == current_session_id)

                        # 格式化时间
                        time_str = format_timestamp(updated_at)

                        # 创建会话项
                        with st.container():
                            # 使用 columns 布局：标题 + 删除按钮
                            col_title, col_delete = st.columns([4, 1])

                            with col_title:
                                # 会话标题和点击区域
                                if is_current:
                                    st.markdown(f"**🔹 {title}**")
                                else:
                                    # 使用 button 模拟可点击区域
                                    button_label = f"📝 {title}"
                                    if st.button(
                                        button_label,
                                        key=f"switch_{session_id}",
                                        help="点击切换到此对话"
                                    ):
                                        if switch_session(session_id):
                                            st.rerun()

                            with col_delete:
                                # 删除按钮
                                if not is_current:
                                    if st.button(
                                        "🗑️",
                                        key=f"delete_{session_id}",
                                        help="删除此对话",
                                        disabled=is_current
                                    ):
                                        delete_session_with_confirmation(session_id, title)
                                        st.rerun()

                            # 显示元数据
                            st.caption(
                                f"🕒 {time_str} | 💬 {message_count} 条消息"
                            )

                            # 添加分隔线（最后一个会话不需要）
                            if idx < len(sessions) - 1:
                                st.markdown("---")

                    except Exception as e:
                        logger.error(f"显示会话项 {idx} 失败: {e}")
                        continue

            else:
                # 空状态
                st.info("📭 暂无历史对话")
                st.markdown("开始新的对话吧！")

            st.markdown("---")

            # 导入/导出区域
            st.markdown("#### 📦 数据管理")

            col_import, col_export = st.columns(2)

            with col_export:
                if st.button("📤 导出全部", use_container_width=True):
                    try:
                        filename, content = export_all_sessions()
                        st.download_button(
                            label="下载",
                            data=content,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    except Exception as e:
                        logger.error(f"导出失败: {e}")
                        st.error(f"导出失败: {e}")

            with col_import:
                # 导入功能（预留接口）
                if st.button("📥 导入", use_container_width=True):
                    st.info("导入功能开发中...")

            # 侧边栏底部信息
            st.markdown("---")
            st.caption("💡 提示：对话会自动保存到浏览器")

    except Exception as e:
        logger.error(f"侧边栏渲染失败: {e}", exc_info=True)
        with st.sidebar:
            st.error("⚠️ 侧边栏加载出错")
            if st.button("🔄 重试"):
                st.rerun()


def query_api_stream(query: str):
    """
    流式调用 API - 处理 SSE 格式

    带有超时和块数限制，防止无限循环。

    Args:
        query: 用户查询文本

    Yields:
        dict: SSE 数据块，格式：
            - {"content": "...", "done": false} - 内容块
            - {"content": "", "done": true, ...} - 结束块
            - {"error": "..."} - 错误块
    """
    try:
        url = f"{API_BASE_URL}/query/stream"
        response = requests.post(
            url,
            json={"query": query, "use_cache": True, "stream": True},
            headers=get_auth_headers(),
            stream=True,
            timeout=120
        )

        if not response.ok:
            # 解析后端返回的详细错误（如 FastAPI detail）
            try:
                body = response.json()
                msg = body.get("detail", str(body))
                if isinstance(msg, list):
                    msg = msg[0].get("msg", str(msg)) if msg else response.reason
                elif not isinstance(msg, str):
                    msg = str(msg)
            except Exception:
                msg = f"{response.status_code} {response.reason}"
            yield {"error": msg}
            return

        chunk_count = 0
        MAX_CHUNKS = 10000  # 防止无限循环

        for line in response.iter_lines():
            # 跳过空行和非字节类型的数据
            if not line or not isinstance(line, bytes):
                continue

            try:
                line = line.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                continue

            if line.startswith('data: '):
                chunk_count += 1
                if chunk_count > MAX_CHUNKS:
                    logger.warning(f"流式响应超过最大块数限制 ({MAX_CHUNKS})，已中断")
                    yield {"error": "响应过长，已中断"}
                    return

                try:
                    data = json.loads(line[6:])
                    if isinstance(data, dict):
                        yield data
                        # 检查是否完成
                        if data.get("done"):
                            return
                except (json.JSONDecodeError, ValueError):
                    continue

    except requests.exceptions.Timeout:
        yield {"error": "请求超时，请稍后重试"}
    except requests.exceptions.ConnectionError:
        yield {"error": f"无法连接后端服务，请确认 API 已启动 ({API_BASE_URL})"}
    except Exception as e:
        yield {"error": str(e)}


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
    """
    显示聊天页面

    主函数，渲染完整的聊天界面，包括：
    - 侧边栏：会话管理
    - 主区域：对话历史和输入框
    """
    # 初始化会话状态
    init_session_state()

    # 渲染会话管理侧边栏
    render_session_sidebar()

    # 主区域标题
    st.subheader("💬 智能对话")

    # 紧急状态恢复按钮（当页面卡住时使用）
    with st.expander("🔧 状态管理", expanded=False):
        st.markdown("**如果页面出现问题，可以使用以下工具恢复：**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 刷新页面", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("🧹 清空当前对话", use_container_width=True):
                st.session_state.chat_messages = []
                save_current_session()
                st.success("对话已清空")
                st.rerun()

        with col3:
            if st.button("⚠️ 紧急重置", use_container_width=True):
                # 保存当前会话数据
                current_session_id = st.session_state.get("current_session_id")
                current_messages = st.session_state.get("chat_messages", [])

                # 重置所有状态
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                # 恢复基本状态
                st.session_state.chat_messages = current_messages
                if current_session_id:
                    st.session_state.current_session_id = current_session_id

                # 重新初始化
                init_session_state()
                st.success("状态已重置")
                st.rerun()

        st.caption("💡 提示：刷新页面通常可以解决临时问题")

    st.markdown("---")

    # 显示当前会话信息
    current_session_id = st.session_state.get("current_session_id", "")
    if current_session_id:
        sessions = session_manager.list_sessions()
        current_session = next(
            (s for s in sessions if s.get("session_id") == current_session_id),
            None
        )
        if current_session:
            st.caption(f"📌 {current_session.get('title', '新对话')}")

    # 显示聊天消息（带错误处理）
    for idx, message in enumerate(st.session_state.chat_messages):
        try:
            # 验证消息基本结构
            if not isinstance(message, dict):
                logger.error(f"消息 {idx} 不是字典类型: {type(message)}")
                continue

            if "role" not in message:
                logger.error(f"消息 {idx} 缺少 role 字段")
                continue

            if "content" not in message:
                logger.error(f"消息 {idx} 缺少 content 字段")
                continue

            # 安全显示消息
            with st.chat_message(message["role"]):
                # 安全显示内容
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                st.markdown(content)

                # 显示反馈按钮（仅助手消息）
                if message["role"] == "assistant" and message.get("query"):
                    try:
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
                    except Exception as e:
                        logger.error(f"显示反馈按钮失败 (消息 {idx}): {e}")

                # 显示详细信息（仅助手消息）
                if message["role"] == "assistant":
                    try:
                        with st.expander("📋 详细信息"):
                            if "response_time" in message:
                                try:
                                    response_time = float(message["response_time"])
                                    st.write(f"**响应时间**: {response_time:.2f} 秒")
                                except (ValueError, TypeError):
                                    st.write(f"**响应时间**: {message.get('response_time', '未知')}")

                            if "model_type" in message:
                                st.write(f"**模型**: {message.get('model_type', 'unknown')}")

                            if "from_cache" in message:
                                cache_status = "是" if message.get("from_cache") else "否"
                                st.write(f"**来自缓存**: {cache_status}")

                            # 安全显示引用列表
                            if "citations" in message:
                                citations = message.get("citations", [])
                                if citations and isinstance(citations, list):
                                    st.write(f"**引用**: {len(citations)} 个")
                                    for citation in citations:
                                        try:
                                            if isinstance(citation, str):
                                                st.caption(citation)
                                            elif isinstance(citation, dict):
                                                # 兼容字典格式
                                                source = citation.get('source', {})
                                                if isinstance(source, dict):
                                                    file_path = source.get('file_path', '未知来源')
                                                else:
                                                    file_path = '未知来源'
                                                st.caption(f"• {file_path}")
                                            else:
                                                st.caption(f"• {str(citation)}")
                                        except Exception as e:
                                            logger.error(f"显示引用失败: {e}")
                                            st.caption("• [引用格式错误]")

                            # 安全显示引用统计
                            if "citation_info" in message:
                                citation_info = message.get("citation_info")
                                if isinstance(citation_info, dict):
                                    if citation_info.get("has_citations"):
                                        citation_count = citation_info.get('citation_count', 0)
                                        st.write(f"**引用统计**: {citation_count} 个引用")
                                        if citation_info.get("was_fixed"):
                                            st.info("引用已自动修复")
                    except Exception as e:
                        logger.error(f"显示详细信息失败 (消息 {idx}): {e}")
                        st.caption("⚠️ 详细信息显示出错")

        except Exception as e:
            logger.error(f"显示消息 {idx} 时发生错误: {e}", exc_info=True)
            # 显示错误提示，但不中断整个对话
            st.error(f"⚠️ 消息 {idx + 1} 显示出错")
            continue

    st.markdown("---")

    # 用户输入（必须在顶层，不能在容器内）
    prompt = st.chat_input("请输入您的问题...")

    # 处理用户输入
    if prompt:
        try:
            # 添加用户消息到对话历史
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })

            # 流式输出助手回复
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                metadata = {}

                # 创建一个状态容器来控制加载状态
                status_container = st.container()
                with status_container:
                    status_placeholder = st.empty()

                # 显示初始加载状态
                status_placeholder.info("正在思考...")

                # 流式处理响应
                first_chunk = True
                chunk_count = 0
                MAX_CHUNKS = 10000  # 防止无限循环

                try:
                    for chunk in query_api_stream(prompt):
                        chunk_count += 1
                        if chunk_count > MAX_CHUNKS:
                            full_response += "\n\n[错误: 响应过长，已中断]"
                            logger.warning(f"响应超过最大块数限制 ({MAX_CHUNKS})，已中断")
                            status_placeholder.empty()
                            break

                        # 清除加载状态（在第一个有效块时）
                        if first_chunk and "content" in chunk:
                            status_placeholder.empty()
                            first_chunk = False

                        # 处理错误
                        if "error" in chunk:
                            full_response = f"错误: {chunk['error']}"
                            message_placeholder.error(full_response)
                            status_placeholder.empty()
                            break

                        # 处理内容
                        if "content" in chunk and chunk["content"]:
                            full_response += chunk["content"]
                            # 显示带光标的文本（打字效果）
                            message_placeholder.markdown(full_response + "▌")

                        # 处理结束标记
                        if "done" in chunk and chunk["done"]:
                            # 安全提取元数据（包含新的引用信息）
                            try:
                                metadata = {
                                    "response_time": float(chunk.get("response_time", 0)),
                                    "model_type": str(chunk.get("model_type", "unknown")),
                                    "from_cache": bool(chunk.get("from_cache", False)),
                                    "context_ids": list(chunk.get("context_ids", [])),
                                    "intent": str(chunk.get("intent", "")),
                                    "citations": list(chunk.get("citations", [])),
                                    "citation_info": dict(chunk.get("citation_info", {}))
                                }
                            except Exception as e:
                                logger.error(f"提取元数据失败: {e}")
                                metadata = {}
                            status_placeholder.empty()
                            break

                except Exception as stream_error:
                    logger.error(f"流式处理出错: {stream_error}", exc_info=True)
                    full_response += f"\n\n⚠️ 响应处理出错: {str(stream_error)}"
                    status_placeholder.empty()

                # 最终显示（去掉光标）
                try:
                    message_placeholder.markdown(full_response)
                except Exception as display_error:
                    logger.error(f"显示最终响应失败: {display_error}")
                    message_placeholder.write(full_response)

            # 保存完整消息到对话历史
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "query": prompt,
                "timestamp": datetime.now().isoformat()
            }

            # 添加元数据（如果有）
            if metadata:
                assistant_message.update(metadata)

            st.session_state.chat_messages.append(assistant_message)

            # 自动保存会话（持久化到浏览器存储）
            try:
                save_current_session()
            except Exception as save_error:
                logger.error(f"保存会话失败: {save_error}")
                # 不中断用户体验，静默失败

        except Exception as e:
            logger.error(f"处理用户输入时发生错误: {e}", exc_info=True)
            st.error(f"⚠️ 处理请求时出错: {str(e)}")
            # 如果已经添加了用户消息，移除它（因为请求失败了）
            if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
                st.session_state.chat_messages.pop()

        # 注意：不要在这里调用 st.rerun()，否则会导致无限循环
        # Streamlit 的聊天界面会自动更新，无需手动刷新

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
