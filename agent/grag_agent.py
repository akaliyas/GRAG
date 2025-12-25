"""
GRAG Agent 层
使用 LangGraph 实现意图识别、工具调用、数据预处理、检索和回答
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from typing_extensions import TypedDict, Annotated
except ImportError:
    StateGraph = None
    ToolNode = None
    TypedDict = None
    Annotated = None

from models.model_manager import ModelManager
from knowledge.lightrag_wrapper import LightRAGWrapper
from utils.monitoring import track_performance
from agent.tools.github_ingestor import GitHubIngestor

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """意图类型"""
    QUERY = "query"  # 直接查询
    GITHUB_INGEST = "github_ingest"  # 需要从 GitHub 获取数据
    PREPROCESS = "preprocess"  # 需要数据预处理
    UNKNOWN = "unknown"  # 未知意图


if TypedDict:
    class AgentState(TypedDict):
        """Agent 状态"""
        messages: Annotated[List[Dict], "消息列表"]
        intent: str  # 意图类型
        query: str  # 查询文本
        need_github_ingest: bool  # 是否需要从 GitHub 获取数据
        need_preprocess: bool  # 是否需要预处理
        github_repo_urls: List[str]  # GitHub 仓库 URL 列表
        documents: List[str]  # 文档列表
        context_ids: List[str]  # 检索到的上下文 ID
        answer: str  # 生成的答案
        error: Optional[str]  # 错误信息
else:
    AgentState = Dict[str, Any]


class GRAGAgent:
    """GRAG Agent"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        lightrag_wrapper: LightRAGWrapper
    ):
        """
        初始化 Agent
        
        Args:
            model_manager: 模型管理器
            lightrag_wrapper: LightRAG 封装实例
        """
        if StateGraph is None:
            raise ImportError("需要安装 langgraph 库: pip install langgraph")
        
        self.model_manager = model_manager
        self.lightrag = lightrag_wrapper
        
        # 初始化 GitHub 提取工具（完全去爬虫化，仅使用 GitHub API）
        self.github_ingestor = GitHubIngestor()
        
        # 构建 Agent 图
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("GRAG Agent 已初始化（Zero-Crawler 模式）")
    
    def _build_graph(self) -> StateGraph:
        """构建 Agent 图"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("intent_recognition", self._intent_recognition)
        graph.add_node("github_ingest_tool", self._github_ingest_tool)
        graph.add_node("preprocess_tool", self._preprocess_tool)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_answer", self._generate_answer)
        
        # 设置入口点
        graph.set_entry_point("intent_recognition")
        
        # 添加边
        graph.add_conditional_edges(
            "intent_recognition",
            self._route_after_intent,
            {
                "github_ingest": "github_ingest_tool",
                "preprocess": "preprocess_tool",
                "retrieve": "retrieve",
                "error": END
            }
        )
        
        graph.add_edge("github_ingest_tool", "preprocess_tool")
        graph.add_edge("preprocess_tool", "retrieve")
        # retrieve 节点后，根据是否有答案决定是否进入 generate_answer
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "generate_answer": "generate_answer",
                "end": END
            }
        )
        graph.add_edge("generate_answer", END)
        
        return graph
    
    @track_performance("intent_recognition")
    def _intent_recognition(self, state: AgentState) -> AgentState:
        """
        意图识别节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            query = state.get("query", "")
            if not query:
                # 从消息中提取查询
                messages = state.get("messages", [])
                if messages:
                    query = messages[-1].get("content", "")
            
            # 使用 LLM 进行意图识别
            intent_prompt = f"""
请分析以下用户查询的意图，返回 JSON 格式：
{{
    "intent": "query|github_ingest|preprocess",
    "need_github_ingest": true/false,
    "need_preprocess": true/false,
    "github_repo_urls": ["https://github.com/owner/repo"]  // 如果需要从 GitHub 获取数据
}}

用户查询：{query}
"""
            
            response = self.model_manager.chat_completion(
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            # 解析意图（简化处理，实际应该更健壮）
            intent_result = response.choices[0].message.content
            
            # TODO: 解析 JSON 结果
            # 这里简化处理，实际应该使用 JSON 解析
            state["intent"] = IntentType.QUERY.value
            state["query"] = query
            state["need_github_ingest"] = False
            state["need_preprocess"] = False
            state["github_repo_urls"] = []
            
            logger.info(f"意图识别完成: {state['intent']}")
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            state["error"] = str(e)
            state["intent"] = IntentType.UNKNOWN.value
        
        return state
    
    def _route_after_intent(self, state: AgentState) -> str:
        """
        根据意图路由到下一个节点
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        if state.get("error"):
            return "error"
        
        if state.get("need_github_ingest"):
            return "github_ingest"
        elif state.get("need_preprocess"):
            return "preprocess"
        else:
            return "retrieve"
    
    def _route_after_retrieve(self, state: AgentState) -> str:
        """
        根据检索结果路由到下一个节点
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        if state.get("error"):
            return "end"
        
        # 如果 LightRAG 已经生成了答案，直接结束
        if state.get("answer"):
            return "end"
        
        # 如果没有答案，进入生成答案节点
        return "generate_answer"
    
    @track_performance("github_ingest_tool")
    def _github_ingest_tool(self, state: AgentState) -> AgentState:
        """
        GitHub 数据提取工具节点（完全去爬虫化，仅使用 GitHub API）
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            repo_urls = state.get("github_repo_urls", [])
            if not repo_urls:
                logger.warning("GitHub 仓库 URL 列表为空")
                state["documents"] = []
                return state
            
            all_documents = []
            for repo_url in repo_urls:
                try:
                    # 使用 GitHubIngestor 提取文档
                    documents = self.github_ingestor.download_and_clean(repo_url)
                    # 提取文档内容
                    doc_contents = [doc['content'] for doc in documents]
                    all_documents.extend(doc_contents)
                    logger.info(f"从 {repo_url} 提取了 {len(doc_contents)} 个文档")
                except Exception as e:
                    logger.error(f"从 {repo_url} 提取文档失败: {e}")
                    continue
            
            state["documents"] = all_documents
            logger.info(f"GitHub 数据提取完成: 共 {len(all_documents)} 个文档")
        except Exception as e:
            logger.error(f"GitHub 数据提取失败: {e}")
            state["error"] = str(e)
        
        return state
    
    @track_performance("preprocess_tool")
    def _preprocess_tool(self, state: AgentState) -> AgentState:
        """
        数据预处理工具节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            documents = state.get("documents", [])
            if documents:
                # 添加到 LightRAG 知识库
                self.lightrag.add_documents(documents)
                logger.info(f"预处理完成: {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            state["error"] = str(e)
        
        return state
    
    @track_performance("retrieve")
    def _retrieve(self, state: AgentState) -> AgentState:
        """
        检索节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            query = state.get("query", "")
            if not query:
                state["error"] = "查询文本为空"
                return state
            
            # 使用 LightRAG 进行检索（LightRAG 已经基于上下文生成了答案）
            result = self.lightrag.query(query, mode="hybrid", top_k=5)
            
            # 保存检索结果
            state["context_ids"] = result.get("context_ids", [])
            state["documents"] = result.get("contexts", [])
            
            # LightRAG 已经生成了答案，直接使用
            # 如果答案为空，则在 generate_answer 节点中基于上下文重新生成
            answer = result.get("answer", "")
            if answer:
                state["answer"] = answer
                logger.info(f"检索完成: {len(state['context_ids'])} 个上下文，已获取答案")
            else:
                logger.info(f"检索完成: {len(state['context_ids'])} 个上下文，答案为空，将在下一步生成")
        except Exception as e:
            logger.error(f"检索失败: {e}")
            state["error"] = str(e)
        
        return state
    
    @track_performance("generate_answer")
    def _generate_answer(self, state: AgentState) -> AgentState:
        """
        生成答案节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 如果 LightRAG 已经生成了答案，直接使用
            answer = state.get("answer", "")
            if answer:
                logger.info("使用 LightRAG 生成的答案")
                return state
            
            # 如果答案为空，基于上下文重新生成
            query = state.get("query", "")
            contexts = state.get("documents", [])
            
            if not contexts:
                # 如果没有上下文，生成一个提示性答案
                state["answer"] = "抱歉，我没有找到相关的上下文信息来回答这个问题。请尝试使用更具体的关键词，或者确保知识库中已包含相关信息。"
                logger.warning("没有上下文信息，返回提示性答案")
                return state
            
            # 构建 prompt
            context_text = "\n\n".join(contexts[:5])  # 取前 5 个上下文
            prompt = f"""
基于以下上下文回答用户问题。如果上下文中没有相关信息，请说明。

上下文：
{context_text}

问题：{query}

答案：
"""
            
            response = self.model_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            state["answer"] = answer
            
            logger.info("答案生成完成（基于上下文）")
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            state["error"] = str(e)
        
        return state
    
    def query(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        执行查询
        
        Args:
            query: 查询文本
            stream: 是否流式返回
            
        Returns:
            查询结果字典
        """
        # 初始化状态
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": query}],
            "query": query,
            "intent": "",
            "need_github_ingest": False,
            "need_preprocess": False,
            "github_repo_urls": [],
            "documents": [],
            "context_ids": [],
            "answer": "",
            "error": None
        }
        
        # 运行 Agent
        try:
            final_state = self.app.invoke(initial_state)
            
            if final_state.get("error"):
                return {
                    "success": False,
                    "error": final_state["error"],
                    "answer": ""
                }
            
            return {
                "success": True,
                "answer": final_state.get("answer", ""),
                "context_ids": final_state.get("context_ids", []),
                "intent": final_state.get("intent", "")
            }
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": ""
            }

