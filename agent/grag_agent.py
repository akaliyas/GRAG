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

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """意图类型"""
    QUERY = "query"  # 直接查询
    CRAWL = "crawl"  # 需要爬取数据
    PREPROCESS = "preprocess"  # 需要数据预处理
    UNKNOWN = "unknown"  # 未知意图


if TypedDict:
    class AgentState(TypedDict):
        """Agent 状态"""
        messages: Annotated[List[Dict], "消息列表"]
        intent: str  # 意图类型
        query: str  # 查询文本
        need_crawl: bool  # 是否需要爬取
        need_preprocess: bool  # 是否需要预处理
        crawl_urls: List[str]  # 需要爬取的 URL 列表
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
        
        # 构建 Agent 图
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("GRAG Agent 已初始化")
    
    def _build_graph(self) -> StateGraph:
        """构建 Agent 图"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("intent_recognition", self._intent_recognition)
        graph.add_node("crawl_tool", self._crawl_tool)
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
                "crawl": "crawl_tool",
                "preprocess": "preprocess_tool",
                "retrieve": "retrieve",
                "error": END
            }
        )
        
        graph.add_edge("crawl_tool", "preprocess_tool")
        graph.add_edge("preprocess_tool", "retrieve")
        graph.add_edge("retrieve", "generate_answer")
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
    "intent": "query|crawl|preprocess",
    "need_crawl": true/false,
    "need_preprocess": true/false,
    "crawl_urls": ["url1", "url2"]  // 如果需要爬取
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
            state["need_crawl"] = False
            state["need_preprocess"] = False
            state["crawl_urls"] = []
            
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
        
        if state.get("need_crawl"):
            return "crawl"
        elif state.get("need_preprocess"):
            return "preprocess"
        else:
            return "retrieve"
    
    @track_performance("crawl_tool")
    def _crawl_tool(self, state: AgentState) -> AgentState:
        """
        爬虫工具节点
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            crawl_urls = state.get("crawl_urls", [])
            # TODO: 调用爬虫模块
            # from crawler.crawler import crawl_documents
            # documents = crawl_documents(crawl_urls)
            # state["documents"] = documents
            
            logger.info(f"爬取完成: {len(crawl_urls)} 个 URL")
        except Exception as e:
            logger.error(f"爬取失败: {e}")
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
            
            # 使用 LightRAG 进行检索
            result = self.lightrag.query(query, mode="hybrid", top_k=5)
            
            state["context_ids"] = result.get("context_ids", [])
            state["documents"] = result.get("contexts", [])
            
            logger.info(f"检索完成: {len(state['context_ids'])} 个上下文")
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
            query = state.get("query", "")
            contexts = state.get("documents", [])
            
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
            
            logger.info("答案生成完成")
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
            "need_crawl": False,
            "need_preprocess": False,
            "crawl_urls": [],
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

