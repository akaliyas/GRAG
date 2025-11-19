"""
LightRAG 封装模块
直接使用 LightRAG API，提供统一的接口
"""
import logging
from typing import List, Dict, Any, Optional

try:
    from lightrag import LightRAG, QueryParam
except ImportError:
    LightRAG = None
    QueryParam = None

from config.config_manager import get_config
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LightRAGWrapper:
    """LightRAG 封装类"""
    
    def __init__(self, model_manager: ModelManager):
        """
        初始化 LightRAG 封装
        
        Args:
            model_manager: 模型管理器实例
        """
        if LightRAG is None:
            raise ImportError("需要安装 lightrag 库: pip install lightrag")
        
        config = get_config()
        lightrag_config = config.get_lightrag_config()
        
        # 获取 LLM 函数（适配 LightRAG）
        self.model_manager = model_manager
        llm_func = self._create_llm_func()
        
        # 获取存储配置
        storage_type = lightrag_config.get('storage_type', 'postgresql')
        db_config = config.get_database_config()
        
        # 构建存储连接字符串
        if storage_type == 'postgresql':
            storage_uri = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
        else:
            # Neo4j 或其他存储
            storage_uri = lightrag_config.get('storage_uri', '')
        
        # 初始化 LightRAG
        # TODO: 根据实际 LightRAG API 调整初始化参数
        self.rag = LightRAG(
            llm_model_func=llm_func,
            storage_type=storage_type,
            storage_uri=storage_uri,
            # 自定义 prompt（TODO: 需要根据实际需求调整）
            entity_extraction_prompt=lightrag_config.get('entity_extraction_prompt', ''),
            relation_extraction_prompt=lightrag_config.get('relation_extraction_prompt', '')
        )
        
        logger.info(f"LightRAG 已初始化，存储类型: {storage_type}")
    
    def _create_llm_func(self):
        """
        创建 LLM 函数，适配 LightRAG
        
        Returns:
            LLM 函数
        """
        def llm_func(messages: List[Dict[str, str]], **kwargs) -> str:
            """
            LightRAG 需要的 LLM 函数格式
            
            Args:
                messages: 消息列表
                **kwargs: 其他参数
                
            Returns:
                模型生成的文本
            """
            try:
                response = self.model_manager.chat_completion(
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 2000),
                    stream=False
                )
                
                # 提取响应文本
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                raise
        
        return llm_func
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表（可选）
        """
        try:
            # TODO: 根据实际 LightRAG API 调整
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                self.rag.add_document(doc, metadata=metadata)
            
            logger.info(f"已添加 {len(documents)} 个文档到知识库")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def query(
        self,
        query: str,
        mode: str = "hybrid",  # "global", "local", "hybrid"
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        查询知识库
        
        Args:
            query: 查询文本
            mode: 检索模式（global/local/hybrid）
            top_k: 返回的 top-k 结果数
            
        Returns:
            查询结果字典，包含 answer, contexts, entities 等
        """
        try:
            # TODO: 根据实际 LightRAG API 调整查询参数
            query_param = QueryParam(
                query=query,
                mode=mode,
                top_k=top_k
            )
            
            result = self.rag.query(query_param)
            
            # 格式化返回结果
            return {
                'answer': result.get('answer', ''),
                'contexts': result.get('contexts', []),
                'entities': result.get('entities', []),
                'relations': result.get('relations', []),
                'context_ids': result.get('context_ids', [])
            }
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise
    
    def get_entity_context(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        获取实体的上下文信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            实体上下文列表
        """
        try:
            # TODO: 根据实际 LightRAG API 调整
            return self.rag.get_entity_context(entity_name)
        except Exception as e:
            logger.error(f"获取实体上下文失败: {e}")
            return []

