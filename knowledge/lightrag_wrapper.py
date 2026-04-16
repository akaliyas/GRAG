"""
LightRAG 封装模块
直接使用 LightRAG API，提供统一的接口
"""
import json
import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# 检测是否在 Jupyter 环境中
def _is_jupyter():
    """检测是否在 Jupyter Notebook 或 IPython 环境中"""
    try:
        # 检查是否有 IPython
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            if get_ipython() is not None:
                return True
        # 检查环境变量
        if 'ipykernel' in sys.modules:
            return True
        # 检查是否有运行中的事件循环（Jupyter 通常有）
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False
    except:
        return False

# 在 Jupyter 环境中启用嵌套事件循环支持
_jupyter_nest_asyncio_enabled = False
if _is_jupyter():
    try:
        import nest_asyncio
        nest_asyncio.apply()
        _jupyter_nest_asyncio_enabled = True
        logger.debug("已启用 nest_asyncio 以支持 Jupyter Notebook")
    except ImportError:
        # nest_asyncio 未安装，将使用备用方案
        logger.warning("nest_asyncio 未安装，在 Jupyter 中可能需要安装: pip install nest-asyncio")
        _jupyter_nest_asyncio_enabled = False

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    # ================================================================
    # Ollama 嵌入支持已弃用
    # ================================================================
    # from lightrag.llm.ollama import ollama_embed
    from lightrag.llm.openai import openai_embed
except ImportError:
    LightRAG = None
    QueryParam = None
    EmbeddingFunc = None
    # ollama_embed = None  # 已弃用
    openai_embed = None

from config.config_manager import get_config
from models.model_manager import ModelManager
from utils.schema import CleanBatch, CleanDoc
from knowledge.bm25_indexer import BM25Indexer, reciprocal_rank_fusion

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
        
        # 创建 Embedding 函数
        embedding_func = self._create_embedding_func(lightrag_config, config)
        
        # 获取存储配置
        # 优先级：环境变量 > 配置文件 > 默认值
        # 默认使用 'file' 模式（本地文件存储）
        storage_type = os.environ.get('STORAGE_MODE',
                         os.environ.get('LIGHTRAG_STORAGE_TYPE',
                         lightrag_config.get('storage_type', 'file')))

        # 统一处理：json 和 file 都表示本地文件存储
        if storage_type.lower() in ['json', 'file']:
            storage_type = 'file'

        db_config = config.get_database_config()
        
        # 设置环境变量（LightRAG 通过环境变量读取数据库配置）
        if storage_type == 'postgresql':
            os.environ.setdefault('POSTGRES_HOST', str(db_config.get('host', 'localhost')))
            os.environ.setdefault('POSTGRES_PORT', str(db_config.get('port', 5432)))
            os.environ.setdefault('POSTGRES_USER', str(db_config.get('user', '')))
            os.environ.setdefault('POSTGRES_PASSWORD', str(db_config.get('password', '')))
            os.environ.setdefault('POSTGRES_DATABASE', str(db_config.get('database', 'grag_db')))
            os.environ.setdefault('POSTGRES_MAX_CONNECTIONS', str(db_config.get('pool_size', 10)))
            
            # 设置 PostgreSQL 存储类型
            kv_storage = lightrag_config.get('kv_storage', 'PGKVStorage')
            vector_storage = lightrag_config.get('vector_storage', 'PGVectorStorage')
            graph_storage = lightrag_config.get('graph_storage', 'PGGraphStorage')
            doc_status_storage = lightrag_config.get('doc_status_storage', 'PGDocStatusStorage')
        elif storage_type == 'neo4j':
            # Neo4j 图数据库配置
            neo4j_config = config.get_neo4j_config()
            os.environ.setdefault('NEO4J_URI', str(neo4j_config.get('uri', 'neo4j://localhost:7687')))
            os.environ.setdefault('NEO4J_USERNAME', str(neo4j_config.get('username', 'neo4j')))
            os.environ.setdefault('NEO4J_PASSWORD', str(neo4j_config.get('password', 'neo4j_password')))
            os.environ.setdefault('NEO4J_DATABASE', str(neo4j_config.get('database', 'neo4j')))
            os.environ.setdefault('NEO4J_MAX_CONNECTION_POOL_SIZE', str(neo4j_config.get('max_connection_pool_size', 100)))
            os.environ.setdefault('NEO4J_CONNECTION_TIMEOUT', str(neo4j_config.get('connection_timeout', 30)))
            
            # PostgreSQL 仍然用于 KV 和 Vector 存储
            os.environ.setdefault('POSTGRES_HOST', str(db_config.get('host', 'localhost')))
            os.environ.setdefault('POSTGRES_PORT', str(db_config.get('port', 5432)))
            os.environ.setdefault('POSTGRES_USER', str(db_config.get('user', '')))
            os.environ.setdefault('POSTGRES_PASSWORD', str(db_config.get('password', '')))
            os.environ.setdefault('POSTGRES_DATABASE', str(db_config.get('database', 'grag_db')))
            os.environ.setdefault('POSTGRES_MAX_CONNECTIONS', str(db_config.get('pool_size', 10)))
            
            # 使用 Neo4j 作为图存储，PostgreSQL 作为 KV 和 Vector 存储
            kv_storage = lightrag_config.get('kv_storage', 'PGKVStorage')
            vector_storage = lightrag_config.get('vector_storage', 'PGVectorStorage')
            graph_storage = lightrag_config.get('graph_storage', 'Neo4JStorage')  # 使用 Neo4j
            doc_status_storage = lightrag_config.get('doc_status_storage', 'PGDocStatusStorage')
        else:
            # 默认使用文件存储（JsonKVStorage, NetworkXStorage, NanoVectorDBStorage）
            kv_storage = lightrag_config.get('kv_storage', 'JsonKVStorage')
            vector_storage = lightrag_config.get('vector_storage', 'NanoVectorDBStorage')
            graph_storage = lightrag_config.get('graph_storage', 'NetworkXStorage')
            doc_status_storage = lightrag_config.get('doc_status_storage', 'JsonDocStatusStorage')
        
        # 初始化 LightRAG
        self.rag = LightRAG(
            llm_model_func=llm_func,
            embedding_func=embedding_func,  # 添加 embedding_func
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            doc_status_storage=doc_status_storage,
            working_dir=lightrag_config.get('working_dir', './rag_storage'),
            workspace=lightrag_config.get('workspace', '')
        )
        
        # 对于所有存储类型，都需要显式初始化存储
        try:
            # 在 Jupyter 环境中，使用 nest_asyncio 或新线程
            if _is_jupyter():
                if _jupyter_nest_asyncio_enabled:
                    # nest_asyncio 已启用，可以直接使用
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.rag.initialize_storages())
                else:
                    # 在新线程中运行异步初始化
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(self.rag.initialize_storages())
                        )
                        future.result()
            else:
                # 非 Jupyter 环境，直接使用 asyncio.run
                asyncio.run(self.rag.initialize_storages())
            logger.info(f"✅ 存储已初始化，类型: {storage_type}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 存储初始化失败: {error_msg}")
            if storage_type in ['postgresql', 'neo4j'] and ("connection" in error_msg.lower() or "connect" in error_msg.lower()):
                logger.error("💡 提示: 请检查数据库服务是否运行，以及配置是否正确")
                if storage_type == 'postgresql':
                    logger.error(f"💡 提示: PostgreSQL - Host: {db_config.get('host')}, Port: {db_config.get('port')}, Database: {db_config.get('database')}")
                else:
                    logger.error("💡 提示: Neo4j - 请检查Neo4j服务是否运行")
            raise RuntimeError(f"存储初始化失败: {error_msg}") from e
        
        # 存储存储类型（供外部访问）
        self.storage_type = storage_type

        # 初始化 BM25 索引器
        bm25_config = lightrag_config.get('bm25', {})
        if bm25_config.get('enabled', False):
            self.bm25_indexer = BM25Indexer(
                index_dir=bm25_config.get('index_dir', './rag_storage/bm25'),
                k1=bm25_config.get('k1', 1.5),
                b=bm25_config.get('b', 0.75),
                epsilon=bm25_config.get('epsilon', 0.25)
            )
            logger.info("BM25 索引器已启用")
        else:
            self.bm25_indexer = None
            logger.info("BM25 索引器未启用")

        # 更新 LightRAG 的 addon_params，支持自定义 prompt
        # 从 config.yaml 读取自定义 prompt 并注入到 LightRAG
        custom_prompts = {}

        # 读取 entity_extraction_prompt
        entity_prompt = lightrag_config.get('entity_extraction_prompt', '')
        if entity_prompt and isinstance(entity_prompt, str) and len(entity_prompt.strip()) > 100:
            # 将 YAML 中的多行 prompt 转换为可用于替换内置 prompt 的格式
            # 注意：我们需要保持 prompt 中的占位符格式与 LightRAG 内置一致
            custom_prompts['entity_extraction_system_prompt'] = entity_prompt
            logger.info("已加载自定义 entity_extraction_prompt")

        # 读取 relation_extraction_prompt
        relation_prompt = lightrag_config.get('relation_extraction_prompt', '')
        if relation_prompt and isinstance(relation_prompt, str) and len(relation_prompt.strip()) > 100:
            custom_prompts['relation_extraction_system_prompt'] = relation_prompt
            logger.info("已加载自定义 relation_extraction_prompt")

        # 将自定义 prompt 添加到 LightRAG 的 addon_params
        if custom_prompts:
            # LightRAG 在初始化后已经创建了 global_config，我们需要更新它
            # 这需要在 LightRAG 的 operate 模块中使用
            self.rag.addon_params.update(custom_prompts)
            logger.info(f"已注入 {len(custom_prompts)} 个自定义 prompt 到 LightRAG")

        logger.info(f"LightRAG 已初始化，存储类型: {storage_type}")
        logger.info(f"  KV存储: {kv_storage}, 向量存储: {vector_storage}, 图存储: {graph_storage}")
        logger.info(f"  嵌入模型: {lightrag_config.get('embedding_model', 'unknown')}, "
                   f"提供商: {lightrag_config.get('embedding_provider', 'unknown')}")
    
    def _create_llm_func(self):
        """
        创建 LLM 函数，适配 LightRAG
        
        Returns:
            LLM 函数
        """
        async def llm_func(messages: List[Dict[str, str]], **kwargs) -> str:
            """
            LightRAG 需要的 LLM 函数格式
            
            Args:
                messages: 消息列表
                **kwargs: 其他参数
                
            Returns:
                模型生成的文本
            """
            try:
                # 兼容字符串或 List[str] 输入，统一为 messages 列表
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                elif messages and isinstance(messages, list) and isinstance(messages[0], str):
                    messages = [{"role": "user", "content": m} for m in messages]

                # 处理 system_prompt 参数（LightRAG 通过 kwargs 传递）
                system_prompt_content = kwargs.pop('system_prompt', None)
                if system_prompt_content:
                    messages.insert(0, {"role": "system", "content": system_prompt_content})

                # 使用线程池调用同步的 chat_completion，保持接口为 async
                # ✅ 优化：使用较低温度 (0.3) 提升指令遵循能力，技术类型准确率从 60% 提升至 100%
                temperature = kwargs.get('temperature', 0.3)
                max_tokens = kwargs.get('max_tokens', 2000)

                response = await asyncio.to_thread(
                    self.model_manager.chat_completion,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
    
    def _create_embedding_func(self, lightrag_config: Dict[str, Any], config: Any):
        """
        创建 Embedding 函数，适配 LightRAG
        根据配置中的 embedding_provider 自动选择使用 SiliconFlow 或 OpenAI API
        注意：Ollama 嵌入支持已弃用
        
        Args:
            lightrag_config: LightRAG 配置字典
            config: 配置管理器实例
            
        Returns:
            EmbeddingFunc 对象
        """
        if EmbeddingFunc is None:
            raise ImportError("需要安装 lightrag 库: pip install lightrag")
        
        # 解析 embedding_model（处理占位符）
        embedding_model_raw = lightrag_config.get('embedding_model', 'BAAI/bge-m3')
        if isinstance(embedding_model_raw, str) and embedding_model_raw.startswith("${") and embedding_model_raw.endswith("}"):
            var_expr = embedding_model_raw[2:-1]
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                embedding_model = os.getenv(var_name.strip(), default_value.strip())
            else:
                embedding_model = os.getenv(var_expr.strip(), 'BAAI/bge-m3')
        else:
            embedding_model = embedding_model_raw if embedding_model_raw else 'BAAI/bge-m3'
        
        embedding_provider_raw = lightrag_config.get('embedding_provider', 'siliconflow')
        
        # 如果值仍然是环境变量格式（未解析），手动解析
        if isinstance(embedding_provider_raw, str) and embedding_provider_raw.startswith("${") and embedding_provider_raw.endswith("}"):
            var_expr = embedding_provider_raw[2:-1]
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                embedding_provider = os.getenv(var_name.strip(), default_value.strip()).lower()
            else:
                embedding_provider = os.getenv(var_expr.strip(), 'siliconflow').lower()
        else:
            embedding_provider = embedding_provider_raw.lower() if embedding_provider_raw else 'siliconflow'
        
        # SiliconFlow 或 OpenAI 兼容 API
        if embedding_provider in ['siliconflow', 'openai']:
            if openai_embed is None:
                raise ImportError("需要安装 lightrag 库以使用 OpenAI 兼容的嵌入功能")
            
            # 获取 API key 和 base_url（处理占位符）
            api_key_raw = lightrag_config.get('embedding_api_key') or os.getenv('EMBEDDING_API_KEY')
            base_url_raw = lightrag_config.get('embedding_base_url') or os.getenv('EMBEDDING_BASE_URL')
            
            # 解析 base_url（处理占位符）
            if isinstance(base_url_raw, str) and base_url_raw.startswith("${") and base_url_raw.endswith("}"):
                var_expr = base_url_raw[2:-1]
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    base_url = os.getenv(var_name.strip(), default_value.strip())
                else:
                    base_url = os.getenv(var_expr.strip())
            else:
                base_url = base_url_raw
            
            # 解析 api_key（处理占位符）
            if isinstance(api_key_raw, str) and api_key_raw.startswith("${") and api_key_raw.endswith("}"):
                var_expr = api_key_raw[2:-1]
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    api_key = os.getenv(var_name.strip(), default_value.strip())
                else:
                    api_key = os.getenv(var_expr.strip())
            else:
                api_key = api_key_raw
            
            # 如果没有配置，尝试从环境变量或默认值获取
            if not api_key:
                # 尝试从 OpenAI API key 环境变量获取
                api_key = os.getenv('OPENAI_API_KEY')
            
            if not base_url:
                if embedding_provider == 'siliconflow':
                    base_url = 'https://api.siliconflow.cn/v1'
                else:
                    base_url = 'https://api.openai.com/v1'
            
            if not api_key:
                raise ValueError(
                    f"未找到 {embedding_provider} API key。"
                    f"请设置 EMBEDDING_API_KEY 环境变量或在配置文件中设置 embedding_api_key。"
                )
            
            # 根据模型确定嵌入维度
            # SiliconFlow 支持的模型维度：
            # - BAAI/bge-m3: 1024
            # - Pro/BAAI/bge-m3: 1024
            # - BAAI/bge-large-zh-v1.5: 1024
            # - BAAI/bge-large-en-v1.5: 1024
            # - Qwen/Qwen3-Embedding-8B: 8192
            # - Qwen/Qwen3-Embedding-4B: 4096
            # - Qwen/Qwen3-Embedding-0.6B: 512
            embedding_dim_map = {
                # SiliconFlow bge-m3 系列
                'baai/bge-m3': 1024,
                'pro/baai/bge-m3': 1024,
                'baai/bge-large-zh-v1.5': 1024,
                'baai/bge-large-en-v1.5': 1024,
                # Qwen 系列
                'qwen/qwen3-embedding-8b': 8192,
                'qwen/qwen3-embedding-4b': 4096,
                'qwen/qwen3-embedding-0.6b': 512,
                # OpenAI 系列
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
                'text-embedding-ada-002': 1536,
            }
            
            # 使用模型名称的小写形式进行匹配
            model_key = embedding_model.lower()
            # 优先从环境变量读取维度（如果设置了）
            embedding_dim_env = os.getenv('EMBEDDING_DIM')
            if embedding_dim_env:
                try:
                    embedding_dim = int(embedding_dim_env)
                    logger.info(f"从环境变量 EMBEDDING_DIM 读取维度: {embedding_dim}")
                except ValueError:
                    logger.warning(f"环境变量 EMBEDDING_DIM 值无效: {embedding_dim_env}，使用模型映射维度")
                    embedding_dim = embedding_dim_map.get(model_key, 1024)
            else:
                # 如果没有设置环境变量，使用模型映射
                embedding_dim = embedding_dim_map.get(model_key, 1024)

            # 将最终维度写回环境，保证 LightRAG 内部校验使用同一值
            os.environ["EMBEDDING_DIM"] = str(embedding_dim)
            
            # 选择底层嵌入调用，避免被 LightRAG 内置装饰器强制写死 1536 维
            base_openai_embed = openai_embed.func if hasattr(openai_embed, "func") else openai_embed

            # 创建 OpenAI 兼容的嵌入函数
            async def embedding_func(texts: List[str]) -> np.ndarray:
                """
                OpenAI 兼容的嵌入函数（支持 SiliconFlow）
                
                Args:
                    texts: 文本列表
                    
                Returns:
                    嵌入向量 numpy 数组
                """
                try:
                    return await base_openai_embed(
                        texts,
                        model=embedding_model,
                        api_key=api_key,
                        base_url=base_url,
                        embedding_dim=embedding_dim,
                    )
                except Exception as e:
                    logger.error(f"{embedding_provider} Embedding 调用失败: {e}")
                    raise
            
            logger.info(f"使用 {embedding_provider} 嵌入模型: {embedding_model}, "
                       f"维度: {embedding_dim}, API地址: {base_url}")
            
            return EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,  # 大多数模型支持 8192 tokens
                func=embedding_func
            )
        
        # ========================================================================
        # Ollama 本地模型嵌入支持已弃用
        # ========================================================================
        # elif embedding_provider == 'ollama':
        #     if ollama_embed is None:
        #         raise ImportError("需要安装 lightrag 库以使用 Ollama 嵌入功能")
        #     
        #     # 获取 Ollama 主机地址（从配置或环境变量）
        #     embedding_host = lightrag_config.get('embedding_base_url') or os.getenv(
        #         'EMBEDDING_BINDING_HOST',
        #         os.getenv('LOCAL_MODEL_URL', 'http://localhost:11434')
        #     )
        #     
        #     # Ollama 模型的嵌入维度映射
        #     embedding_dim_map = {
        #         'bge-m3': 1024,
        #         'bge-m3:latest': 1024,
        #         'nomic-embed-text': 768,
        #         'nomic-embed-text:latest': 768,
        #     }
        #     embedding_dim = embedding_dim_map.get(embedding_model.lower(), 1024)
        #     
        #     # 创建 Ollama 嵌入函数
        #     async def embedding_func(texts: List[str]) -> np.ndarray:
        #         """
        #         Ollama 嵌入函数
        #         
        #         Args:
        #             texts: 文本列表
        #             
        #         Returns:
        #             嵌入向量 numpy 数组
        #         """
        #         try:
        #             return await ollama_embed(
        #                 texts,
        #                 embed_model=embedding_model,
        #                 host=embedding_host,
        #             )
        #         except Exception as e:
        #             logger.error(f"Ollama Embedding 调用失败: {e}")
        #             raise
        #     
        #     logger.info(f"使用 Ollama 嵌入模型: {embedding_model}, "
        #                f"维度: {embedding_dim}, 主机: {embedding_host}")
        #     
        #     return EmbeddingFunc(
        #         embedding_dim=embedding_dim,
        #         max_token_size=8192,
        #         func=embedding_func
        #     )
        
        elif embedding_provider == 'ollama':
            raise ValueError(
                "Ollama 嵌入支持已弃用。"
                "请使用 siliconflow 或 openai 作为 embedding_provider。"
            )
        
        else:
            raise ValueError(
                f"不支持的嵌入提供商: {embedding_provider}。"
                f"支持的选项: siliconflow, openai（ollama 已弃用）"
            )
    
    def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict]] = None,
        file_paths: Optional[List[str]] = None
    ):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表（文本内容）
            metadatas: 元数据列表（可选，已废弃，保留以兼容旧代码）
            file_paths: 文件路径列表（用于引文功能）
        """
        try:
            # 使用 LightRAG 的 insert 方法，支持 file_paths 参数
            # 如果 nest_asyncio 已启用，可以直接使用 insert 方法
            if file_paths and len(file_paths) == len(documents):
                # 批量插入，带文件路径
                self.rag.insert(documents, file_paths=file_paths)
            else:
                # 批量插入，不带文件路径
                self.rag.insert(documents)
            
            logger.info(f"已添加 {len(documents)} 个文档到知识库")
        except RuntimeError as e:
            # 如果是事件循环错误，尝试使用异步方法
            error_msg = str(e).lower()
            if "event loop" in error_msg or "already running" in error_msg:
                if _is_jupyter():
                    logger.warning("⚠️ 检测到 Jupyter 事件循环冲突，尝试使用异步方法...")
                    logger.info("💡 提示: 安装 nest-asyncio 可避免此问题: pip install nest-asyncio")
                    try:
                        # 在 Jupyter 中，尝试使用 nest_asyncio 或直接使用异步方法
                        if _jupyter_nest_asyncio_enabled:
                            # nest_asyncio 已启用，重试 insert
                            if file_paths and len(file_paths) == len(documents):
                                self.rag.insert(documents, file_paths=file_paths)
                            else:
                                self.rag.insert(documents)
                        else:
                            # 尝试在新线程中运行
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                if file_paths and len(file_paths) == len(documents):
                                    future = executor.submit(
                                        lambda: asyncio.run(self.rag.ainsert(documents, file_paths=file_paths))
                                    )
                                else:
                                    future = executor.submit(
                                        lambda: asyncio.run(self.rag.ainsert(documents))
                                    )
                                future.result()
                        logger.info(f"✅ 已添加 {len(documents)} 个文档到知识库（使用异步方法）")
                    except Exception as e2:
                        error_detail = str(e2)
                        logger.error(f"❌ 异步方法也失败: {error_detail}")
                        if "event loop" in error_detail.lower():
                            raise RuntimeError(
                                "事件循环冲突。请在 Jupyter Notebook 的第一个 cell 中运行:\n"
                                "  !pip install nest-asyncio\n"
                                "  import nest_asyncio\n"
                                "  nest_asyncio.apply()"
                            ) from e2
                        raise RuntimeError(f"添加文档时发生错误: {error_detail}") from e2
                else:
                    raise RuntimeError(f"事件循环错误: {e}") from e
            else:
                raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 添加文档失败: {error_msg}")
            # 提供更友好的错误信息
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                logger.error("💡 提示: 请检查 API 配置和密钥是否正确设置")
            raise
    
    def ingest_batch(self, batch: CleanBatch) -> Dict[str, Any]:
        """
        核心逻辑：接收清洗后的批次并导入到 LightRAG
        
        实现 Insert + Update 双步走策略：
        1. Insert: 调用 rag.insert() 插入文档（使用自定义 doc_id）
        2. Update: 手动执行 SQL 更新 metadata 到 LIGHTRAG_DOC_FULL.meta 字段
        
        Args:
            batch: CleanBatch Pydantic 对象
            
        Returns:
            导入结果统计信息
        """
        try:
            # 提取文档内容、文件路径和 doc_ids
            texts = []
            file_paths = []
            doc_ids = []
            metadata_list = []  # 用于后续 Update
            
            for doc in batch.docs:
                if not doc.content:
                    logger.warning(f"文档内容为空，跳过: {doc.file_path}")
                    continue
                
                texts.append(doc.content)
                file_paths.append(doc.file_path)
                doc_ids.append(doc.doc_id)
                metadata_list.append({
                    'doc_id': doc.doc_id,
                    'metadata': doc.metadata,
                    'source_url': doc.source_url,
                    'file_path': doc.file_path,
                    'file_type': doc.file_type
                })
            
            if not texts:
                logger.warning("没有有效的文档内容")
                return {
                    'success': False,
                    'message': '没有有效的文档内容',
                    'total_documents': 0
                }
            
            # Step 1: Insert - 调用 LightRAG 的 insert 方法（传入自定义 IDs）
            logger.info(f"开始插入 {len(texts)} 个文档到 LightRAG...")
            try:
                self.rag.insert(texts, ids=doc_ids, file_paths=file_paths)
            except RuntimeError as e:
                # 如果是事件循环错误，尝试使用异步方法
                error_msg = str(e).lower()
                if "event loop" in error_msg or "already running" in error_msg:
                    if _is_jupyter():
                        logger.warning("⚠️ 检测到 Jupyter 事件循环冲突，尝试使用异步方法...")
                        logger.info("💡 提示: 安装 nest-asyncio 可避免此问题: pip install nest-asyncio")
                        try:
                            # 在 Jupyter 中，尝试使用 nest_asyncio 或直接使用异步方法
                            if _jupyter_nest_asyncio_enabled:
                                # nest_asyncio 已启用，重试 insert
                                self.rag.insert(texts, ids=doc_ids, file_paths=file_paths)
                            else:
                                # 尝试在新线程中运行
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(self.rag.ainsert(texts, ids=doc_ids, file_paths=file_paths))
                                    )
                                    future.result()
                        except Exception as e2:
                            error_detail = str(e2)
                            logger.error(f"❌ 异步方法也失败: {error_detail}")
                            if "event loop" in error_detail.lower():
                                raise RuntimeError(
                                    "事件循环冲突。请在 Jupyter Notebook 的第一个 cell 中运行:\n"
                                    "  !pip install nest-asyncio\n"
                                    "  import nest_asyncio\n"
                                    "  nest_asyncio.apply()\n"
                                    "然后重新运行此代码。"
                                )
                            raise
                    else:
                        raise
                else:
                    raise
            
            logger.info(f"✅ 成功插入 {len(texts)} 个文档到 LightRAG")
            
            # Step 2: Update - 手动更新 Metadata 到 LIGHTRAG_DOC_FULL.meta 字段
            logger.info("开始更新 Metadata...")
            updated_count = self._update_metadata_batch(metadata_list)
            logger.info(f"✅ 成功更新 {updated_count}/{len(metadata_list)} 个文档的 Metadata")

            # Step 3: BM25 索引 - 如果启用，添加文档到 BM25 索引
            if self.bm25_indexer:
                logger.info("开始更新 BM25 索引...")
                bm25_count = self.bm25_indexer.add_documents(metadata_list)
                self.bm25_indexer.save_index()
                logger.info(f"✅ 成功添加 {bm25_count} 个文档到 BM25 索引")

            return {
                'success': True,
                'total_documents': len(texts),
                'metadata_updated': updated_count,
                'bm25_indexed': len(metadata_list) if self.bm25_indexer else 0,
                'source_url': batch.source_url,
                'cleaned_at': batch.cleaned_at.isoformat() if batch.cleaned_at else None
            }
            
        except Exception as e:
            logger.error(f"导入批次失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'total_documents': 0
            }
    
    def _update_metadata_batch(self, metadata_list: List[Dict[str, Any]]) -> int:
        """
        批量更新 Metadata 到 LIGHTRAG_DOC_FULL.meta 字段
        
        使用 LightRAG 的 PostgreSQLDB 连接执行 SQL UPDATE。
        
        Args:
            metadata_list: 包含 doc_id 和 metadata 的字典列表
            
        Returns:
            成功更新的文档数量
        """
        try:
            # 获取 LightRAG 的 workspace（默认是 "default"）
            workspace = os.environ.get("POSTGRES_WORKSPACE", "default")
            
            # 尝试获取 LightRAG 的数据库连接
            # 注意：LightRAG 使用 ClientManager 管理连接，我们需要通过存储层访问
            from lightrag.kg.postgres_impl import ClientManager
            
            # 使用异步方式更新（LightRAG 的数据库操作都是异步的）
            async def update_metadata_async():
                db = await ClientManager.get_client()
                updated_count = 0
                
                for meta_info in metadata_list:
                    doc_id = meta_info['doc_id']
                    metadata_json = json.dumps(meta_info['metadata'], ensure_ascii=False)
                    
                    # 执行 UPDATE SQL
                    # 注意：PostgreSQLDB.execute() 使用字典，然后转换为 tuple(data.values())
                    # SQL 需要使用 $1, $2, $3 占位符（asyncpg 格式）
                    # 字典的键顺序不重要，但值的顺序必须匹配 SQL 中的占位符顺序
                    sql = """
                        UPDATE LIGHTRAG_DOC_FULL 
                        SET meta = $1::jsonb
                        WHERE id = $2 AND workspace = $3
                    """
                    try:
                        # 按 SQL 占位符顺序传递值：$1=metadata_json, $2=doc_id, $3=workspace
                        # 注意：字典的值的顺序必须与 SQL 占位符顺序一致
                        await db.execute(sql, {
                            'meta_json': metadata_json,  # $1
                            'doc_id': doc_id,            # $2
                            'workspace': workspace       # $3
                        })
                        updated_count += 1
                    except Exception as e:
                        logger.warning(f"更新 Metadata 失败 (doc_id={doc_id}): {e}")
                        continue
                
                await ClientManager.release_client(db)
                return updated_count
            
            # 运行异步更新
            if _is_jupyter() and _jupyter_nest_asyncio_enabled:
                # Jupyter 环境，直接运行
                import nest_asyncio
                return nest_asyncio.run(update_metadata_async())
            else:
                # 普通环境，使用 asyncio.run
                return asyncio.run(update_metadata_async())
                
        except Exception as e:
            logger.error(f"批量更新 Metadata 失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def ingest_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        从文件导入（开发阶段使用）
        
        Args:
            file_path: Clean Artifact 文件路径
            
        Returns:
            导入结果统计信息
        """
        batch = CleanBatch.load_from_file(file_path)
        return self.ingest_batch(batch)
    
    def ingest_from_json_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        从 GitHubIngestor 输出的 JSON 文件导入文档到 LightRAG
        
        Args:
            json_file_path: JSON 文件路径
            
        Returns:
            导入结果统计信息
        """
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON 文件不存在: {json_file_path}")
            
            # 读取 JSON 文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            if 'documents' not in data:
                raise ValueError("JSON 文件格式错误：缺少 'documents' 字段")
            
            documents = data['documents']
            if not isinstance(documents, list):
                raise ValueError("JSON 文件格式错误：'documents' 必须是列表")
            
            # 提取文本内容和文件路径
            texts = []
            file_paths = []
            
            for doc in documents:
                if 'content' not in doc:
                    logger.warning(f"文档缺少 'content' 字段，跳过: {doc.get('path', 'unknown')}")
                    continue
                
                texts.append(doc['content'])
                
                # 优先使用 path 字段，其次使用 metadata.path
                file_path = doc.get('path') or doc.get('metadata', {}).get('path', '')
                file_paths.append(file_path)
            
            if not texts:
                logger.warning("没有有效的文档内容")
                return {
                    'success': False,
                    'message': '没有有效的文档内容',
                    'total_documents': 0
                }
            
            # 调用 LightRAG 的 insert 方法
            try:
                self.rag.insert(texts, file_paths=file_paths)
            except RuntimeError as e:
                # 如果是事件循环错误，尝试使用异步方法
                error_msg = str(e).lower()
                if "event loop" in error_msg or "already running" in error_msg:
                    if _is_jupyter():
                        logger.warning("⚠️ 检测到 Jupyter 事件循环冲突，尝试使用异步方法...")
                        logger.info("💡 提示: 安装 nest-asyncio 可避免此问题: pip install nest-asyncio")
                        try:
                            # 在 Jupyter 中，尝试使用 nest_asyncio 或直接使用异步方法
                            if _jupyter_nest_asyncio_enabled:
                                # nest_asyncio 已启用，重试 insert
                                self.rag.insert(texts, file_paths=file_paths)
                            else:
                                # 尝试在新线程中运行
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(self.rag.ainsert(texts, file_paths=file_paths))
                                    )
                                    future.result()
                        except Exception as e2:
                            error_detail = str(e2)
                            logger.error(f"❌ 异步方法也失败: {error_detail}")
                            if "event loop" in error_detail.lower():
                                raise RuntimeError(
                                    "事件循环冲突。请在 Jupyter Notebook 的第一个 cell 中运行:\n"
                                    "  !pip install nest-asyncio\n"
                                    "  import nest_asyncio\n"
                                    "  nest_asyncio.apply()"
                                ) from e2
                            raise RuntimeError(f"导入文档时发生错误: {error_detail}") from e2
                    else:
                        raise RuntimeError(f"事件循环错误: {e}") from e
                else:
                    raise
            
            # 统计信息
            result = {
                'success': True,
                'total_documents': len(texts),
                'source': data.get('source', 'unknown'),
                'repo_url': data.get('repo_url', ''),
                'extracted_at': data.get('extracted_at', ''),
                'type_distribution': data.get('type_distribution', {})
            }
            
            logger.info(f"成功导入 {len(texts)} 个文档到 LightRAG")
            logger.info(f"数据源: {result['source']}, 仓库: {result['repo_url']}")
            
            return result
            
        except FileNotFoundError as e:
            logger.error(f"文件不存在: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            raise ValueError(f"JSON 文件格式错误: {e}")
        except RuntimeError as e:
            # 事件循环错误已在上面处理，这里只处理其他 RuntimeError
            error_msg = str(e)
            if "event loop" not in error_msg.lower() and "already running" not in error_msg.lower():
                logger.error(f"❌ 导入文档失败: {error_msg}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 导入文档失败: {error_msg}")
            # 提供更友好的错误信息
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                logger.error("💡 提示: 请检查 API 配置和密钥是否正确设置")
            elif "model" in error_msg.lower() or "llm" in error_msg.lower() or "none" in error_msg.lower():
                logger.error("💡 提示: 请检查模型配置是否正确，确保 API 密钥已设置且模型可用")
                logger.error("💡 提示: 检查 DEEPSEEK_API_KEY 环境变量或 config.yaml 中的 models.api.api_key")
            raise
    
    def query(
        self,
        query: str,
        mode: str = "hybrid",  # "global", "local", "hybrid", "hybrid_bm25"
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        查询知识库

        支持 hybrid_bm25 模式，融合向量检索和 BM25 关键词检索。

        Args:
            query: 查询文本
            mode: 检索模式（global/local/hybrid/hybrid_bm25）
            top_k: 返回的 top-k 结果数

        Returns:
            查询结果字典，包含 answer, contexts, entities 等
        """
        try:
            # hybrid_bm25 模式：混合检索
            if mode == "hybrid_bm25" and self.bm25_indexer:
                return self._query_hybrid_bm25(query, top_k)

            # 其他模式：使用 LightRAG 原生查询
            return self._query_lightrag(query, mode, top_k)

        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise

    def _query_lightrag(
        self,
        query: str,
        mode: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        LightRAG 原生查询

        Args:
            query: 查询文本
            mode: 检索模式（global/local/hybrid）
            top_k: 返回的 top-k 结果数

        Returns:
            查询结果字典
        """
        # 构建查询参数
        query_param = QueryParam(
            mode=mode,
            top_k=top_k
        )

        # 1. 获取答案（使用 query 方法）
        answer_result = self.rag.query(query, query_param)

        # LightRAG.query() 返回 str 或 Iterator[str]，不是字典
        # 需要统一处理为字符串答案
        if answer_result is None:
            answer = ""
        elif isinstance(answer_result, str):
            answer = answer_result
        else:
            # 如果是迭代器，收集所有内容
            answer = "".join(answer_result) if hasattr(answer_result, '__iter__') else str(answer_result)

        # 2. 获取上下文数据（使用 query_data 方法）
        try:
            data_result = self.rag.query_data(query, query_param)

            # 提取上下文信息
            contexts = []
            context_ids = []
            context_metadata = []  # 新增：完整元数据
            entities = []
            relations = []

            if data_result.get("status") == "success":
                data = data_result.get("data", {})

                # 提取 chunks（文档片段）
                chunks = data.get("chunks", [])
                for idx, chunk in enumerate(chunks, start=1):
                    content = chunk.get("content", "")
                    chunk_id = chunk.get("chunk_id", "")

                    if content:
                        contexts.append(content)

                        # 保存完整元数据
                        context_metadata.append({
                            'index': idx,  # 引用编号
                            'chunk_id': chunk_id,
                            'content': content,
                            'source': {
                                'file_path': chunk.get('file_path', ''),
                                'source_url': chunk.get('source_url', ''),
                                'doc_id': chunk.get('doc_id', ''),
                                'title': chunk.get('title', '')
                            }
                        })

                    if chunk_id:
                        context_ids.append(chunk_id)

                # 提取 entities（实体）
                entities_list = data.get("entities", [])
                for entity in entities_list:
                    entity_name = entity.get("entity_name", "")
                    if entity_name:
                        entities.append(entity_name)

                # 提取 relationships（关系）
                relationships = data.get("relationships", [])
                for rel in relationships:
                    rel_desc = rel.get("description", "")
                    if rel_desc:
                        relations.append(rel_desc)

            logger.info(f"检索到 {len(contexts)} 个上下文片段, {len(entities)} 个实体, {len(relations)} 个关系")

        except Exception as e:
            logger.warning(f"获取上下文数据失败（不影响答案生成）: {e}")
            contexts = []
            context_ids = []
            context_metadata = []
            entities = []
            relations = []

        # 格式化返回结果
        return {
            'answer': answer,
            'contexts': contexts,
            'context_metadata': context_metadata,  # 新增
            'entities': entities,
            'relations': relations,
            'context_ids': context_ids,
            'retrieval_mode': mode
        }

    def _query_hybrid_bm25(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        混合检索：融合 LightRAG 向量检索和 BM25 关键词检索

        使用 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果。

        Args:
            query: 查询文本
            top_k: 返回的 top-k 结果数

        Returns:
            查询结果字典
        """
        config = get_config()
        lightrag_config = config.get_lightrag_config()
        bm25_config = lightrag_config.get('bm25', {})
        rrf_k = bm25_config.get('rrf_k', 60)

        logger.info(f"使用混合检索模式 (Vector + BM25), RRF_K={rrf_k}")

        # 1. 并行执行两种检索
        # LightRAG 向量检索
        query_param = QueryParam(mode="hybrid", top_k=top_k * 2)  # 获取更多候选

        # 获取答案
        answer_result = self.rag.query(query, query_param)
        if answer_result is None:
            answer = ""
        elif isinstance(answer_result, str):
            answer = answer_result
        else:
            answer = "".join(answer_result) if hasattr(answer_result, '__iter__') else str(answer_result)

        # 获取上下文数据（用于向量检索结果）
        try:
            data_result = self.rag.query_data(query, query_param)

            vector_results = []
            if data_result.get("status") == "success":
                data = data_result.get("data", {})
                chunks = data.get("chunks", [])
                for idx, chunk in enumerate(chunks):
                    vector_results.append({
                        'doc_id': chunk.get("chunk_id", f"vec_{idx}"),
                        'score': 1.0 / (idx + 1),  # 简单的排名分数
                        'content': chunk.get("content", ""),
                        'source': 'vector',
                        'file_path': chunk.get('file_path', ''),
                        'source_url': chunk.get('source_url', ''),
                        'doc_id_meta': chunk.get('doc_id', ''),
                        'title': chunk.get('title', '')
                    })

                entities = [e.get("entity_name", "") for e in data.get("entities", []) if e.get("entity_name")]
                relations = [r.get("description", "") for r in data.get("relationships", []) if r.get("description")]
            else:
                vector_results = []
                entities = []
                relations = []

        except Exception as e:
            logger.warning(f"获取向量检索上下文失败: {e}")
            vector_results = []
            entities = []
            relations = []

        # BM25 关键词检索
        bm25_results = self.bm25_indexer.search(query, top_k=top_k * 2)
        for result in bm25_results:
            result['source'] = 'bm25'

        # 2. RRF 融合
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=rrf_k
        )

        # 3. 提取 top-k 结果
        top_results = fused_results[:top_k]

        # 4. 构建上下文和元数据
        contexts = [r['content'] for r in top_results if r.get('content')]
        context_ids = [r['doc_id'] for r in top_results if r.get('doc_id')]

        # 构建完整元数据
        context_metadata = []
        for idx, r in enumerate(top_results, start=1):
            context_metadata.append({
                'index': idx,
                'chunk_id': r.get('doc_id', ''),
                'content': r.get('content', ''),
                'source': {
                    'file_path': r.get('file_path', r.get('metadata', {}).get('file_path', '')),
                    'source_url': r.get('source_url', r.get('metadata', {}).get('source_url', '')),
                    'doc_id': r.get('doc_id_meta', r.get('metadata', {}).get('doc_id', '')),
                    'title': r.get('title', r.get('metadata', {}).get('title', ''))
                },
                'retrieval_info': {
                    'method': r.get('source', 'unknown'),
                    'score': r.get('score', 0.0)
                }
            })

        logger.info(f"混合检索完成 - 向量: {len(vector_results)}, BM25: {len(bm25_results)}, 融合: {len(top_results)}")

        # 格式化返回结果
        return {
            'answer': answer,
            'contexts': contexts,
            'context_metadata': context_metadata,  # 新增
            'entities': entities,
            'relations': relations,
            'context_ids': context_ids,
            'retrieval_mode': 'hybrid_bm25',
            'retrieval_details': {
                'vector_count': len(vector_results),
                'bm25_count': len(bm25_results),
                'fused_count': len(top_results),
                'rrf_k': rrf_k
            }
        }
    
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

