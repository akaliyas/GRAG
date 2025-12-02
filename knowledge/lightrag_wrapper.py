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
        storage_type = lightrag_config.get('storage_type', 'postgresql')
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
        
        # 对于 PostgreSQL 存储，需要显式初始化数据库连接
        if storage_type == 'postgresql':
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
                logger.info("✅ PostgreSQL 数据库连接已初始化")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ PostgreSQL 数据库初始化失败: {error_msg}")
                if "connection" in error_msg.lower() or "connect" in error_msg.lower():
                    logger.error("💡 提示: 请检查 PostgreSQL 服务是否运行，以及数据库配置是否正确")
                    logger.error(f"💡 提示: 检查数据库连接 - Host: {db_config.get('host')}, Port: {db_config.get('port')}, Database: {db_config.get('database')}")
                raise RuntimeError(f"数据库初始化失败: {error_msg}") from e
        
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
        
        embedding_model = lightrag_config.get('embedding_model', 'BAAI/bge-m3')
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
            
            # 获取 API key 和 base_url
            api_key = lightrag_config.get('embedding_api_key') or os.getenv('EMBEDDING_API_KEY')
            base_url = lightrag_config.get('embedding_base_url') or os.getenv('EMBEDDING_BASE_URL')
            
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
            embedding_dim = embedding_dim_map.get(model_key, 1024)  # 默认 1024
            
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
                    return await openai_embed(
                        texts,
                        model=embedding_model,
                        api_key=api_key,
                        base_url=base_url,
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
            
            return {
                'success': True,
                'total_documents': len(texts),
                'metadata_updated': updated_count,
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

