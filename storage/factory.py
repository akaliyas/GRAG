"""
存储工厂 - 根据配置创建存储实例

此模块提供统一的存储实例创建接口，支持依赖注入模式。
根据配置自动选择合适的存储后端实现。

使用示例：
    factory = StorageFactory(config)
    cache_storage = factory.create_cache_storage()
    graph_storage = factory.create_graph_storage()
"""
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from config.config_manager import get_config
from storage.interface import ICacheStorage, IGraphStorage, IKnowledgeStorage, StorageBackend

logger = logging.getLogger(__name__)


class StorageFactory:
    """
    存储工厂

    根据配置创建存储实例，实现依赖注入。

    Attributes:
        config: 配置管理器实例
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化存储工厂

        Args:
            config: 配置字典，如果为 None 则从默认位置加载
        """
        if config is None:
            config_obj = get_config()
            self.config = config_obj.config
        else:
            self.config = config

        logger.info("存储工厂初始化完成")

    def _get_storage_config(self, storage_type: str) -> Dict[str, Any]:
        """
        获取存储配置

        Args:
            storage_type: 存储类型 (cache, graph, knowledge)

        Returns:
            存储配置字典
        """
        # 获取存储部分配置
        storage_config = self.config.get('storage', {})

        # 获取特定存储类型的配置
        type_config = storage_config.get(f'{storage_type}_storage', {})

        # 合并默认配置
        defaults = self._get_default_config(storage_type)
        return {**defaults, **type_config}

    def _get_default_config(self, storage_type: str) -> Dict[str, Any]:
        """获取默认配置"""
        # 检查部署模式
        deployment_mode = os.environ.get('DEPLOYMENT_MODE', 'local')

        if storage_type == 'cache':
            return {
                'backend': StorageBackend.JSON,
                'cache_dir': './data/cache',
            }
        elif storage_type == 'graph':
            # 根据部署模式选择默认图存储
            if deployment_mode == 'local':
                return {
                    'backend': StorageBackend.JSON,
                    'data_dir': './rag_storage',
                }
            else:
                return {
                    'backend': StorageBackend.NEO4J,
                }
        elif storage_type == 'knowledge':
            return {
                'backend': StorageBackend.JSON,
                'data_dir': './rag_storage',
            }
        return {}

    def create_cache_storage(self) -> ICacheStorage:
        """
        创建缓存存储实例

        根据配置选择存储后端：
        - json: JSONCacheStorage (本地应用)
        - postgresql: PostgreSQLCacheStorage (Docker/云)
        - redis: RedisCacheStorage (高性能)

        Returns:
            缓存存储实例
        """
        config = self._get_storage_config('cache')
        backend = config.get('backend', StorageBackend.JSON).lower()

        logger.info(f"创建缓存存储: {backend}")

        if backend == StorageBackend.JSON or backend == 'json':
            from storage.json_cache_storage import JSONCacheStorage
            return JSONCacheStorage(cache_dir=config.get('cache_dir', './data/cache'))

        elif backend == StorageBackend.POSTGRESQL or backend == 'postgresql':
            try:
                from storage.postgres_cache_storage import PostgreSQLCacheStorage
                db_config = self.config.get('database', {}).get('postgresql', {})
                return PostgreSQLCacheStorage(
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    database=db_config.get('database', 'grag_db'),
                    user=db_config.get('user', 'grag_user'),
                    password=db_config.get('password', '')
                )
            except ImportError:
                logger.warning(
                    "PostgreSQL 存储实现不可用，回退到 JSON 存储。"
                    "要启用 PostgreSQL，请实现 storage/postgres_cache_storage.py"
                )
                from storage.json_cache_storage import JSONCacheStorage
                return JSONCacheStorage(cache_dir=config.get('cache_dir', './data/cache'))

        elif backend == 'redis':
            from storage.redis_cache_storage import RedisCacheStorage
            redis_config = config.get('redis', {})
            return RedisCacheStorage(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password', None)
            )

        else:
            raise ValueError(f"不支持的缓存存储后端: {backend}")

    def create_graph_storage(self) -> Optional[IGraphStorage]:
        """
        创建图存储实例

        Returns:
            图存储实例，如果不需要则返回 None
        """
        config = self._get_storage_config('graph')
        backend = config.get('backend', StorageBackend.NEO4J).lower()

        logger.info(f"创建图存储: {backend}")

        if backend == StorageBackend.NEO4J or backend == 'neo4j':
            try:
                from storage.neo4j_graph_storage import Neo4jGraphStorage
                db_config = self.config.get('database', {}).get('neo4j', {})
                return Neo4jGraphStorage(
                    uri=db_config.get('uri', 'neo4j://localhost:7687'),
                    username=db_config.get('username', 'neo4j'),
                    password=db_config.get('password', 'neo4j_password')
                )
            except ImportError:
                logger.warning(
                    "Neo4j 存储实现不可用，回退到 JSON 存储。"
                    "要启用 Neo4j，请实现 storage/neo4j_graph_storage.py"
                )
                from storage.json_graph_storage import JSONGraphStorage
                return JSONGraphStorage(data_dir=config.get('data_dir', './rag_storage'))

        elif backend == StorageBackend.JSON or backend == 'json':
            from storage.json_graph_storage import JSONGraphStorage
            data_dir = config.get('data_dir', './rag_storage')
            return JSONGraphStorage(data_dir=data_dir)

        else:
            logger.warning(f"图存储后端 '{backend}' 暂未实现")
            return None

    def create_knowledge_storage(self) -> Optional[IKnowledgeStorage]:
        """
        创建知识库存储实例

        Returns:
            知识库存储实例，如果不需要则返回 None
        """
        config = self._get_storage_config('knowledge')
        backend = config.get('backend', StorageBackend.JSON).lower()

        logger.info(f"创建知识库存储: {backend}")

        if backend == StorageBackend.JSON or backend == 'json':
            from storage.json_knowledge_storage import JSONKnowledgeStorage
            data_dir = config.get('data_dir', './rag_storage')
            return JSONKnowledgeStorage(data_dir=data_dir)

        else:
            logger.warning(f"知识库存储后端 '{backend}' 暂未实现")
            return None


def create_storage_factory(preset: str = 'auto') -> StorageFactory:
    """
    便捷函数：创建存储工厂

    Args:
        preset: 预设配置 ('auto', 'local', 'docker')

    Returns:
        存储工厂实例
    """
    if preset == 'auto':
        # 自动检测环境
        if os.getenv('DOCKER'):
            preset = 'docker'
        else:
            preset = 'local'

    # 获取基础配置
    config_manager = get_config()
    base_config = config_manager.config.copy()

    # 加载预设配置文件并合并
    if preset == 'local':
        config_file = Path(__file__).parent.parent / 'config' / 'presets' / 'local.yaml'
    elif preset == 'docker':
        config_file = Path(__file__).parent.parent / 'config' / 'presets' / 'docker.yaml'
    else:
        config_file = None

    if config_file and config_file.exists():
        logger.info(f"加载预设配置: {preset} ({config_file})")
        with open(config_file, 'r', encoding='utf-8') as f:
            preset_config = yaml.safe_load(f)

        # 深度合并配置（预设配置覆盖基础配置）
        merged_config = _deep_merge(base_config, preset_config)

        # 再次解析环境变量（预设配置可能包含新的环境变量）
        merged_config = _resolve_env_vars(merged_config)

        return StorageFactory(config=merged_config)
    else:
        logger.info(f"使用默认配置（未找到预设配置文件: {config_file}）")
        return StorageFactory(config=base_config)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_env_vars(obj: Any) -> Any:
    """
    递归解析环境变量
    支持格式: ${VAR_NAME:default_value}
    """
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # 检查是否是环境变量格式
        if obj.startswith("${") and obj.endswith("}"):
            # 解析 ${VAR_NAME:default_value} 格式
            var_expr = obj[2:-1]
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                resolved = os.getenv(var_name.strip(), default_value.strip())
                return resolved
            else:
                var_value = os.getenv(var_expr.strip())
                if var_value is None:
                    # 使用空字符串作为默认值（与 ConfigManager 行为一致）
                    return ""
                return var_value
        # 如果字符串包含环境变量格式（可能被其他内容包围），尝试提取
        elif "${" in obj and "}" in obj:
            pattern = r'\$\{([^}]+)\}'
            def replace_env(match):
                var_expr = match.group(1)
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    return os.getenv(var_name.strip(), default_value.strip())
                else:
                    var_value = os.getenv(var_expr.strip())
                    return var_value if var_value is not None else match.group(0)
            return re.sub(pattern, replace_env, obj)
    return obj
