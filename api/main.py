"""
FastAPI 主应用

支持双方案部署：
- 本地应用模式：使用 JSON 文件存储
- Docker 模式：使用 PostgreSQL + Neo4j 存储

通过存储工厂自动选择合适的存储实现。
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径（支持从项目根目录运行）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置UTF-8编码（解决Windows中文乱码）
from utils.encoding import ensure_utf8_encoding
ensure_utf8_encoding()

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config_manager import get_config
from utils.logger import setup_logger
from api.routes import router, init_dependencies
from agent.grag_agent import GRAGAgent
from storage.cache_manager import CacheManager
from storage.factory import StorageFactory, create_storage_factory
from storage.interface import ICacheStorage
from models.model_manager import ModelManager
from knowledge.lightrag_wrapper import LightRAGWrapper

# 设置日志
setup_logger()
logger = logging.getLogger(__name__)

# 获取配置
config = get_config()
api_config = config.get_api_config()

# 创建 FastAPI 应用
app = FastAPI(
    title=api_config.get("title", "GRAG 技术文档智能问答系统"),
    version=api_config.get("version", "1.0.0"),
    debug=api_config.get("debug", True)
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


def detect_deployment_mode() -> str:
    """
    自动检测部署模式

    Returns:
        'local' 或 'docker'
    """
    # 检查环境变量
    if os.getenv('DOCKER'):
        return 'docker'

    # 检查是否在 PyInstaller 打包的环境中运行
    if getattr(sys, 'frozen', False):
        return 'local'

    # 检查数据库连接是否可用
    # 如果 PostgreSQL 可用则使用 docker 模式
    try:
        import psycopg2
        db_config = config.get_database_config().get('postgresql', {})
        conn = psycopg2.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            user=db_config.get('user', ''),
            password=db_config.get('password', ''),
            connect_timeout=2
        )
        conn.close()
        return 'docker'
    except:
        pass

    # 默认使用本地模式
    return 'local'


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("正在初始化系统组件...")

    try:
        # 检测部署模式
        deployment_mode = detect_deployment_mode()
        logger.info(f"检测到部署模式: {deployment_mode}")

        # 创建存储工厂
        storage_factory = create_storage_factory(preset=deployment_mode)

        # 初始化模型管理器
        model_manager = ModelManager()
        logger.info("模型管理器初始化完成")

        # 初始化 LightRAG（根据配置自动选择存储）
        lightrag = LightRAGWrapper(model_manager)
        logger.info(f"LightRAG 初始化完成，存储类型: {lightrag.storage_type}")

        # 初始化 Agent
        agent = GRAGAgent(model_manager, lightrag)
        logger.info("Agent 初始化完成")

        # 初始化缓存管理器（根据存储工厂创建）
        cache_storage = storage_factory.create_cache_storage()
        cache_manager = CacheManager(cache_storage=cache_storage)
        logger.info(f"缓存管理器初始化完成，存储后端: {type(cache_storage).__name__}")

        # 初始化全局依赖
        init_dependencies(agent, cache_manager, model_manager)
        logger.info("全局依赖初始化完成")

        logger.info(f"系统初始化完成！部署模式: {deployment_mode}")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    logger.info("正在关闭系统...")

    try:
        # 关闭缓存管理器
        cache_manager = CacheManager()
        cache_manager.shutdown()
        logger.info("缓存管理器已关闭")

        # 保存指标
        from utils.monitoring import get_metrics_collector
        collector = get_metrics_collector()
        collector.save_metrics()
        logger.info("指标已保存")
    except Exception as e:
        logger.error(f"系统关闭失败: {e}")


@app.get("/")
def root():
    """根路径"""
    return {
        "message": "GRAG 技术文档智能问答系统 API",
        "version": api_config.get("version", "1.0.0"),
        "docs": "/docs",
        "deployment": detect_deployment_mode()
    }


@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "healthy",
        "deployment_mode": detect_deployment_mode()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("debug", True)
    )
