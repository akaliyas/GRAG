"""
FastAPI 主应用
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config_manager import get_config
from utils.logger import setup_logger
from api.routes import router, init_dependencies
from agent.grag_agent import GRAGAgent
from storage.cache_manager import CacheManager
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


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("正在初始化系统组件...")
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        logger.info("模型管理器初始化完成")
        
        # 初始化 LightRAG
        lightrag = LightRAGWrapper(model_manager)
        logger.info("LightRAG 初始化完成")
        
        # 初始化 Agent
        agent = GRAGAgent(model_manager, lightrag)
        logger.info("Agent 初始化完成")
        
        # 初始化缓存管理器
        cache_manager = CacheManager()
        logger.info("缓存管理器初始化完成")
        
        # 初始化全局依赖
        init_dependencies(agent, cache_manager, model_manager)
        logger.info("全局依赖初始化完成")
        
        logger.info("系统初始化完成！")
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
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("debug", True)
    )

