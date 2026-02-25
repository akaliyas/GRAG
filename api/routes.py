"""
FastAPI 路由模块
RESTful 风格 API
"""
import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import verify_credentials
from agent.grag_agent import GRAGAgent
from storage.cache_manager import CacheManager
from models.model_manager import ModelManager
from utils.monitoring import get_metrics_collector, track_performance

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["GRAG API"])

# 全局依赖（在实际应用中应该使用依赖注入）
_agent: Optional[GRAGAgent] = None
_cache_manager: Optional[CacheManager] = None
_model_manager: Optional[ModelManager] = None


def get_agent() -> GRAGAgent:
    """获取 Agent 实例"""
    global _agent
    if _agent is None:
        raise HTTPException(status_code=500, detail="Agent 未初始化")
    return _agent


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        raise HTTPException(status_code=500, detail="缓存管理器未初始化")
    return _cache_manager


def get_model_manager() -> ModelManager:
    """获取模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="模型管理器未初始化")
    return _model_manager


# 请求/响应模型
class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    use_cache: bool = True
    stream: bool = False


class QueryResponse(BaseModel):
    """查询响应"""
    success: bool
    answer: str
    context_ids: list = []
    response_time: float
    model_type: str
    from_cache: bool = False
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    """反馈请求"""
    query: str
    is_positive: bool  # True 为正面反馈，False 为负面反馈


class FeedbackResponse(BaseModel):
    """反馈响应"""
    success: bool
    message: str


class ModelSwitchRequest(BaseModel):
    """模型切换请求"""
    model_type: str  # "api" 或 "local"（local 暂时禁用）


class ModelSwitchResponse(BaseModel):
    """模型切换响应"""
    success: bool
    current_model: str
    message: str


@router.post("/query", response_model=QueryResponse)
@track_performance("api_query")
def query(
    request: QueryRequest,
    username: str = Depends(verify_credentials),
    agent: GRAGAgent = Depends(get_agent),
    cache_manager: CacheManager = Depends(get_cache_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    问答查询接口
    
    Args:
        request: 查询请求
        username: 认证用户名
        agent: Agent 实例
        cache_manager: 缓存管理器
        model_manager: 模型管理器
        
    Returns:
        查询响应
    """
    start_time = time.time()
    
    try:
        # 检查缓存
        if request.use_cache:
            cached_result = cache_manager.get_cache(request.query)
            if cached_result:
                logger.info(f"缓存命中: {request.query[:50]}...")
                return QueryResponse(
                    success=True,
                    answer=cached_result['answer'],
                    context_ids=cached_result.get('context_ids', []),
                    response_time=time.time() - start_time,
                    model_type=cached_result.get('model_type', 'unknown'),
                    from_cache=True
                )
        
        # 执行查询
        result = agent.query(request.query, stream=request.stream)
        
        if not result.get("success"):
            return QueryResponse(
                success=False,
                answer="",
                response_time=time.time() - start_time,
                model_type=model_manager.get_current_model_type(),
                error=result.get("error", "查询失败")
            )
        
        response_time = time.time() - start_time
        
        # 保存到缓存
        if request.use_cache:
            cache_manager.set_cache(
                query=request.query,
                answer=result["answer"],
                context_ids=result.get("context_ids", []),
                model_type=model_manager.get_current_model_type(),
                response_time=response_time
            )
        
        # 记录指标
        collector = get_metrics_collector()
        collector.record_api_call("query", response_time, success=True)
        
        return QueryResponse(
            success=True,
            answer=result["answer"],
            context_ids=result.get("context_ids", []),
            response_time=response_time,
            model_type=model_manager.get_current_model_type(),
            from_cache=False
        )
    
    except Exception as e:
        logger.error(f"查询失败: {e}")
        collector = get_metrics_collector()
        collector.record_api_call("query", time.time() - start_time, success=False)
        
        return QueryResponse(
            success=False,
            answer="",
            response_time=time.time() - start_time,
            model_type=model_manager.get_current_model_type(),
            error=str(e)
        )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    username: str = Depends(verify_credentials),
    agent: GRAGAgent = Depends(get_agent),
    cache_manager: CacheManager = Depends(get_cache_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    流式问答查询接口（Server-Sent Events 格式）

    Args:
        request: 查询请求
        username: 认证用户名
        agent: Agent 实例
        cache_manager: 缓存管理器
        model_manager: 模型管理器

    Returns:
        SSE 格式的流式响应，每个 chunk 格式：
        data: {"content": "...", "done": false}
        data: {"content": "", "done": true, "context_ids": [...], "response_time": ...}
    """
    import json
    import asyncio

    async def generate():
        """生成 SSE 格式的流式响应"""
        start_time = time.time()

        try:
            # 检查缓存
            if request.use_cache:
                cached_result = cache_manager.get_cache(request.query)
                if cached_result:
                    logger.info(f"缓存命中: {request.query[:50]}...")
                    # 缓存命中，流式返回完整答案（模拟字符级流式）
                    answer = cached_result['answer']
                    for char in answer:
                        chunk_data = {
                            "content": char,
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)  # 模拟打字效果

                    # 发送结束标记
                    final_chunk = {
                        "content": "",
                        "done": True,
                        "context_ids": cached_result.get('context_ids', []),
                        "response_time": time.time() - start_time,
                        "model_type": cached_result.get('model_type', 'unknown'),
                        "from_cache": True
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    return

            # 执行流式查询
            async for chunk in agent.process_query_stream(request.query):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # 如果是最后一个 chunk，保存到缓存
                if chunk.get("done"):
                    if request.use_cache:
                        cache_manager.set_cache(
                            query=request.query,
                            answer=chunk.get("full_answer", ""),
                            context_ids=chunk.get("context_ids", []),
                            model_type=chunk.get("model_type", model_manager.get_current_model_type()),
                            response_time=chunk.get("response_time", time.time() - start_time)
                        )

                    # 记录指标
                    collector = get_metrics_collector()
                    response_time = chunk.get("response_time", time.time() - start_time)
                    collector.record_api_call("query_stream", response_time, success=True)

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            error_chunk = {
                "content": "",
                "done": True,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

            # 记录失败指标
            collector = get_metrics_collector()
            collector.record_api_call("query_stream", time.time() - start_time, success=False)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/feedback", response_model=FeedbackResponse)
@track_performance("api_feedback")
def feedback(
    request: FeedbackRequest,
    username: str = Depends(verify_credentials),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """
    用户反馈接口
    
    Args:
        request: 反馈请求
        username: 认证用户名
        cache_manager: 缓存管理器
        
    Returns:
        反馈响应
    """
    try:
        cache_manager.update_feedback(request.query, request.is_positive)
        
        feedback_type = "正面" if request.is_positive else "负面"
        return FeedbackResponse(
            success=True,
            message=f"反馈已记录（{feedback_type}）"
        )
    except Exception as e:
        logger.error(f"反馈处理失败: {e}")
        return FeedbackResponse(
            success=False,
            message=f"反馈处理失败: {str(e)}"
        )


@router.post("/model/switch", response_model=ModelSwitchResponse)
@track_performance("api_model_switch")
def switch_model(
    request: ModelSwitchRequest,
    username: str = Depends(verify_credentials),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    切换模型接口
    
    Args:
        request: 模型切换请求
        username: 认证用户名
        model_manager: 模型管理器
        
    Returns:
        切换响应
    """
    try:
        success = model_manager.switch_model(request.model_type)
        
        if success:
            return ModelSwitchResponse(
                success=True,
                current_model=model_manager.get_current_model_type(),
                message=f"已切换到 {request.model_type}"
            )
        else:
            return ModelSwitchResponse(
                success=False,
                current_model=model_manager.get_current_model_type(),
                message=f"无法切换到 {request.model_type}，模型不可用"
            )
    except Exception as e:
        logger.error(f"模型切换失败: {e}")
        return ModelSwitchResponse(
            success=False,
            current_model=model_manager.get_current_model_type(),
            message=f"模型切换失败: {str(e)}"
        )


@router.get("/stats")
def get_stats(
    username: str = Depends(verify_credentials),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """
    获取系统统计信息
    
    Args:
        username: 认证用户名
        cache_manager: 缓存管理器
        
    Returns:
        统计信息字典
    """
    try:
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        cache_stats = cache_manager.get_cache_stats()
        
        return {
            "metrics": metrics,
            "cache": cache_stats
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def init_dependencies(
    agent: GRAGAgent,
    cache_manager: CacheManager,
    model_manager: ModelManager
):
    """
    初始化全局依赖
    
    Args:
        agent: Agent 实例
        cache_manager: 缓存管理器
        model_manager: 模型管理器
    """
    global _agent, _cache_manager, _model_manager
    _agent = agent
    _cache_manager = cache_manager
    _model_manager = model_manager

