"""
缓存管理模块
实现 PostgreSQL 缓存表、LRU 清理、质量评分等功能
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, TIMESTAMP, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import threading
import time

from config.config_manager import get_config

logger = logging.getLogger(__name__)
Base = declarative_base()


class QueryCache(Base):
    """查询缓存表模型"""
    __tablename__ = "query_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_ids = Column(ARRAY(String))  # 关联的文档/实体ID列表
    quality_score = Column(Float, default=0.5)
    feedback_count = Column(Integer, default=0)
    positive_feedback = Column(Integer, default=0)
    negative_feedback = Column(Integer, default=0)
    model_type = Column(String(20))
    response_time = Column(Float)
    created_at = Column(TIMESTAMP, default=func.now())
    last_accessed_at = Column(TIMESTAMP, default=func.now(), index=True)
    access_count = Column(Integer, default=1)


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        config = get_config()
        db_config = config.get_database_config()
        cache_config = config.get_cache_config()
        
        # 创建数据库连接
        db_url = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.engine = create_engine(
            db_url,
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_timeout=db_config.get('pool_timeout', 30)
        )
        
        # 创建表（如果不存在）
        Base.metadata.create_all(self.engine)
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # 缓存配置
        self.max_size = cache_config.get('lru', {}).get('max_size', 10000)
        self.cleanup_interval = cache_config.get('lru', {}).get('cleanup_interval', 3600)
        self.cleanup_batch_size = cache_config.get('lru', {}).get('cleanup_batch_size', 100)
        self.low_threshold = cache_config.get('quality', {}).get('low_threshold', 0.3)
        self.high_threshold = cache_config.get('quality', {}).get('high_threshold', 0.7)
        
        # 启动定时清理任务
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """启动定时清理任务"""
        def cleanup_loop():
            while not self._stop_cleanup:
                try:
                    self.cleanup_lru()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"LRU 清理任务出错: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("LRU 定时清理任务已启动")
    
    def _hash_query(self, query: str) -> str:
        """生成查询的 hash 值"""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()
    
    def get_cache(self, query: str) -> Optional[Dict]:
        """
        从缓存中获取查询结果
        
        Args:
            query: 查询文本
            
        Returns:
            缓存结果字典，包含 answer, context_ids 等，如果不存在则返回 None
        """
        query_hash = self._hash_query(query)
        session = self.SessionLocal()
        
        try:
            cache_entry = session.query(QueryCache).filter(
                QueryCache.query_hash == query_hash
            ).first()
            
            if cache_entry:
                # 更新访问时间和次数
                cache_entry.last_accessed_at = datetime.now()
                cache_entry.access_count += 1
                session.commit()
                
                return {
                    'answer': cache_entry.answer,
                    'context_ids': cache_entry.context_ids or [],
                    'quality_score': cache_entry.quality_score,
                    'model_type': cache_entry.model_type,
                    'response_time': cache_entry.response_time
                }
            return None
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def set_cache(
        self,
        query: str,
        answer: str,
        context_ids: List[str],
        model_type: str,
        response_time: float
    ):
        """
        设置缓存
        
        Args:
            query: 查询文本
            answer: 生成的答案
            context_ids: 关联的文档/实体ID列表
            model_type: 使用的模型类型
            response_time: 响应时间（秒）
        """
        query_hash = self._hash_query(query)
        session = self.SessionLocal()
        
        try:
            # 检查是否已存在
            cache_entry = session.query(QueryCache).filter(
                QueryCache.query_hash == query_hash
            ).first()
            
            if cache_entry:
                # 更新现有缓存
                cache_entry.answer = answer
                cache_entry.context_ids = context_ids
                cache_entry.model_type = model_type
                cache_entry.response_time = response_time
                cache_entry.last_accessed_at = datetime.now()
            else:
                # 创建新缓存
                cache_entry = QueryCache(
                    query_hash=query_hash,
                    query_text=query,
                    answer=answer,
                    context_ids=context_ids,
                    model_type=model_type,
                    response_time=response_time
                )
                session.add(cache_entry)
            
            session.commit()
            logger.debug(f"缓存已保存: {query_hash[:8]}...")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            session.rollback()
        finally:
            session.close()
    
    def update_feedback(self, query: str, is_positive: bool):
        """
        更新用户反馈
        
        Args:
            query: 查询文本
            is_positive: 是否为正面反馈
        """
        query_hash = self._hash_query(query)
        session = self.SessionLocal()
        
        try:
            cache_entry = session.query(QueryCache).filter(
                QueryCache.query_hash == query_hash
            ).first()
            
            if cache_entry:
                # 更新反馈计数
                cache_entry.feedback_count += 1
                if is_positive:
                    cache_entry.positive_feedback += 1
                else:
                    cache_entry.negative_feedback += 1
                
                # 更新质量评分
                cache_entry.quality_score = self._calculate_quality_score(cache_entry)
                
                session.commit()
                logger.info(f"反馈已更新: {query_hash[:8]}..., 评分: {cache_entry.quality_score:.2f}")
            else:
                logger.warning(f"未找到缓存条目: {query_hash[:8]}...")
        except Exception as e:
            logger.error(f"更新反馈失败: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _calculate_quality_score(self, cache_entry: QueryCache) -> float:
        """
        计算质量评分
        
        Args:
            cache_entry: 缓存条目
            
        Returns:
            质量评分 (0-1)
        """
        # 基础评分
        quality_score = cache_entry.quality_score
        
        # 根据反馈更新评分
        if cache_entry.positive_feedback > 0:
            # 正面反馈：提升评分
            quality_score = min(1.0, quality_score + 0.1 * (1 - quality_score))
        
        if cache_entry.negative_feedback > 0:
            # 负面反馈：降低评分
            quality_score = max(0.0, quality_score - 0.15 * quality_score)
        
        # 考虑反馈次数权重（反馈越多，权重越大）
        weight = min(1.0, cache_entry.feedback_count / 10)
        final_score = quality_score * weight + 0.5 * (1 - weight)
        
        return max(0.0, min(1.0, final_score))
    
    def cleanup_lru(self):
        """
        执行 LRU 清理
        优先清理低质量、长时间未访问的缓存
        """
        session = self.SessionLocal()
        
        try:
            # 获取当前缓存数量
            total_count = session.query(QueryCache).count()
            
            if total_count <= self.max_size:
                logger.debug(f"缓存数量 ({total_count}) 未超过限制 ({self.max_size})")
                return
            
            # 需要清理的数量
            cleanup_count = total_count - self.max_size + self.cleanup_batch_size
            
            # 优先清理：低质量 + 长时间未访问
            entries_to_delete = session.query(QueryCache).filter(
                QueryCache.quality_score < self.low_threshold
            ).order_by(
                QueryCache.last_accessed_at.asc()
            ).limit(cleanup_count).all()
            
            # 如果还不够，继续清理长时间未访问的
            if len(entries_to_delete) < cleanup_count:
                remaining = cleanup_count - len(entries_to_delete)
                additional_entries = session.query(QueryCache).filter(
                    QueryCache.id.notin_([e.id for e in entries_to_delete])
                ).order_by(
                    QueryCache.last_accessed_at.asc()
                ).limit(remaining).all()
                entries_to_delete.extend(additional_entries)
            
            # 删除缓存条目
            for entry in entries_to_delete:
                session.delete(entry)
            
            session.commit()
            logger.info(f"LRU 清理完成: 删除了 {len(entries_to_delete)} 条缓存")
        except Exception as e:
            logger.error(f"LRU 清理失败: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        session = self.SessionLocal()
        
        try:
            total = session.query(QueryCache).count()
            avg_quality = session.query(func.avg(QueryCache.quality_score)).scalar() or 0.0
            total_feedback = session.query(func.sum(QueryCache.feedback_count)).scalar() or 0
            
            return {
                'total_entries': total,
                'average_quality_score': float(avg_quality),
                'total_feedback': int(total_feedback),
                'max_size': self.max_size
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}
        finally:
            session.close()
    
    def shutdown(self):
        """关闭缓存管理器"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("缓存管理器已关闭")

