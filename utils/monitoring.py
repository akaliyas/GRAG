"""
性能监控模块
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime
from collections import defaultdict
import threading

from config.config_manager import get_config

logger = logging.getLogger(__name__)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        config = get_config()
        monitoring_config = config.get("monitoring", {})
        
        self.enabled = monitoring_config.get("enabled", True)
        self.metrics_file = monitoring_config.get("metrics_file", "logs/metrics.json")
        self.track_api_calls = monitoring_config.get("track_api_calls", True)
        self.track_response_time = monitoring_config.get("track_response_time", True)
        
        # 指标数据
        self.api_call_count = defaultdict(int)  # API 调用次数
        self.response_times = defaultdict(list)  # 响应时间列表
        self.error_count = defaultdict(int)  # 错误次数
        
        # 锁
        self._lock = threading.Lock()
        
        # 创建日志目录
        if self.metrics_file:
            Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    
    def record_api_call(self, api_name: str, response_time: Optional[float] = None, success: bool = True):
        """
        记录 API 调用
        
        Args:
            api_name: API 名称
            response_time: 响应时间（秒）
            success: 是否成功
        """
        if not self.enabled:
            return
        
        with self._lock:
            if self.track_api_calls:
                self.api_call_count[api_name] += 1
            
            if self.track_response_time and response_time is not None:
                self.response_times[api_name].append(response_time)
            
            if not success:
                self.error_count[api_name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取指标统计
        
        Returns:
            指标字典
        """
        with self._lock:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'api_calls': dict(self.api_call_count),
                'errors': dict(self.error_count),
                'response_times': {}
            }
            
            # 计算响应时间统计
            for api_name, times in self.response_times.items():
                if times:
                    metrics['response_times'][api_name] = {
                        'count': len(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'p95': sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
                    }
            
            return metrics
    
    def save_metrics(self):
        """保存指标到文件"""
        if not self.enabled or not self.metrics_file:
            return
        
        try:
            metrics = self.get_metrics()
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.debug(f"指标已保存到 {self.metrics_file}")
        except Exception as e:
            logger.error(f"保存指标失败: {e}")
    
    def reset(self):
        """重置所有指标"""
        with self._lock:
            self.api_call_count.clear()
            self.response_times.clear()
            self.error_count.clear()


# 全局指标收集器实例
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_performance(api_name: str):
    """
    性能跟踪装饰器
    
    Args:
        api_name: API 名称
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"{api_name} 执行失败: {e}")
                raise
            finally:
                response_time = time.time() - start_time
                collector.record_api_call(api_name, response_time, success)
        
        return wrapper
    return decorator

