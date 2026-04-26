"""
JSON 文件缓存存储实现

适用于本地应用部署，数据持久化到 JSON 文件。
支持 LRU 清理、质量评分、并发安全等特性。

存储结构：
data/cache/
├── query_cache.json      # 查询缓存数据
├── cache_stats.json      # 缓存统计信息
└── cache_index.json      # LRU 索引（按访问时间排序）
"""
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from storage.interface import ICacheStorage

logger = logging.getLogger(__name__)


class JSONCacheStorage(ICacheStorage):
    """
    JSON 文件缓存存储

    特性：
    - 线程安全（使用文件锁）
    - 自动保存
    - LRU 索引
    - 质量评分

    Attributes:
        cache_dir: 缓存目录路径
        cache_file: 缓存数据文件路径
        stats_file: 统计信息文件路径
        index_file: LRU 索引文件路径
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        初始化 JSON 文件缓存存储

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "query_cache.json"
        self.stats_file = self.cache_dir / "cache_stats.json"
        self.index_file = self.cache_dir / "cache_index.json"

        # 线程锁（防止并发写入）
        self._lock = threading.RLock()

        # 初始化文件
        self._init_files()

        logger.info(f"JSON 文件缓存存储初始化完成: {cache_dir}")

    def _init_files(self):
        """初始化缓存文件"""
        # 缓存数据
        if not self.cache_file.exists():
            self._write_json(self.cache_file, {})

        # 统计信息
        if not self.stats_file.exists():
            stats = {
                "total_count": 0,
                "total_size": 0,
                "hit_count": 0,
                "miss_count": 0,
                "last_cleanup": None
            }
            self._write_json(self.stats_file, stats)

        # LRU 索引
        if not self.index_file.exists():
            self._write_json(self.index_file, [])

    def _read_json(self, file_path: Path) -> Any:
        """读取 JSON 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"读取 JSON 文件失败 {file_path}: {e}")
            if file_path.name == "query_cache.json":
                return {}
            elif file_path.name == "cache_stats.json":
                return {
                    "total_count": 0,
                    "total_size": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "last_cleanup": None
                }
            else:
                return []

    def _write_json(self, file_path: Path, data: Any) -> bool:
        """写入 JSON 文件（原子写入）"""
        try:
            # 先写临时文件，再重命名（原子操作）
            temp_file = file_path.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 重命名（原子操作）
            temp_file.replace(file_path)
            return True
        except Exception as e:
            logger.error(f"写入 JSON 文件失败 {file_path}: {e}")
            return False

    def _get_query_hash(self, query: str) -> str:
        """获取查询的哈希值（用作键）"""
        import hashlib
        return hashlib.sha256(query.encode('utf-8')).hexdigest()

    def get_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取查询缓存
        """
        with self._lock:
            try:
                query_hash = self._get_query_hash(query)
                cache_data = self._read_json(self.cache_file)

                if query_hash not in cache_data:
                    # 记录未命中
                    self._update_stats("miss")
                    return None

                # 获取缓存
                entry = cache_data[query_hash]

                # 更新访问时间（LRU）
                self._update_lru_index(query_hash)

                # 记录命中
                self._update_stats("hit")

                return {
                    "answer": entry.get("answer", ""),
                    "context_ids": entry.get("context_ids", []),
                    "context_metadata": entry.get("context_metadata", []),
                    "model_type": entry.get("model_type", "unknown"),
                    "response_time": entry.get("response_time", 0),
                    "created_at": entry.get("created_at"),
                    "last_accessed_at": entry.get("last_accessed_at")
                }
            except Exception as e:
                logger.error(f"获取缓存失败: {e}")
                self._update_stats("miss")
                return None

    def set_cache(
        self,
        query: str,
        answer: str,
        context_ids: List[str],
        model_type: str,
        response_time: float,
        context_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """设置查询缓存"""
        with self._lock:
            try:
                query_hash = self._get_query_hash(query)
                now = datetime.now().isoformat()

                cache_data = self._read_json(self.cache_file)

                # 新建或更新缓存条目
                cache_data[query_hash] = {
                    "query": query,
                    "query_hash": query_hash,
                    "answer": answer,
                    "context_ids": context_ids,
                    "context_metadata": context_metadata or [],
                    "model_type": model_type,
                    "response_time": response_time,
                    "created_at": cache_data.get(query_hash, {}).get("created_at", now),
                    "last_accessed_at": now,
                    "quality_score": cache_data.get(query_hash, {}).get("quality_score", 0.5),
                    "feedback_count": cache_data.get(query_hash, {}).get("feedback_count", 0),
                    "positive_feedback": cache_data.get(query_hash, {}).get("positive_feedback", 0),
                    "negative_feedback": cache_data.get(query_hash, {}).get("negative_feedback", 0),
                    "access_count": cache_data.get(query_hash, {}).get("access_count", 0) + 1
                }

                # 保存缓存数据
                if not self._write_json(self.cache_file, cache_data):
                    return False

                # 更新 LRU 索引
                self._update_lru_index(query_hash)

                # 更新统计
                self._update_stats("set")

                return True
            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
                return False

    def update_feedback(self, query: str, is_positive: bool) -> bool:
        """更新用户反馈"""
        with self._lock:
            try:
                query_hash = self._get_query_hash(query)
                cache_data = self._read_json(self.cache_file)

                if query_hash not in cache_data:
                    logger.warning(f"缓存条目不存在: {query}")
                    return False

                entry = cache_data[query_hash]
                entry["feedback_count"] = entry.get("feedback_count", 0) + 1

                if is_positive:
                    entry["positive_feedback"] = entry.get("positive_feedback", 0) + 1
                    # 提高质量评分
                    entry["quality_score"] = min(1.0, entry.get("quality_score", 0.5) + 0.1)
                else:
                    entry["negative_feedback"] = entry.get("negative_feedback", 0) + 1
                    # 降低质量评分
                    entry["quality_score"] = max(0.0, entry.get("quality_score", 0.5) - 0.2)

                # 保存
                return self._write_json(self.cache_file, cache_data)
            except Exception as e:
                logger.error(f"更新反馈失败: {e}")
                return False

    def cleanup_lru(self, max_size: int, batch_size: int) -> int:
        """LRU 清理"""
        with self._lock:
            try:
                cache_data = self._read_json(self.cache_file)
                current_size = len(cache_data)

                if current_size <= max_size:
                    return 0

                # 读取 LRU 索引（已按访问时间排序）
                lru_index = self._read_json(self.index_file)

                # 计算需要删除的数量
                to_remove_count = min(batch_size, current_size - max_size)

                # 获取最久未访问的条目
                to_remove = lru_index[:to_remove_count]

                # 从缓存中删除
                for query_hash in to_remove:
                    if query_hash in cache_data:
                        del cache_data[query_hash]

                # 保存缓存数据
                if not self._write_json(self.cache_file, cache_data):
                    return 0

                # 更新 LRU 索引
                lru_index = lru_index[to_remove_count:]
                self._write_json(self.index_file, lru_index)

                # 更新统计
                stats = self._read_json(self.stats_file)
                stats["last_cleanup"] = datetime.now().isoformat()
                self._write_json(self.stats_file, stats)

                logger.info(f"LRU 清理完成: 删除 {to_remove_count} 条缓存")
                return to_remove_count
            except Exception as e:
                logger.error(f"LRU 清理失败: {e}")
                return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            try:
                stats = self._read_json(self.stats_file)
                cache_data = self._read_json(self.cache_file)

                stats["total_count"] = len(cache_data)
                stats["total_size"] = self.cache_file.stat().st_size if self.cache_file.exists() else 0

                return stats
            except Exception as e:
                logger.error(f"获取缓存统计失败: {e}")
                return {
                    "total_count": 0,
                    "total_size": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "last_cleanup": None
                }

    def clear_all(self) -> bool:
        """清空所有缓存"""
        with self._lock:
            try:
                self._write_json(self.cache_file, {})
                self._write_json(self.index_file, [])

                # 重置统计
                stats = {
                    "total_count": 0,
                    "total_size": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "last_cleanup": None
                }
                self._write_json(self.stats_file, stats)

                logger.info("缓存已清空")
                return True
            except Exception as e:
                logger.error(f"清空缓存失败: {e}")
                return False

    def _update_lru_index(self, query_hash: str):
        """更新 LRU 索引"""
        try:
            lru_index = self._read_json(self.index_file)

            # 如果已存在，先移除
            if query_hash in lru_index:
                lru_index.remove(query_hash)

            # 添加到末尾（最新访问）
            lru_index.append(query_hash)

            # 保存（只保留最新的 10000 条索引）
            if len(lru_index) > 10000:
                lru_index = lru_index[-10000:]

            self._write_json(self.index_file, lru_index)
        except Exception as e:
            logger.warning(f"更新 LRU 索引失败: {e}")

    def _update_stats(self, action: str):
        """更新统计信息"""
        try:
            stats = self._read_json(self.stats_file)

            if action == "hit":
                stats["hit_count"] = stats.get("hit_count", 0) + 1
            elif action == "miss":
                stats["miss_count"] = stats.get("miss_count", 0) + 1
            elif action == "set":
                # 设置缓存时不更新计数，在 get_cache_stats 中计算
                pass

            self._write_json(self.stats_file, stats)
        except Exception as e:
            logger.warning(f"更新统计信息失败: {e}")
