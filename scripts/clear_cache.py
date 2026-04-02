"""
清除GRAG系统缓存

支持清除:
1. LLM响应缓存 (llm_response_cache)
2. 查询结果缓存 (通过PostgreSQL)

使用方法:
    # 仅清除LLM缓存
    uv run scripts/clear_cache.py --llm

    # 仅清除查询缓存
    uv run scripts/clear_cache.py --query

    # 清除所有缓存
    uv run scripts/clear_cache.py --all
"""
import argparse
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clear_llm_cache():
    """清除LLM响应缓存"""
    cache_file = Path("rag_storage/kv_store_llm_response_cache.json")

    if not cache_file.exists():
        logger.info(f"LLM缓存文件不存在: {cache_file}")
        return

    # 备份当前缓存
    backup_file = cache_file.with_suffix(f".json.backup_{cache_file.stat().st_mtime}")

    try:
        # 读取并统计缓存条目
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        cache_count = len(cache_data)
        cache_size = cache_file.stat().st_size

        # 备份
        import shutil
        shutil.copy2(cache_file, backup_file)

        # 清空缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)

        logger.info(f"✅ LLM缓存已清除:")
        logger.info(f"   条目数: {cache_count}")
        logger.info(f"   大小: {cache_size / 1024 / 1024:.2f} MB")
        logger.info(f"   备份: {backup_file.name}")

    except Exception as e:
        logger.error(f"清除LLM缓存失败: {e}")


def clear_query_cache():
    """清除查询结果缓存（PostgreSQL）"""
    try:
        from storage.cache_manager import CacheManager
        import asyncio

        async def clear_cache():
            cache_manager = CacheManager()
            # 注意: 这里需要根据实际的CacheManager API调整
            logger.info("查询缓存清除功能待实现（需要CacheManager.clear_all()方法）")
            logger.info("当前方案: 重启API服务可清空内存缓存")

        asyncio.run(clear_cache())

    except ImportError:
        logger.warning("CacheManager未可用，跳过查询缓存清除")


def main():
    parser = argparse.ArgumentParser(description="清除GRAG系统缓存")
    parser.add_argument("--llm", action="store_true", help="清除LLM响应缓存")
    parser.add_argument("--query", action="store_true", help="清除查询结果缓存")
    parser.add_argument("--all", action="store_true", help="清除所有缓存")

    args = parser.parse_args()

    if not any([args.llm, args.query, args.all]):
        parser.print_help()
        return

    logger.info("=" * 60)
    logger.info("GRAG 缓存清理工具")
    logger.info("=" * 60)

    if args.all or args.llm:
        logger.info("\n[1/2] 清除LLM响应缓存...")
        clear_llm_cache()

    if args.all or args.query:
        logger.info("\n[2/2] 清除查询结果缓存...")
        clear_query_cache()

    logger.info("\n" + "=" * 60)
    logger.info("缓存清理完成")
    logger.info("=" * 60)
    logger.info("\n提示:")
    logger.info("  - API服务内存缓存: 重启服务清空")
    logger.info("  - PostgreSQL查询缓存: 待实现")


if __name__ == "__main__":
    main()
