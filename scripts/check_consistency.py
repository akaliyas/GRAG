"""
GRAG 一致性防呆检查脚本
用于验证系统各部分的一致性，防止常见的一致性漏洞。

使用方法:
    uv run scripts/check_consistency.py              # 运行所有检查
    uv run scripts/check_consistency.py --category data  # 仅检查数据层
    uv run scripts/check_consistency.py --category cache # 仅检查缓存
    uv run scripts/check_consistency.py --quick        # 快速检查（关键项）
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config_manager import get_config
from utils.logger import setup_logger

# 初始化日志
logger = setup_logger(__name__)

class ConsistencyChecker:
    """一致性检查器"""

    def __init__(self):
        self.config = get_config()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed_checks: List[str] = []

    def check(self, category: str = "all", quick: bool = False) -> bool:
        """
        执行一致性检查

        Args:
            category: 检查类别 (all, data, config, cache, api)
            quick: 是否仅执行快速检查（关键项）

        Returns:
            检查是否全部通过
        """
        print(f"\n{'='*60}")
        print(f"GRAG 一致性防呆检查")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"检查类别: {category}")
        print(f"快速模式: {'是' if quick else '否'}")
        print(f"{'='*60}\n")

        if category in ["all", "data"]:
            self._check_data_layer(quick)

        if category in ["all", "config"]:
            self._check_config_consistency()

        if category in ["all", "cache"]:
            self._check_cache_consistency()

        if category in ["all", "api"]:
            self._check_api_contract()

        # 输出结果
        self._print_results()

        return len(self.errors) == 0

    def _check_data_layer(self, quick: bool):
        """检查数据与索引层"""
        print("📊 检查数据与索引层...")

        # 1. 检查存储目录是否存在
        working_dir = Path(self.config.get('lightrag', {}).get('working_dir', './rag_storage'))
        if not working_dir.exists():
            self.errors.append(f"❌ LIGHTRAG_WORKING_DIR 不存在: {working_dir}")
        else:
            self.passed_checks.append(f"✅ 工作目录存在: {working_dir}")

        if not quick:
            # 2. 检查索引文件完整性
            vector_store = working_dir / "vector_store_*.json"
            graph_store = working_dir / "graph_store_*.json"
            kv_store = working_dir / "kv_store_*.json"

            if not list(working_dir.glob("vector_store_*.json")):
                self.warnings.append("⚠️  未找到向量存储文件，可能需要运行 ingest")
            else:
                self.passed_checks.append("✅ 向量存储文件存在")

            if not list(working_dir.glob("graph_store_*.json")):
                self.warnings.append("⚠️  未找到图存储文件")
            else:
                self.passed_checks.append("✅ 图存储文件存在")

        # 3. 检查 BM25 索引
        bm25_enabled = self.config.get('lightrag', {}).get('bm25', {}).get('enabled', False)
        if bm25_enabled:
            bm25_dir = Path(self.config.get('lightrag', {}).get('bm25', {}).get('index_dir', './rag_storage/bm25'))
            if not bm25_dir.exists():
                self.errors.append(f"❌ BM25 已启用但索引目录不存在: {bm25_dir}")
            else:
                self.passed_checks.append("✅ BM25 索引目录存在")

    def _check_config_consistency(self):
        """检查配置一致性"""
        print("🔧 检查配置一致性...")

        # 1. 检查存储模式配置
        storage_mode = os.getenv('STORAGE_MODE')
        lightrag_storage = self.config.get('lightrag', {}).get('storage_type')

        if storage_mode == 'json' and lightrag_storage not in ['file', None]:
            self.warnings.append(f"⚠️  STORAGE_MODE=json 但 LIGHTRAG_STORAGE_TYPE={lightrag_storage}，可能不一致")
        else:
            self.passed_checks.append("✅ 存储模式配置一致")

        # 2. 检查 API 密钥配置
        required_keys = []
        model_priority = self.config.get('model_switch', {}).get('priority', 'qwen')

        if model_priority == 'ds':
            required_keys.append('DEEPSEEK_API_KEY')
        elif model_priority == 'qwen':
            required_keys.append('QWEN_API_KEY')

        # 嵌入模型密钥
        required_keys.append('EMBEDDING_API_KEY')

        for key in required_keys:
            if not os.getenv(key):
                self.errors.append(f"❌ 必需的环境变量未设置: {key}")
            else:
                self.passed_checks.append(f"✅ {key} 已设置")

        # 3. 检查嵌入模型配置
        embedding_provider = self.config.get('lightrag', {}).get('embedding_provider')
        if embedding_provider == 'siliconflow':
            base_url = self.config.get('lightrag', {}).get('embedding_base_url', '')
            if 'siliconflow' not in base_url.lower():
                self.warnings.append(f"⚠️  embedding_provider=siliconflow 但 base_url 不匹配: {base_url}")
            else:
                self.passed_checks.append("✅ 嵌入模型配置一致")

    def _check_cache_consistency(self):
        """检查缓存一致性"""
        print("💾 检查缓存一致性...")

        cache_dir = Path("data/cache")
        if not cache_dir.exists():
            self.warnings.append("⚠️  缓存目录不存在: data/cache")
            return

        # 检查缓存文件的年龄
        cache_files = list(cache_dir.glob("*.json"))
        if not cache_files:
            self.passed_checks.append("✅ 缓存目录为空")
            return

        # 检查是否有超过7天的缓存文件
        now = datetime.now()
        old_files = [f for f in cache_files if (now - datetime.fromtimestamp(f.stat().st_mtime)) > timedelta(days=7)]

        if old_files:
            self.warnings.append(f"⚠️  发现 {len(old_files)} 个超过7天的缓存文件，建议清理")
        else:
            self.passed_checks.append("✅ 缓存文件年龄合理")

    def _check_api_contract(self):
        """检查 API 契约一致性"""
        print("🔌 检查 API 契约...")

        # 导入必要的模块
        try:
            from utils.schema import QueryResponse, QueryRequest

            # 检查 QueryResponse 的必需字段
            required_fields = ['success', 'answer', 'context_metadata', 'citations']
            schema_fields = QueryResponse.model_fields.keys()

            for field in required_fields:
                if field in schema_fields:
                    self.passed_checks.append(f"✅ QueryResponse.{field} 存在")
                else:
                    self.errors.append(f"❌ QueryResponse 缺少必需字段: {field}")

        except ImportError as e:
            self.warnings.append(f"⚠️  无法导入 schema 模块: {e}")

    def _print_results(self):
        """打印检查结果"""
        print(f"\n{'='*60}")
        print(f"检查结果")
        print(f"{'='*60}\n")

        # 通过的检查
        if self.passed_checks:
            print(f"✅ 通过 ({len(self.passed_checks)} 项):")
            for check in self.passed_checks:
                print(f"   {check}")
            print()

        # 警告
        if self.warnings:
            print(f"⚠️  警告 ({len(self.warnings)} 项):")
            for warning in self.warnings:
                print(f"   {warning}")
            print()

        # 错误
        if self.errors:
            print(f"❌ 错误 ({len(self.errors)} 项):")
            for error in self.errors:
                print(f"   {error}")
            print()

        print(f"{'='*60}")
        print(f"总结: {len(self.passed_checks)} 通过, {len(self.warnings)} 警告, {len(self.errors)} 错误")
        print(f"{'='*60}\n")

        if self.errors:
            print("🔧 发现一致性错误，请修复后重试。")
            print("   常见解决方案:")
            print("   - 运行 ingest: uv run scripts/pipeline_ingest.py --input <clean_data>")
            print("   - 清理缓存: uv run scripts/clear_cache.py")
            print("   - 验证配置: uv run python scripts/verify_config.py")
        elif self.warnings:
            print("⚠️  发现警告，建议检查但不影响基本功能。")
        else:
            print("✅ 所有一致性检查通过！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GRAG 一致性防呆检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有检查
  uv run scripts/check_consistency.py

  # 仅检查数据层
  uv run scripts/check_consistency.py --category data

  # 快速检查（关键项）
  uv run scripts/check_consistency.py --quick

  # 运行特定类别
  uv run scripts/check_consistency.py --category config
  uv run scripts/check_consistency.py --category cache
  uv run scripts/check_consistency.py --category api
        """
    )

    parser.add_argument(
        '--category',
        choices=['all', 'data', 'config', 'cache', 'api'],
        default='all',
        help='检查类别 (默认: all)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速检查模式（仅检查关键项）'
    )

    args = parser.parse_args()

    # 运行检查
    checker = ConsistencyChecker()
    success = checker.check(category=args.category, quick=args.quick)

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
