"""
简化版基准测试运行器

直接使用JSON存储模式，无需外部数据库。
"""
import sys
import os
from pathlib import Path

# 强制使用JSON文件存储模式
os.environ['STORAGE_MODE'] = 'json'
os.environ['LIGHTRAG_STORAGE_TYPE'] = 'file'
os.environ['LIGHTRAG_GRAPH_STORAGE'] = 'NetworkXStorage'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.test_deployment import DeploymentTest

def main():
    """运行简化的基准测试"""
    print("\n" + "=" * 60)
    print("GRAG Simplified Benchmark Test")
    print("=" * 60)
    print("Storage mode: JSON (no external database required)")
    print("=" * 60)

    # 由于没有实际的知识库数据，我们只运行一个测试场景作为演示
    print("\nWARNING: Knowledge base is empty. This test only validates the framework.")
    print("Please run pipeline to ingest data first.")

    # 初始化测试
    test = DeploymentTest()

    # 获取场景列表
    scenarios = test.get_scenarios()
    print(f"\n测试场景总数: {len(scenarios)}")

    # 显示前3个场景
    print("\n前3个测试场景:")
    for scenario in scenarios[:3]:
        print(f"\n  ID: {scenario['scenario_id']}")
        print(f"  标题: {scenario['title']}")
        print(f"  查询: {scenario['user_query']}")
        print(f"  难度: {scenario['difficulty']}")

    print("\n" + "=" * 60)
    print("Test framework is ready")
    print("\nNext steps:")
    print("  1. Ingest data: uv run scripts/pipeline_ingest.py --input <clean_data.json>")
    print("  2. Run tests: uv run tests/run_benchmark_tests.py --full")
    print("=" * 60)

if __name__ == "__main__":
    main()
