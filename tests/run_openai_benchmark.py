"""
OpenAI基准测试运行器

测试OpenAI部署配置场景。
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

from tests.benchmarks.test_openai_deployment import OpenAIDeploymentTest


def main():
    """运行OpenAI基准测试"""
    print("\n" + "=" * 60)
    print("GRAG OpenAI Deployment Benchmark Test")
    print("=" * 60)
    print("Storage mode: JSON (no external database)")
    print("=" * 60)

    # 初始化测试
    test = OpenAIDeploymentTest()

    # 运行所有场景
    print("\n开始运行测试场景...")
    results = test.run_all_scenarios()

    # 生成报告
    report = test.generate_report(
        results,
        output_path="artifacts/test_results/benchmarks/openai_deployment_report.json"
    )

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
