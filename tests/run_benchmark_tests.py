"""
GRAG 真实场景基准测试运行器

运行所有基准测试并生成综合报告。
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.test_deployment import DeploymentTest
from tests.benchmarks.test_api_usage import APIUsageTest
from tests.benchmarks.test_troubleshooting import TroubleshootingTest


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self):
        """初始化运行器"""
        self.results_dir = project_root / "artifacts" / "test_results" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all_tests(
        self,
        category: str = None,
        difficulty: str = None
    ) -> Dict[str, Any]:
        """
        运行所有基准测试

        Args:
            category: 可选类别过滤 (deployment, api_usage, troubleshooting)
            difficulty: 可选难度过滤 (basic, intermediate, advanced)

        Returns:
            综合测试报告
        """
        print("\n" + "=" * 60)
        print("GRAG 真实场景基准测试")
        print("=" * 60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_results = {}

        # 确定要运行的测试
        tests_to_run = []

        if category == "deployment" or category is None:
            tests_to_run.append(("deployment", DeploymentTest))

        if category == "api_usage" or category is None:
            tests_to_run.append(("api_usage", APIUsageTest))

        if category == "troubleshooting" or category is None:
            tests_to_run.append(("troubleshooting", TroubleshootingTest))

        # 运行测试
        for category_name, test_class in tests_to_run:
            print(f"\n{'='*60}")
            print(f"运行类别: {category_name}")
            print(f"{'='*60}")

            test_instance = test_class()
            results = test_instance.run_all_scenarios(difficulty=difficulty)

            # 生成报告
            report = test_instance.generate_report(
                results,
                output_path=self.results_dir / f"{category_name}_report.json"
            )

            all_results[category_name] = report

        # 生成综合报告
        final_report = self._generate_final_report(all_results)

        return final_report

    def _generate_final_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合测试报告

        Args:
            all_results: 所有类别的测试结果

        Returns:
            综合报告
        """
        print("\n" + "=" * 60)
        print("综合测试报告")
        print("=" * 60)

        # 统计总体数据
        total_scenarios = sum(r["total_scenarios"] for r in all_results.values())
        total_success = sum(r["success_count"] for r in all_results.values())
        total_recall = sum(r["recall_count"] for r in all_results.values())

        # 计算平均指标
        avg_quality = sum(r["avg_quality_score"] for r in all_results.values()) / len(all_results) if all_results else 0
        avg_response = sum(r["avg_response_time"] for r in all_results.values()) / len(all_results) if all_results else 0

        # 打印综合摘要
        print(f"\n总体统计:")
        print(f"  总场景数: {total_scenarios}")
        print(f"  成功数: {total_success}")
        print(f"  召回数: {total_recall}")
        print(f"  成功率: {total_success / total_scenarios:.1%}" if total_scenarios > 0 else "  成功率: N/A")
        print(f"  召回率: {total_recall / total_scenarios:.1%}" if total_scenarios > 0 else "  召回率: N/A")
        print(f"  平均质量分: {avg_quality:.1f}/100")
        print(f"  平均响应时间: {avg_response:.3f}秒")

        # 按类别打印
        print(f"\n分类统计:")
        for category, report in all_results.items():
            print(f"  {category}:")
            print(f"    成功率: {report['success_rate']:.1%}")
            print(f"    召回率: {report['recall_rate']:.1%}")
            print(f"    质量分: {report['avg_quality_score']:.1f}/100")
            print(f"    响应时间: {report['avg_response_time']:.3f}秒")

        # 构建最终报告
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_scenarios": total_scenarios,
                "total_success": total_success,
                "total_recall": total_recall,
                "overall_success_rate": total_success / total_scenarios if total_scenarios > 0 else 0,
                "overall_recall_rate": total_recall / total_scenarios if total_scenarios > 0 else 0,
                "avg_quality_score": avg_quality,
                "avg_response_time": avg_response
            },
            "categories": all_results
        }

        # 保存综合报告
        report_path = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        print(f"\n综合报告已保存到: {report_path}")

        return final_report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GRAG真实场景基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有测试
  uv run tests/run_benchmark_tests.py --full

  # 运行特定类别
  uv run tests/run_benchmark_tests.py --category deployment
  uv run tests/run_benchmark_tests.py --category api_usage
  uv run tests/run_benchmark_tests.py --category troubleshooting

  # 运行特定难度
  uv run tests/run_benchmark_tests.py --difficulty basic
  uv run tests/run_benchmark_tests.py --difficulty intermediate
  uv run tests/run_benchmark_tests.py --difficulty advanced
        """
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="运行所有测试"
    )

    parser.add_argument(
        "--category",
        choices=["deployment", "api_usage", "troubleshooting"],
        help="运行特定类别的测试"
    )

    parser.add_argument(
        "--difficulty",
        choices=["basic", "intermediate", "advanced"],
        help="运行特定难度的测试"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.full or args.category or args.difficulty:
        # 运行测试
        runner.run_all_tests(
            category=args.category,
            difficulty=args.difficulty
        )
    else:
        # 默认运行所有测试
        parser.print_help()


if __name__ == "__main__":
    main()
