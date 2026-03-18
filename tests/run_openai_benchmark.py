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

# 配置UTF-8编码（解决Windows中文乱码）
from utils.encoding import ensure_utf8_encoding
ensure_utf8_encoding()

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

    # 输出详细结果到控制台
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总场景数: {report['total_scenarios']}")
    print(f"成功数: {report['success_count']}")
    print(f"召回数: {report['recall_count']}")
    print(f"成功率: {report['success_rate']*100:.1f}%")
    print(f"召回率: {report['recall_rate']*100:.1f}%")
    print(f"平均质量分: {report['avg_quality_score']:.2f}/100")
    print(f"平均响应时间: {report['avg_response_time']:.2f}秒")

    # 输出每个场景的详细结果
    print("\n" + "=" * 60)
    print("场景详细结果")
    print("=" * 60)

    for result in report['results']:
        scenario = result['scenario']
        eval_result = result['evaluation']

        print(f"\n场景: {scenario['scenario_id']}")
        print(f"标题: {scenario['title']}")
        print(f"难度: {scenario['difficulty']}")
        print(f"用户查询: {scenario['user_query']}")
        print("-" * 60)
        print(f"响应时间: {eval_result['response_time']:.2f}秒")
        print(f"有答案: {'是' if eval_result['has_answer'] else '否'}")
        print(f"有上下文: {'是' if eval_result['has_contexts'] else '否'}")
        print(f"有引用: {'是' if eval_result['has_citations'] else '否'}")
        print(f"引用数量: {eval_result['citation_count']}")
        print(f"质量评分: {eval_result['quality_score']:.2f}/100")
        print(f"召回成功: {'是' if eval_result['recall'] else '否'}")
        print(f"满足标准: {', '.join(eval_result['criteria_met']) if eval_result['criteria_met'] else '无'}")

        # 输出答案的前200字符
        answer = result['result']['answer']
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"\n答案预览:\n{preview}")

    print("\n" + "=" * 60)
    print(f"详细报告已保存至: {report.get('output_path', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
