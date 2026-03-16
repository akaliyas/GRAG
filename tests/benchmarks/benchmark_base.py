"""
基准测试基类

提供所有基准测试的通用功能和方法。
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.grag_agent import GRAGAgent
from models.model_manager import ModelManager
from knowledge.lightrag_wrapper import LightRAGWrapper


class BenchmarkTest(ABC):
    """基准测试基类"""

    def __init__(self):
        """初始化测试环境"""
        self.model_manager = ModelManager()
        self.lightrag = LightRAGWrapper(self.model_manager)
        self.agent = GRAGAgent(self.model_manager, self.lightrag)
        self.results = []

    @abstractmethod
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """
        获取测试场景列表

        Returns:
            场景定义列表
        """
        pass

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个测试场景

        Args:
            scenario: 场景定义

        Returns:
            测试结果字典
        """
        scenario_id = scenario.get("scenario_id", "unknown")
        query = scenario.get("user_query", "")

        print(f"\n{'='*60}")
        print(f"场景: {scenario_id}")
        print(f"标题: {scenario.get('title', '')}")
        print(f"难度: {scenario.get('difficulty', '')}")
        print(f"{'='*60}")
        print(f"用户查询: {query}")
        print("-" * 60)

        # 执行查询
        start_time = time.time()
        result = self.agent.query(query, stream=False)
        response_time = time.time() - start_time

        # 分析结果
        answer = result.get("answer", "")
        context_ids = result.get("context_ids", [])
        citations = result.get("citations", [])
        citation_info = result.get("citation_info", {})

        print(f"响应时间: {response_time:.3f}秒")
        print(f"上下文数量: {len(context_ids)}")
        print(f"包含引用: {citation_info.get('has_citations', False)}")
        print(f"引用数量: {citation_info.get('citation_count', 0)}")

        print("\n答案:")
        print("-" * 60)
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("-" * 60)

        # 评估结果
        evaluation = self._evaluate_result(scenario, result, response_time)

        return {
            "scenario_id": scenario_id,
            "scenario": scenario,
            "result": result,
            "response_time": response_time,
            "evaluation": evaluation
        }

    def _evaluate_result(
        self,
        scenario: Dict[str, Any],
        query_result: Dict[str, Any],
        response_time: float
    ) -> Dict[str, Any]:
        """
        评估测试结果

        Args:
            scenario: 场景定义
            query_result: 查询结果
            response_time: 响应时间

        Returns:
            评估结果字典
        """
        answer = query_result.get("answer", "")
        context_ids = query_result.get("context_ids", [])
        citation_info = query_result.get("citation_info", {})

        evaluation = {
            "success": query_result.get("success", False),
            "has_answer": len(answer) > 0,
            "has_contexts": len(context_ids) > 0,
            "has_citations": citation_info.get("has_citations", False),
            "citation_count": citation_info.get("citation_count", 0),
            "response_time": response_time,
            "quality_score": 0.0,
            "recall": False,
            "criteria_met": []
        }

        # 基础评分
        if evaluation["success"]:
            evaluation["quality_score"] += 20

        if evaluation["has_answer"]:
            evaluation["quality_score"] += 20

        if evaluation["has_contexts"]:
            evaluation["quality_score"] += 20

        # 引用评分
        min_citations = scenario.get("min_citations", 1)
        if evaluation["citation_count"] >= min_citations:
            evaluation["quality_score"] += 20
            evaluation["criteria_met"].append("min_citations")

        # 质量标准评分
        quality_criteria = scenario.get("quality_criteria", {})

        if quality_criteria.get("contains_command", False):
            if self._check_contains_command(answer):
                evaluation["quality_score"] += 10
                evaluation["criteria_met"].append("contains_command")

        if quality_criteria.get("contains_example", False):
            if self._check_contains_example(answer):
                evaluation["quality_score"] += 10
                evaluation["criteria_met"].append("contains_example")

        if quality_criteria.get("mentions_prerequisites", False):
            if self._check_mentions_prerequisites(answer):
                evaluation["quality_score"] += 10
                evaluation["criteria_met"].append("mentions_prerequisites")

        # 预期主题检查
        expected_topics = scenario.get("expected_topics", [])
        if expected_topics:
            topics_found = 0
            for topic in expected_topics:
                if topic.lower() in answer.lower():
                    topics_found += 1

            topic_coverage = topics_found / len(expected_topics)
            if topic_coverage >= 0.5:
                evaluation["quality_score"] += 10 * topic_coverage
                evaluation["recall"] = True

        # 评分上限100
        evaluation["quality_score"] = min(evaluation["quality_score"], 100)

        return evaluation

    def _check_contains_command(self, answer: str) -> bool:
        """检查答案是否包含命令"""
        command_patterns = ["kubectl", "docker", "git", "curl", "wget", "pip", "npm"]
        return any(pattern in answer.lower() for pattern in command_patterns)

    def _check_contains_example(self, answer: str) -> bool:
        """检查答案是否包含示例"""
        example_patterns = ["```", "例如", "example", "如下", "如下：", "example:"]
        return any(pattern in answer for pattern in example_patterns)

    def _check_mentions_prerequisites(self, answer: str) -> bool:
        """检查答案是否提及前置条件"""
        prereq_patterns = ["需要", "前提", "首先", "ensure", "require", "before", "prerequisite"]
        return any(pattern in answer for pattern in prereq_patterns)

    def run_all_scenarios(self, difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        运行所有测试场景

        Args:
            difficulty: 可选难度过滤 (basic, intermediate, advanced)

        Returns:
            所有测试结果列表
        """
        scenarios = self.get_scenarios()

        if difficulty:
            scenarios = [s for s in scenarios if s.get("difficulty") == difficulty]

        print(f"\n{'='*60}")
        print(f"运行 {self.__class__.__name__}")
        print(f"场景数量: {len(scenarios)}")
        print(f"{'='*60}")

        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        return results

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成测试报告

        Args:
            results: 测试结果列表
            output_path: 可选输出路径

        Returns:
            报告字典
        """
        total = len(results)
        success_count = sum(1 for r in results if r["evaluation"]["success"])
        recall_count = sum(1 for r in results if r["evaluation"]["recall"])
        avg_quality = sum(r["evaluation"]["quality_score"] for r in results) / total if total > 0 else 0
        avg_response_time = sum(r["response_time"] for r in results) / total if total > 0 else 0

        report = {
            "test_class": self.__class__.__name__,
            "total_scenarios": total,
            "success_count": success_count,
            "recall_count": recall_count,
            "success_rate": success_count / total if total > 0 else 0,
            "recall_rate": recall_count / total if total > 0 else 0,
            "avg_quality_score": round(avg_quality, 2),
            "avg_response_time": round(avg_response_time, 3),
            "results": results
        }

        # 打印摘要
        print("\n" + "=" * 60)
        print("测试报告摘要")
        print("=" * 60)
        print(f"总场景数: {total}")
        print(f"成功数: {success_count}")
        print(f"召回数: {recall_count}")
        print(f"成功率: {report['success_rate']:.1%}")
        print(f"召回率: {report['recall_rate']:.1%}")
        print(f"平均质量分: {report['avg_quality_score']:.1f}/100")
        print(f"平均响应时间: {report['avg_response_time']:.3f}秒")

        # 保存报告
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\n报告已保存到: {output_path}")

        return report
