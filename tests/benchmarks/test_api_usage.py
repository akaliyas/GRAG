"""
API使用测试

测试Kubernetes API相关的场景。
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from .benchmark_base import BenchmarkTest


class APIUsageTest(BenchmarkTest):
    """API使用测试"""

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """获取API使用测试场景"""
        scenarios_file = Path(__file__).parent / "scenarios" / "kubernetes_scenarios.json"

        with open(scenarios_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 过滤出 api_usage 类别的场景
        all_scenarios = data.get("scenarios", [])
        return [s for s in all_scenarios if s.get("category") == "api_usage"]
