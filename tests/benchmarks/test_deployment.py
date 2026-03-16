"""
部署配置测试

测试Kubernetes部署相关的场景。
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from .benchmark_base import BenchmarkTest


class DeploymentTest(BenchmarkTest):
    """部署配置测试"""

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """获取部署配置测试场景"""
        scenarios_file = Path(__file__).parent / "scenarios" / "kubernetes_scenarios.json"

        with open(scenarios_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("scenarios", [])
