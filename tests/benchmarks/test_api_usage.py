"""
API使用测试

测试Docker API相关的场景。
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from .benchmark_base import BenchmarkTest


class APIUsageTest(BenchmarkTest):
    """API使用测试"""

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """获取API使用测试场景"""
        scenarios_file = Path(__file__).parent / "scenarios" / "docker_scenarios.json"

        with open(scenarios_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("scenarios", [])
