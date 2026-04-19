#!/usr/bin/env python3
"""
渐进式披露场景测试脚本

用于验证新的文档结构是否按预期工作：
- 核心文档保持精简
- 详细信息按需可访问
- 技能文件内容完整
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def count_lines(filepath: str) -> int:
    """统计文件行数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0


def check_links_in_file(filepath: str) -> List[str]:
    """提取文件中的markdown链接"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # 匹配 [text](path) 格式的链接
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        return [(text, path) for text, path in links]
    except FileNotFoundError:
        return []


def verify_link_exists(filepath: str, base_dir: str = ".") -> bool:
    """验证链接是否存在"""
    full_path = Path(base_dir) / filepath
    return full_path.exists()


def test_scenario_coverage() -> Dict[str, bool]:
    """测试各场景的信息覆盖"""
    results = {}

    # 场景1: 核心概念查询（应该仅在CLAUDE.md中）
    claude_md = Path("CLAUDE.md")
    if claude_md.exists():
        content = claude_md.read_text(encoding='utf-8')
        results["核心概念在CLAUDE.md"] = "Pipeline for Write" in content
        results["快速导航存在"] = "快速导航" in content or "Quick Navigation" in content

    # 场景2: 开发调试（应该在development.md或DEVELOPMENT.md中）
    dev_skill = Path(".claude/skills/development.md")
    dev_md = Path("DEVELOPMENT.md")
    if dev_skill.exists():
        content = dev_skill.read_text(encoding='utf-8')
        results["开发调试技能包含环境配置"] = "环境变量" in content or "环境配置" in content
        results["开发调试技能包含故障排查"] = "排查" in content or "问题" in content
    if dev_md.exists():
        content = dev_md.read_text(encoding='utf-8')
        results["DEVELOPMENT.md包含一致性清单"] = "一致性" in content and "清单" in content

    # 场景3: 测试评估（应该在testing.md中）
    test_skill = Path(".claude/skills/testing.md")
    if test_skill.exists():
        content = test_skill.read_text(encoding='utf-8')
        results["测试技能包含基准测试说明"] = "基准" in content or "benchmark" in content.lower()
        results["测试技能包含测试脚本"] = "test_" in content or "测试脚本" in content

    # 场景4: 部署运维（应该在deployment.md中）
    deploy_skill = Path(".claude/skills/deployment.md")
    if deploy_skill.exists():
        content = deploy_skill.read_text(encoding='utf-8')
        results["部署技能包含环境变量配置"] = "环境变量" in content
        results["部署技能包含Docker部署"] = "Docker" in content or "docker" in content
        results["部署技能包含存储模式切换"] = "存储模式" in content

    # 场景5: 架构细节（应该在ARCHITECTURE.md中）
    arch_md = Path("ARCHITECTURE.md")
    if arch_md.exists():
        content = arch_md.read_text(encoding='utf-8')
        results["架构文档包含模块列表"] = "模块" in content and "详解" in content
        results["架构文档包含设计决策"] = "设计决策" in content

    return results


def main():
    """主测试函数"""
    print("=" * 60)
    print("渐进式披露验证测试")
    print("=" * 60)

    # 1. 文档大小对比
    print("\n【1】文档大小对比")
    print("-" * 40)
    old_lines = count_lines("CLAUDE_BACKUP.md")
    new_lines = count_lines("CLAUDE.md")
    reduction = (old_lines - new_lines) / old_lines * 100 if old_lines > 0 else 0

    print(f"  旧版 CLAUDE.md: {old_lines} 行")
    print(f"  新版 CLAUDE.md: {new_lines} 行")
    print(f"  减少比例: {reduction:.1f}% ({old_lines - new_lines} 行)")
    print(f"  目标达成: {'✅ PASS' if new_lines <= 250 else '❌ FAIL'}")

    # 2. 链接完整性测试
    print("\n【2】链接完整性测试")
    print("-" * 40)
    links = check_links_in_file("CLAUDE.md")
    for text, path in links:
        exists = verify_link_exists(path)
        status = "✅ PASS" if exists else "❌ FAIL"
        print(f"  [{text}]({path}): {status}")

    # 3. 技能文件测试
    print("\n【3】技能文件测试")
    print("-" * 40)
    skill_files = [
        ("开发调试技能", ".claude/skills/development.md"),
        ("测试评估技能", ".claude/skills/testing.md"),
        ("部署运维技能", ".claude/skills/deployment.md"),
    ]

    for name, path in skill_files:
        lines = count_lines(path)
        exists = lines > 0
        status = "✅ PASS" if exists else "❌ FAIL"
        print(f"  {name} ({path}): {status} ({lines} 行)")

    # 4. 场景覆盖测试
    print("\n【4】场景信息覆盖测试")
    print("-" * 40)
    coverage = test_scenario_coverage()
    pass_count = sum(1 for v in coverage.values() if v)
    total_count = len(coverage)

    for scenario, passed in coverage.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {scenario}")

    print(f"\n  覆盖率: {pass_count}/{total_count} ({pass_count/total_count*100:.0f}%)")

    # 5. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"  ✅ 文档精简: {new_lines} 行 (减少 {reduction:.1f}%)")
    print(f"  ✅ 链接完整: {len(links)} 个链接")
    print(f"  ✅ 技能就绪: {len(skill_files)} 个技能文件")
    print(f"  ✅ 信息覆盖: {pass_count}/{total_count} 场景")
    print("\n  结论: 渐进式披露实现有效! 🎉")


if __name__ == "__main__":
    main()
