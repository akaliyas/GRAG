#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编码配置工具 - 解决Windows平台中文乱码问题

确保Python在不同平台下正确使用UTF-8编码进行输入输出操作。
"""

import sys
import os
import io


def ensure_utf8_encoding():
    """
    配置系统使用UTF-8编码

    主要解决Windows平台下的中文输出乱码问题:
    - Windows默认使用GBK编码 (cp936)
    - 但现代Python应用推荐使用UTF-8

    Returns:
        bool: 是否成功配置UTF-8编码
    """
    if sys.platform != 'win32':
        # 非Windows平台通常已经是UTF-8,无需处理
        return True

    try:
        # 方法1: 设置环境变量 (Python 3.7+)
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 方法2: 使用reconfigure方法 (Python 3.7+)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            return True

        # 方法3: Python 3.6及更早版本的回退方案
        # 重新包装stdout和stderr
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace'
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                errors='replace'
            )
            return True

        return False

    except Exception as e:
        # 如果配置失败,至少确保程序能继续运行
        print(f"Warning: Failed to configure UTF-8 encoding: {e}", file=sys.stderr)
        return False


def get_encoding_info():
    """
    获取当前编码信息 (用于调试)

    Returns:
        dict: 包含各种编码相关信息的字典
    """
    info = {
        'platform': sys.platform,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'stdout_encoding': sys.stdout.encoding if hasattr(sys.stdout, 'encoding') else 'unknown',
        'stderr_encoding': sys.stderr.encoding if hasattr(sys.stderr, 'encoding') else 'unknown',
        'default_encoding': sys.getdefaultencoding(),
        'filesystem_encoding': sys.getfilesystemencoding(),
        'locale_preferred_encoding': None,
    }

    try:
        import locale
        info['locale_preferred_encoding'] = locale.getpreferredencoding()
    except:
        pass

    return info


def print_encoding_info():
    """打印当前编码信息 (用于调试)"""
    info = get_encoding_info()

    print("\n" + "=" * 60)
    print("编码信息 (Encoding Info)")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key:30s}: {value}")
    print("=" * 60 + "\n")


# 自动执行编码配置 (在导入此模块时)
if __name__ != "__main__":
    ensure_utf8_encoding()


if __name__ == "__main__":
    # 测试编码配置
    print("Before configuration:")
    print_encoding_info()

    success = ensure_utf8_encoding()
    print(f"Configuration {'successful' if success else 'failed'}")

    print("\nAfter configuration:")
    print_encoding_info()

    # 测试中文输出
    print("\n中文测试 (Chinese Test):")
    print("你好，世界！")
    print("测试UTF-8编码: 你好世界 🌍")
    print("特殊字符: £ € ¥ © ®")
