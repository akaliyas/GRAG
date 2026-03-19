"""
Context7 API 连接诊断工具
用于测试 Context7 API 连接状态
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx


def test_context7_api():
    """测试 Context7 API 连接"""

    # 从环境变量读取 API Key
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("CONTEXT7_API_KEY")

    print("=" * 60)
    print("Context7 API 连接诊断")
    print("=" * 60)

    # 1. 检查 API Key
    print("\n1. 检查 API Key")
    if not api_key:
        print("   ❌ CONTEXT7_API_KEY 未设置")
        print("   💡 请在 .env 文件中设置: CONTEXT7_API_KEY=your_key_here")
        return False

    print(f"   ✅ API Key: {api_key[:20]}...")
    print(f"   格式检查: ", end="")
    if api_key.startswith("ctx7sk-"):
        print("✅ 正确（ctx7sk-格式）")
    else:
        print("⚠️  可能不正确（应以 ctx7sk- 开头）")

    # 2. 测试 API 连接
    print("\n2. 测试 API 连接")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    test_endpoints = [
        {
            "name": "搜索库 (官方文档示例)",
            "url": "https://context7.com/api/v2/libs/search",
            "params": {"libraryName": "next.js", "query": "setup ssr"}
        },
        {
            "name": "获取文档上下文",
            "url": "https://context7.com/api/v2/context",
            "params": {"libraryId": "/vercel/next.js", "query": "setup ssr", "type": "json"}
        }
    ]

    for endpoint in test_endpoints:
        print(f"\n   测试: {endpoint['name']}")
        print(f"   URL: {endpoint['url']}")

        try:
            response = httpx.get(
                endpoint['url'],
                headers=headers,
                params=endpoint['params'],
                timeout=30
            )

            print(f"   状态码: {response.status_code}")

            if response.status_code == 200:
                print("   ✅ 连接成功!")
                try:
                    data = response.json()
                    print(f"   响应类型: {type(data).__name__}")
                    if isinstance(data, list):
                        print(f"   返回 {len(data)} 条结果")
                        if len(data) > 0:
                            print(f"   示例结果: {str(data[0])[:100]}...")
                    else:
                        print(f"   响应数据: {str(data)[:200]}...")
                    return True
                except Exception as e:
                    print(f"   JSON 解析失败: {e}")
                    print(f"   原始响应: {response.text[:200]}...")
                    return True  # 仍然认为是成功的，因为 HTTP 200
            elif response.status_code == 401:
                print("   ❌ 401 Unauthorized: API Key 无效")
                print("   💡 请检查 API Key 是否正确")
            elif response.status_code == 403:
                print("   ❌ 403 Forbidden: 访问被拒绝")
                print("   💡 可能原因:")
                print("      1. API Key 未激活（新创建的需要等待几分钟）")
                print("      2. API Key 权限不足")
                print("      3. IP 地址被限制")
                print("      4. 账户需要验证")
            elif response.status_code == 429:
                print("   ⚠️  429 Too Many Requests: 速率限制")
                print("   💡 请稍后再试")
            elif response.status_code == 404:
                print("   ⚠️  404 Not Found: 资源不存在")
            else:
                print(f"   ⚠️  其他错误: {response.status_code}")
                print(f"   响应: {response.text[:200]}")

        except httpx.ConnectTimeout:
            print("   ❌ 连接超时")
            print("   💡 请检查网络连接")
        except httpx.ConnectError as e:
            print(f"   ❌ 连接失败: {e}")
            print("   💡 请检查网络连接或防火墙设置")
        except Exception as e:
            print(f"   ❌ 错误: {type(e).__name__}: {e}")

    # 3. 建议
    print("\n3. 故障排查建议")
    print("   如果连续遇到 403 错误，请尝试:")
    print("   a) 访问 https://context7.com/dashboard 确认 API Key 状态")
    print("   b) 检查账户是否需要验证")
    print("   c) 尝试重新生成 API Key")
    print("   d) 检查是否有地区限制")
    print("   e) 如果是新创建的 API Key，等待 5-10 分钟后再试")

    return False


if __name__ == "__main__":
    success = test_context7_api()

    print("\n" + "=" * 60)
    if success:
        print("✅ Context7 API 连接成功")
    else:
        print("❌ Context7 API 连接失败")
        print("\n💡 Context7 是可选功能，不影响 GRAG 主流程")
    print("=" * 60)

    sys.exit(0 if success else 1)
