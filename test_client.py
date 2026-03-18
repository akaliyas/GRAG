"""
GRAG客户端测试脚本

快速测试GRAG系统的各项功能。
"""
import requests
import json
import time
from typing import Dict, Any

class GRAGClient:
    """GRAG客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"

    def query(
        self,
        query_text: str,
        use_cache: bool = False,
        stream: bool = False
    ) -> Dict[str, Any]:
        """执行查询"""
        url = f"{self.base_url}{self.api_prefix}/query"

        payload = {
            "query": query_text,
            "use_cache": use_cache,
            "stream": stream
        }

        start = time.time()
        response = requests.post(url, json=payload)
        elapsed = time.time() - start

        response.raise_for_status()
        result = response.json()

        # 添加网络延迟
        result["network_time"] = elapsed

        return result

    def submit_feedback(self, query: str, is_positive: bool) -> Dict[str, Any]:
        """提交反馈"""
        url = f"{self.base_url}{self.api_prefix}/feedback"

        payload = {
            "query": query,
            "is_positive": is_positive
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()

    def switch_model(self, model_type: str) -> Dict[str, Any]:
        """切换模型"""
        url = f"{self.base_url}{self.api_prefix}/model/switch"

        payload = {
            "model_type": model_type
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()


def test_basic_query(client: GRAGClient):
    """测试基础查询"""
    print("\n" + "="*60)
    print("测试1: 基础查询")
    print("="*60)

    queries = [
        "如何安装OpenAI Python库？",
        "API密钥如何配置？",
        "GPT-4和GPT-3.5有什么区别？"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        print("-"*60)

        result = client.query(query, use_cache=False)

        print(f"✓ 成功: {result['success']}")
        print(f"✓ 答案长度: {len(result['answer'])} 字符")
        print(f"✓ 响应时间: {result['response_time']:.3f}秒")
        print(f"✓ 上下文数: {len(result.get('context_ids', []))}")
        print(f"✓ 引用数: {result.get('citation_info', {}).get('citation_count', 0)}")
        print(f"✓ 来自缓存: {result.get('from_cache', False)}")

        print(f"\n答案预览:")
        print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])


def test_with_cache(client: GRAGClient):
    """测试缓存功能"""
    print("\n" + "="*60)
    print("测试2: 缓存功能")
    print("="*60)

    query = "如何安装OpenAI Python库？"

    # 第一次查询 (无缓存)
    print(f"\n第一次查询 (无缓存): {query}")
    result1 = client.query(query, use_cache=True)
    print(f"✓ 响应时间: {result1['response_time']:.3f}秒")
    print(f"✓ 来自缓存: {result1.get('from_cache', False)}")

    # 第二次查询 (有缓存)
    print(f"\n第二次查询 (有缓存): {query}")
    result2 = client.query(query, use_cache=True)
    print(f"✓ 响应时间: {result2['response_time']:.3f}秒")
    print(f"✓ 来自缓存: {result2.get('from_cache', False)}")

    # 性能对比
    if result2.get('from_cache'):
        speedup = result1['response_time'] / result2['response_time']
        print(f"\n✓ 缓存加速: {speedup:.1f}x")


def test_feedback(client: GRAGClient):
    """测试反馈功能"""
    print("\n" + "="*60)
    print("测试3: 反馈功能")
    print("="*60)

    query = "如何安装OpenAI Python库？"

    # 正面反馈
    print(f"\n提交正面反馈: {query}")
    result = client.submit_feedback(query, is_positive=True)
    print(f"✓ {result['message']}")


def test_citations(client: GRAGClient):
    """测试引用功能"""
    print("\n" + "="*60)
    print("测试4: 引用功能")
    print("="*60)

    query = "GPT-4和GPT-3.5有什么区别？"

    result = client.query(query, use_cache=False)

    citations = result.get('citations', [])
    citation_info = result.get('citation_info', {})

    print(f"\n查询: {query}")
    print(f"✓ 有引用: {citation_info.get('has_citations', False)}")
    print(f"✓ 引用数量: {citation_info.get('citation_count', 0)}")

    if citations:
        print(f"\n引用详情:")
        for i, citation in enumerate(citations[:3], 1):
            print(f"\n  引用 {i}:")
            print(f"    ID: {citation.get('chunk_id', 'N/A')}")
            print(f"    来源: {citation.get('source', 'N/A')}")
            print(f"    内容: {citation.get('content', 'N/A')[:100]}...")


def test_performance(client: GRAGClient, num_queries: int = 10):
    """测试性能"""
    print("\n" + "="*60)
    print(f"测试5: 性能测试 ({num_queries}次查询)")
    print("="*60)

    query = "如何安装OpenAI Python库？"

    times = []
    for i in range(num_queries):
        result = client.query(query, use_cache=False)
        times.append(result['response_time'])
        print(f"  查询 {i+1}: {result['response_time']:.3f}秒")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n性能统计:")
    print(f"  平均: {avg_time:.3f}秒")
    print(f"  最快: {min_time:.3f}秒")
    print(f"  最慢: {max_time:.3f}秒")
    print(f"  QPS: {1/avg_time:.1f}")


def main():
    """主函数"""
    print("="*60)
    print("GRAG客户端测试".center(60))
    print("="*60)

    # 初始化客户端
    client = GRAGClient(base_url="http://localhost:8000")

    try:
        # 运行测试
        test_basic_query(client)
        test_with_cache(client)
        test_feedback(client)
        test_citations(client)
        test_performance(client, num_queries=5)

        print("\n" + "="*60)
        print("✓ 所有测试完成".center(60))
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n✗ 错误: 无法连接到服务器")
        print("  请确认后端服务已启动: uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


if __name__ == "__main__":
    main()
