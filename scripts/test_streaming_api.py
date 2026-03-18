"""
流式API测试脚本

测试改进后的流式实现：
1. 真正的token级流式（非字符模拟）
2. 引用系统与流式的兼容
3. 性能提升（无人工延迟）

使用方式：
    uv run scripts/test_streaming_api.py

要求：
    - 后端服务已启动: uvicorn api.main:app --reload
    - 或使用: uv run uvicorn api.main:app --reload
"""
import requests
import json
import time
from typing import Dict, Any, Generator


class StreamingAPIClient:
    """流式API客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        self.username = "admin"
        self.password = ""  # 根据实际配置修改

    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def query_stream(
        self,
        query_text: str,
        use_cache: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式查询

        Args:
            query_text: 查询文本
            use_cache: 是否使用缓存

        Yields:
            dict: SSE格式的响应块
        """
        url = f"{self.base_url}{self.api_prefix}/query/stream"

        payload = {
            "query": query_text,
            "use_cache": use_cache,
            "stream": True
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._get_auth_headers(),
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not isinstance(line, bytes):
                    continue

                try:
                    line = line.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    continue

                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if isinstance(data, dict):
                            yield data
                            if data.get("done"):
                                return
                    except (json.JSONDecodeError, ValueError):
                        continue

        except requests.exceptions.Timeout:
            yield {"error": "请求超时"}
        except requests.exceptions.ConnectionError:
            yield {"error": "无法连接后端服务"}
        except Exception as e:
            yield {"error": str(e)}

    def query(self, query_text: str, use_cache: bool = False) -> Dict[str, Any]:
        """非流式查询（对比用）"""
        url = f"{self.base_url}{self.api_prefix}/query"

        payload = {
            "query": query_text,
            "use_cache": use_cache,
            "stream": False
        }

        start = time.time()
        response = requests.post(
            url,
            json=payload,
            headers=self._get_auth_headers(),
            timeout=120
        )
        elapsed = time.time() - start

        response.raise_for_status()
        result = response.json()
        result["network_time"] = elapsed

        return result


def test_streaming_basic(client: StreamingAPIClient):
    """测试基础流式查询"""
    print("\n" + "=" * 60)
    print("测试1: 基础流式查询")
    print("=" * 60)

    query = "什么是 GRAG 系统？"
    print(f"\n查询: {query}")
    print("流式输出:\n")

    start_time = time.time()
    full_response = ""
    chunk_count = 0
    token_count = 0

    for chunk in client.query_stream(query, use_cache=False):
        chunk_count += 1

        if "error" in chunk:
            print(f"\n错误: {chunk['error']}")
            return False

        if "content" in chunk and chunk["content"]:
            full_response += chunk["content"]
            token_count += len(chunk["content"])
            print(chunk["content"], end='', flush=True)

        if "done" in chunk and chunk["done"]:
            elapsed = time.time() - start_time
            print(f"\n\n流式响应完成！")
            print(f"  总块数: {chunk_count}")
            print(f"  总字符数: {len(full_response)}")
            print(f"  响应时间: {elapsed:.2f} 秒")
            print(f"  模型类型: {chunk.get('model_type', 'unknown')}")
            print(f"  上下文数: {len(chunk.get('context_ids', []))}")
            print(f"  引用数: {len(chunk.get('citations', []))}")

            # 验证引用信息
            if chunk.get('citations'):
                print(f"\n  引用详情:")
                for i, citation in enumerate(chunk['citations'][:3], 1):
                    source = citation.get('source', {})
                    print(f"    [{i}] {source.get('file_path', '未知来源')}")

            citation_info = chunk.get('citation_info', {})
            if citation_info.get('has_citations'):
                print(f"\n  引用统计:")
                print(f"    引用数量: {citation_info.get('citation_count', 0)}")
                if citation_info.get('was_fixed'):
                    print(f"    引用已修复: 是")

            return True

    return False


def test_streaming_performance(client: StreamingAPIClient):
    """测试流式性能"""
    print("\n" + "=" * 60)
    print("测试2: 流式性能测试")
    print("=" * 60)

    query = "如何使用 GRAG 系统？"
    print(f"\n查询: {query}")

    # 流式查询
    print("\n流式查询:")
    start_time = time.time()
    full_response = ""
    first_chunk_time = None

    for chunk in client.query_stream(query, use_cache=False):
        if first_chunk_time is None and "content" in chunk:
            first_chunk_time = time.time() - start_time

        if "content" in chunk and chunk["content"]:
            full_response += chunk["content"]

        if chunk.get("done"):
            stream_time = time.time() - start_time
            break

    print(f"  首块时间: {first_chunk_time:.3f} 秒")
    print(f"  总时间: {stream_time:.3f} 秒")
    print(f"  字符数: {len(full_response)}")

    # 非流式查询对比
    print("\n非流式查询（对比）:")
    start_time = time.time()
    result = client.query(query, use_cache=False)
    normal_time = time.time() - start_time

    print(f"  总时间: {normal_time:.3f} 秒")
    print(f"  字符数: {len(result.get('answer', ''))}")

    # 性能对比
    speedup = normal_time / stream_time if stream_time > 0 else 0
    print(f"\n性能对比:")
    print(f"  流式加速: {speedup:.2f}x")
    print(f"  首块等待: {first_chunk_time:.3f} 秒")


def test_citations_in_stream(client: StreamingAPIClient):
    """测试流式中的引用功能"""
    print("\n" + "=" * 60)
    print("测试3: 流式引用功能")
    print("=" * 60)

    query = "GRAG 系统的架构是什么？"
    print(f"\n查询: {query}")

    for chunk in client.query_stream(query, use_cache=False):
        if "done" in chunk and chunk["done"]:
            citations = chunk.get('citations', [])
            citation_info = chunk.get('citation_info', {})

            print(f"\n引用结果:")
            print(f"  包含引用: {citation_info.get('has_citations', False)}")
            print(f"  引用数量: {citation_info.get('citation_count', 0)}")
            print(f"  引用已修复: {citation_info.get('was_fixed', False)}")

            if citations:
                print(f"\n引用列表:")
                for i, citation in enumerate(citations, 1):
                    source = citation.get('source', {})
                    print(f"  [{i}] {source.get('file_path', '未知来源')}")
                    print(f"      内容: {citation.get('content', '')[:80]}...")

            return True

        if "error" in chunk:
            print(f"\n错误: {chunk['error']}")
            return False

    return False


def test_cache_with_streaming(client: StreamingAPIClient):
    """测试缓存与流式的结合"""
    print("\n" + "=" * 60)
    print("测试4: 缓存+流式")
    print("=" * 60)

    query = "什么是 LightRAG？"

    # 第一次查询（无缓存）
    print(f"\n第一次查询（无缓存）: {query}")
    start_time = time.time()
    first_response = ""

    for chunk in client.query_stream(query, use_cache=True):
        if "content" in chunk and chunk["content"]:
            first_response += chunk["content"]
            print(chunk["content"], end='', flush=True)
        if chunk.get("done"):
            first_time = time.time() - start_time
            from_cache = chunk.get("from_cache", False)
            break

    print(f"\n  时间: {first_time:.3f} 秒")
    print(f"  来自缓存: {from_cache}")

    # 第二次查询（有缓存）
    print(f"\n第二次查询（有缓存）: {query}")
    start_time = time.time()
    second_response = ""

    for chunk in client.query_stream(query, use_cache=True):
        if "content" in chunk and chunk["content"]:
            second_response += chunk["content"]
            print(chunk["content"], end='', flush=True)
        if chunk.get("done"):
            second_time = time.time() - start_time
            from_cache = chunk.get("from_cache", False)
            break

    print(f"\n  时间: {second_time:.3f} 秒")
    print(f"  来自缓存: {from_cache}")

    # 缓存效果
    if from_cache and second_time > 0:
        speedup = first_time / second_time
        print(f"\n缓存加速: {speedup:.1f}x")


def main():
    """主测试函数"""
    print("=" * 60)
    print("GRAG 流式API测试套件（改进版）".center(60))
    print("=" * 60)
    print("\n改进特性:")
    print("  ✓ 真正的token级流式（非字符模拟）")
    print("  ✓ 引用系统与流式兼容")
    print("  ✓ 移除人工延迟，性能提升")

    # 初始化客户端
    client = StreamingAPIClient()

    # 测试连接
    print("\n检查后端连接...")
    try:
        result = client.query("测试连接", use_cache=False)
        print("✓ 后端连接正常")
    except requests.exceptions.ConnectionError:
        print("\n✗ 错误: 无法连接到后端服务")
        print("  请先启动后端: uv run uvicorn api.main:app --reload")
        return
    except Exception as e:
        print(f"\n✗ 连接失败: {e}")
        return

    # 运行测试
    try:
        test_streaming_basic(client)
        test_streaming_performance(client)
        test_citations_in_stream(client)
        test_cache_with_streaming(client)

        print("\n" + "=" * 60)
        print("✓ 所有测试完成".center(60))
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
