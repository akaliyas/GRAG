"""
Context7 客户端工具
作为 GRAG 项目的补充工具（脚手架），用于：
1. 发现相关技术库
2. 获取最新的 API 文档上下文
3. 增强元数据

生态位定位：补充作用，非主要数据源
主要数据源仍是 GitHub API（零爬虫策略）
"""
import logging
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@dataclass
class Context7Library:
    """Context7 库信息"""
    library_id: str  # 格式: "/vendor/project"
    name: str
    description: str
    total_snippets: int = 0
    total_tokens: int = 0
    stars: int = 0
    trust_score: int = 0
    benchmark_score: float = 0.0
    versions: List[str] = None
    branch: str = None
    state: str = None
    last_update_date: str = None

    def __post_init__(self):
        if self.versions is None:
            self.versions = []


@dataclass
class Context7CodeSnippet:
    """代码片段"""
    code_title: str
    code_description: str
    code_language: str
    code_tokens: int
    code_id: str
    page_title: str
    code_list: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.code_list is None:
            self.code_list = []


@dataclass
class Context7InfoSnippet:
    """信息片段"""
    page_id: str
    breadcrumb: str
    content: str
    content_tokens: int


@dataclass
class Context7DocContext:
    """Context7 文档上下文"""
    library_id: str
    query: str
    code_snippets: List[Context7CodeSnippet]
    info_snippets: List[Context7InfoSnippet]
    retrieved_at: datetime
    response_format: str = "json"  # json or txt

    @property
    def all_snippets(self) -> List[Dict[str, Any]]:
        """返回所有片段的统一格式"""
        result = []
        for snippet in self.code_snippets:
            result.append({
                "type": "code",
                "title": snippet.code_title,
                "description": snippet.code_description,
                "language": snippet.code_language,
                "content": snippet.code_list[0]["code"] if snippet.code_list else "",
                "page_title": snippet.page_title
            })
        for snippet in self.info_snippets:
            result.append({
                "type": "info",
                "title": snippet.breadcrumb,
                "content": snippet.content,
                "page_id": snippet.page_id
            })
        return result


class Context7Client:
    """
    Context7 API 客户端

    功能：
    - 搜索相关技术库
    - 获取最新的文档上下文
    - 补充 GitHub API 数据的不足

    使用场景：
    1. Pipeline 开始前：发现相关技术栈
    2. 元数据增强：为文档添加相关库链接
    3. 上下文补充：获取最新 API 变更

    注意：这是辅助工具，不替代 GitHub API
    """

    BASE_URL = "https://context7.com/api"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        """
        初始化 Context7 客户端

        Args:
            api_key: Context7 API Key（从环境变量 CONTEXT7_API_KEY 读取）
            timeout: HTTP 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("CONTEXT7_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            logger.warning("CONTEXT7_API_KEY 未设置，Context7 功能将受限")

        self.client = httpx.Client(
            base_url=self.BASE_URL,
            headers=self._get_headers(),
            timeout=timeout
        )

        logger.info("Context7 客户端已初始化")

    def _get_headers(self) -> Dict[str, str]:
        """构建请求头（完全匹配官方文档格式）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        return headers

    def is_available(self) -> bool:
        """检查 Context7 服务是否可用（通过简单搜索测试）"""
        if not self.api_key:
            return False

        try:
            # 尝试一个简单的搜索来测试连接
            response = self.client.get(
                "/v2/libs/search",
                params={"libraryName": "test", "query": "test"}
            )
            # 只要能连通（不是网络错误），就认为可用
            return response.status_code in [200, 401, 404, 429]
        except Exception as e:
            logger.debug(f"Context7 服务检查失败: {e}")
            return False

    def search_library(
        self,
        library_name: str,
        query: str = "",
        limit: int = 5
    ) -> List[Context7Library]:
        """
        搜索相关技术库

        这是脚手架功能，用于：
        - 发现 GitHub 仓库依赖的相关库
        - 为元数据增强提供候选库

        Args:
            library_name: 库名称（如 "fastapi", "react", "nextjs"）
            query: 查询问题（用于相关性排序，如 "I need to build an API"）
            limit: 返回结果数量

        Returns:
            Context7Library 列表
        """
        if not self.api_key:
            logger.warning("Context7 API Key 未设置，跳过库搜索")
            return []

        try:
            # API: GET /api/v2/libs/search
            # 官方文档: https://context7.com/docs/api-guide
            response = self.client.get(
                "/v2/libs/search",
                params={
                    "libraryName": library_name,
                    "query": query
                }
            )

            # 处理不同的响应状态
            if response.status_code == 401:
                logger.error("Context7 API Key 无效（格式应为 ctx7sk-...）")
                logger.info("💡 请检查 https://context7.com/dashboard 中的 API Key")
                return []
            elif response.status_code == 403:
                logger.error("Context7 API 访问被拒绝（403 Forbidden）")
                logger.info("💡 可能原因：API Key 未激活、需要验证账户、或 IP 限制")
                logger.info("💡 新 API Key 可能需要等待 5-10 分钟才能激活")
                logger.info("💡 运行诊断: python scripts/test_context7_api.py")
                return []
            elif response.status_code == 429:
                logger.warning("Context7 API 速率限制")
                return []
            elif response.status_code == 404:
                logger.warning(f"未找到库: {library_name}")
                return []

            response.raise_for_status()
            data = response.json()

            # API 返回格式: {"results": [...]}
            results = data.get("results", data) if isinstance(data, dict) else data

            # 处理数组格式的库列表
            libraries = []
            for item in (results[:limit] if isinstance(results, list) else []):
                lib = Context7Library(
                    library_id=item.get("id", ""),
                    name=item.get("title", ""),
                    description=item.get("description", ""),
                    total_snippets=item.get("totalSnippets", 0),
                    total_tokens=item.get("totalTokens", 0),
                    stars=item.get("stars", 0),
                    trust_score=item.get("trustScore", 0),
                    benchmark_score=item.get("benchmarkScore", 0.0),
                    versions=item.get("versions", []),
                    branch=item.get("branch"),
                    state=item.get("state"),
                    last_update_date=item.get("lastUpdateDate")
                )
                libraries.append(lib)

            logger.info(f"✅ 通过 Context7 发现 {len(libraries)} 个相关库")
            return libraries

        except httpx.HTTPStatusError as e:
            logger.error(f"Context7 API 错误: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"搜索库失败: {e}")
            return []

    def get_documentation_context(
        self,
        library_id: str,
        query: str,
        response_format: str = "json"
    ) -> Optional[Context7DocContext]:
        """
        获取文档上下文

        这是补充功能，用于：
        - 获取最新的 API 文档
        - 补充 GitHub 文档的时效性不足

        Args:
            library_id: Context7 库 ID（如 "/vercel/next.js"）
            query: 查询内容
            response_format: 响应格式，"json" 或 "txt"

        Returns:
            Context7DocContext 对象
        """
        if not self.api_key:
            logger.warning("Context7 API Key 未设置，跳过文档获取")
            return None

        try:
            # API: GET /api/v2/context
            # 官方文档: https://context7.com/docs/api-guide
            response = self.client.get(
                "/v2/context",
                params={
                    "libraryId": library_id,
                    "query": query,
                    "type": response_format
                }
            )

            # 处理不同的响应状态
            if response.status_code == 401:
                logger.error("Context7 API Key 无效")
                return None
            elif response.status_code == 404:
                logger.warning(f"库不存在: {library_id}")
                return None
            elif response.status_code == 422:
                logger.warning(f"库太大或无代码: {library_id}")
                return None
            elif response.status_code == 429:
                logger.warning("Context7 API 速率限制")
                return None

            response.raise_for_status()

            # JSON 格式返回: {codeSnippets: [...], infoSnippets: [...]}
            # txt 格式返回纯文本
            if response_format == "json":
                data = response.json()

                # 解析代码片段
                code_snippets = []
                for item in data.get("codeSnippets", []):
                    snippet = Context7CodeSnippet(
                        code_title=item.get("codeTitle", ""),
                        code_description=item.get("codeDescription", ""),
                        code_language=item.get("codeLanguage", ""),
                        code_tokens=item.get("codeTokens", 0),
                        code_id=item.get("codeId", ""),
                        page_title=item.get("pageTitle", ""),
                        code_list=item.get("codeList", [])
                    )
                    code_snippets.append(snippet)

                # 解析信息片段
                info_snippets = []
                for item in data.get("infoSnippets", []):
                    snippet = Context7InfoSnippet(
                        page_id=item.get("pageId", ""),
                        breadcrumb=item.get("breadcrumb", ""),
                        content=item.get("content", ""),
                        content_tokens=item.get("contentTokens", 0)
                    )
                    info_snippets.append(snippet)

                doc_context = Context7DocContext(
                    library_id=library_id,
                    query=query,
                    code_snippets=code_snippets,
                    info_snippets=info_snippets,
                    retrieved_at=datetime.now(),
                    response_format=response_format
                )
            else:
                # txt 格式
                doc_context = Context7DocContext(
                    library_id=library_id,
                    query=query,
                    code_snippets=[],
                    info_snippets=[],
                    retrieved_at=datetime.now(),
                    response_format=response_format
                )

            logger.info(f"✅ 从 Context7 获取文档上下文: {library_id}")
            logger.info(f"   代码片段: {len(doc_context.code_snippets)}, 信息片段: {len(doc_context.info_snippets)}")
            return doc_context

        except httpx.HTTPStatusError as e:
            logger.error(f"Context7 API 错误: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"获取文档上下文失败: {e}")
            return None

    def enhance_metadata(
        self,
        repo_name: str,
        existing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        增强元数据（脚手架功能）

        基于仓库名称，搜索相关库并添加到元数据中。

        Args:
            repo_name: 仓库名称（如 "fastapi/fastapi"）
            existing_metadata: 现有元数据

        Returns:
            增强后的元数据
        """
        enhanced = existing_metadata.copy()

        # 提取库名
        library_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name

        # 搜索相关库
        related_libs = self.search_library(library_name, limit=3)

        if related_libs:
            enhanced["context7_related_libraries"] = [
                {
                    "library_id": lib.library_id,
                    "name": lib.name,
                    "description": lib.description,
                    "relevance_score": lib.relevance_score
                }
                for lib in related_libs
            ]
            enhanced["context7_enhanced_at"] = datetime.now().isoformat()

        return enhanced

    def close(self):
        """关闭客户端连接"""
        if hasattr(self, 'client'):
            self.client.close()

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.close()


# 便捷函数
def get_context7_client() -> Optional[Context7Client]:
    """
    获取 Context7 客户端实例

    如果 API key 未配置，返回 None（静默失败，不阻塞主流程）

    Returns:
        Context7Client 实例或 None
    """
    api_key = os.getenv("CONTEXT7_API_KEY")
    if not api_key:
        logger.debug("CONTEXT7_API_KEY 未配置，Context7 功能禁用")
        return None

    return Context7Client(api_key=api_key)


if __name__ == "__main__":
    # 测试代码
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Context7 客户端测试")
    print("=" * 50)

    # 检查 API key
    if not os.getenv("CONTEXT7_API_KEY"):
        print("⚠️  CONTEXT7_API_KEY 未设置")
        print("   请在 .env 文件中设置：CONTEXT7_API_KEY=your_key_here")
        sys.exit(1)

    # 测试客户端
    with Context7Client() as client:
        # 1. 检查可用性
        print(f"\n1. 服务可用性: {client.is_available()}")

        # 2. 搜索库
        print("\n2. 搜索 'fastapi' 相关库:")
        libraries = client.search_library("fastapi")
        for lib in libraries:
            print(f"   - {lib.name} ({lib.library_id})")
            print(f"     {lib.description[:80]}...")

        # 3. 增强元数据
        print("\n3. 增强元数据:")
        enhanced = client.enhance_metadata(
            "fastapi/fastapi",
            {"version": "0.100.0"}
        )
        print(f"   增强后元数据: {enhanced}")
