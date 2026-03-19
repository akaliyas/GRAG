"""
测试 GRAG Pipeline 使用 RAGFlow 项目
展示 GitHub API + Context7 API 的完整工作流
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime

# 配置日志并设置 UTF-8 编码
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置标准输出编码为 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def test_ragflow_pipeline():
    """
    测试 RAGFlow 项目的完整管线流程

    RAGFlow: https://github.com/infiniflow/ragflow
    - 基于深度文档理解的 RAG 引擎
    - 技术栈：Python, Elasticsearch, Redis, MySQL, MinIO
    - 丰富的文档和代码示例
    """

    print("=" * 80)
    print("GRAG Pipeline 测试: RAGFlow 项目")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: GitHub API 提取
    print("Step 1: GitHub API 提取（主要数据源）")
    print("-" * 80)

    try:
        from agent.tools.github_ingestor import GitHubIngestor
        from utils.schema import IngestionBatch

        ragflow_url = "https://github.com/infiniflow/ragflow"
        raw_output = "artifacts/01_raw/ragflow_raw.json"

        print(f"📥 拉取仓库: {ragflow_url}")
        print(f"📁 输出文件: {raw_output}")

        ingestor = GitHubIngestor()

        # 提取文档（返回 IngestionBatch）
        batch = ingestor.extract_repo_docs(
            repo_url=ragflow_url,
            file_extensions=['.md', '.mdx'],
            max_files=50,  # 限制文件数量用于测试
            include_paths=['README.md', 'docs/', 'doc/']
        )

        docs = batch.docs
        print(f"\n[+] 提取完成:")
        print(f"    总文档数: {len(docs)}")
        print(f"    Markdown: {sum(1 for d in docs if d.file_type == 'markdown')}")
        print(f"    Notebook: {sum(1 for d in docs if d.file_type == 'notebook')}")

        # 保存 Raw Artifact
        Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
        batch.save_to_file(raw_output)
        print(f"    [+] 已保存: {raw_output}")

        # 显示一些示例文档
        print(f"\n[*] 示例文档:")
        for doc in docs[:3]:
            print(f"    - {doc.path[:60]}")
            print(f"      类型: {doc.file_type}, 大小: {len(doc.content)} bytes")

    except Exception as e:
        logger.error(f"GitHub API 提取失败: {e}")
        print(f"[-] Step 1 失败: {e}")
        return False

    # Step 2: Context7 增强
    print("\n" + "=" * 80)
    print("Step 2: Context7 增强（补充工具/脚手架）")
    print("-" * 80)

    try:
        from utils.context7_client import Context7Client

        client = Context7Client()

        if not client.api_key:
            print("[!] Context7 API Key 未设置，跳过增强")
        else:
            print("[*] 使用 Context7 发现相关技术库...")

            # 搜索 RAGFlow 相关的技术栈
            tech_stack = [
                ("elasticsearch", "全文搜索引擎"),
                ("redis", "缓存和消息队列"),
                ("mysql", "关系型数据库"),
                ("minio", "对象存储"),
                ("langchain", "LLM 应用框架")
            ]

            related_libs = []
            for tech, desc in tech_stack:
                print(f"\n    搜索: {tech} ({desc})")
                libs = client.search_library(tech, "RAG and vector search", limit=2)
                for lib in libs:
                    print(f"    [+] 找到: {lib.name}")
                    print(f"       ID: {lib.library_id}")
                    print(f"       描述: {lib.description[:80]}...")
                    print(f"       Stars: {lib.stars}, Snippets: {lib.total_snippets}")
                    related_libs.append({
                        "tech": tech,
                        "name": lib.name,
                        "library_id": lib.library_id,
                        "description": lib.description
                    })

            print(f"\n[+] Context7 增强完成:")
            print(f"    发现 {len(related_libs)} 个相关技术库")

            # 获取一个具体的技术文档示例
            if related_libs:
                print(f"\n[*] 获取技术文档示例:")
                lib = related_libs[0]
                context = client.get_documentation_context(
                    lib["library_id"],
                    "how to use in RAG applications"
                )
                if context:
                    print(f"    库: {lib['name']}")
                    print(f"    代码片段: {len(context.code_snippets)}")
                    print(f"    信息片段: {len(context.info_snippets)}")

                    if context.code_snippets:
                        snippet = context.code_snippets[0]
                        print(f"\n    示例代码:")
                        print(f"    标题: {snippet.code_title}")
                        print(f"    语言: {snippet.code_language}")
                        if snippet.code_list:
                            code = snippet.code_list[0].get("code", "")[:100]
                            print(f"    代码: {code}...")

            client.close()

    except Exception as e:
        logger.error(f"Context7 增强失败: {e}")
        print(f"[!] Step 2 跳过: {e}")

    # Step 3: 管线效果总结
    print("\n" + "=" * 80)
    print("管线效果总结")
    print("=" * 80)

    print("\n[*] 数据源对比:")
    print("[+] GitHub API (主要数据源)")
    print("    - 拉取了 {} 个文档".format(len(docs)))
    print("    - 包含 README、教程、API 文档等")
    print("    - 源码即真理（Source Code is Truth）")
    print("[+] Context7 API (补充工具/脚手架)")
    print("    - 发现了 {} 个相关技术库".format(len(related_libs) if related_libs else 0))
    print("    - 提供最新 API 文档和代码示例")
    print("    - 补充时效性不足的文档")

    print("\n[+] 管线优势:")
    print("    1. 零爬虫策略: 仅使用结构化 API（GitHub + Context7）")
    print("    2. 数据质量控制: Pydantic Schema 确保数据契约")
    print("    3. 幂等性保证: 基于 MD5 的确定性 ID")
    print("    4. 可追溯性: 完整的元数据和来源记录")
    print("    5. 脚手架作用: Context7 帮助发现相关技术栈")

    print("\n[!] RAGFlow 项目特点:")
    print("    - 丰富的技术文档（README + docs/ 目录）")
    print("    - 复杂的技术栈（Elasticsearch, Redis, MySQL, MinIO）")
    print("    - 与 GRAG 项目相似（都是 RAG 系统）")
    print("    - 适合测试知识图谱构建效果")

    print("\n[>] 下一步:")
    print("    1. 运行清洗脚本: uv run scripts/pipeline_clean.py")
    print("    2. 运行 Context7 增强: uv run scripts/pipeline_context7_enhance.py")
    print("    3. 导入 LightRAG: uv run scripts/pipeline_ingest.py")
    print("    4. 测试问答功能")

    return True


if __name__ == "__main__":
    try:
        success = test_ragflow_pipeline()
        print("\n" + "=" * 80)
        if success:
            print("[+] 测试完成")
        else:
            print("[-] 测试失败")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!] 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.exception("测试过程发生错误")
        print(f"\n[-] 错误: {e}")
        sys.exit(1)
