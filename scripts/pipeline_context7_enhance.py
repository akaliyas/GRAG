"""
Pipeline Step 2.5: Context7 增强（可选）
使用 Context7 API 增强文档元数据，添加相关库信息

使用方式：
    uv run scripts/pipeline_context7_enhance.py --input artifacts/02_clean/fastapi_clean.json --output artifacts/02_clean/fastapi_enhanced.json

注意：这是可选步骤，Context7 不可用时会跳过增强
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_uv_environment():
    """检查是否在 uv 环境中运行"""
    if os.getenv('UV_PROJECT_ENVIRONMENT'):
        return True

    project_root = Path(__file__).parent.parent
    venv_path = project_root / '.venv'
    if venv_path.exists() and sys.prefix and str(venv_path) in sys.prefix:
        return True

    if sys.executable:
        executable_path = Path(sys.executable)
        if '.venv' in str(executable_path):
            return True

    if venv_path.exists():
        return True

    return False


def ensure_uv_environment():
    """确保在 uv 环境中运行，否则提示用户"""
    if not check_uv_environment():
        logger = logging.getLogger(__name__)
        logger.warning("⚠️  建议使用 'uv run' 来执行此脚本")
        logger.warning("   示例: uv run scripts/pipeline_context7_enhance.py --input <input> --output <output>")
        logger.warning("   当前将继续执行，但可能缺少依赖...")
        logger.warning("")


from utils.schema import CleanBatch
from utils.context7_client import get_context7_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def enhance_batch(
    batch: CleanBatch,
    repo_name: str
) -> CleanBatch:
    """
    使用 Context7 增强批次元数据

    Args:
        batch: 清洗后的批次
        repo_name: 仓库名称（如 "fastapi/fastapi"）

    Returns:
        增强后的批次
    """
    context7_client = get_context7_client()

    if not context7_client:
        logger.warning("Context7 不可用，跳过增强")
        logger.info("💡 提示：在 .env 文件中设置 CONTEXT7_API_KEY 以启用此功能")
        return batch

    try:
        logger.info(f"开始使用 Context7 增强元数据...")
        logger.info(f"源仓库: {repo_name}")

        # 增强批次级别的元数据
        enhanced_metadata = context7_client.enhance_metadata(
            repo_name=repo_name,
            existing_metadata={}
        )

        # 为每个文档添加增强元数据
        enhanced_docs = []
        for doc in batch.docs:
            # 创建文档的增强元数据副本
            doc_enhanced_metadata = doc.metadata.copy()

            # 添加 Context7 发现的相关库信息
            if "context7_related_libraries" in enhanced_metadata:
                doc_enhanced_metadata["context7_related_libraries"] = enhanced_metadata["context7_related_libraries"]
                doc_enhanced_metadata["context7_enhanced_at"] = enhanced_metadata.get("context7_enhanced_at")

            # 创建增强后的文档
            from utils.schema import CleanDoc
            enhanced_doc = CleanDoc(
                doc_id=doc.doc_id,
                content=doc.content,
                source_url=doc.source_url,
                file_path=doc.file_path,
                file_type=doc.file_type,
                metadata=doc_enhanced_metadata,
                cleaning_log=doc.cleaning_log
            )
            enhanced_docs.append(enhanced_doc)

        # 创建增强后的批次
        enhanced_batch = CleanBatch(
            source_url=batch.source_url,
            docs=enhanced_docs,
            cleaned_at=batch.cleaned_at,
            raw_batch_path=batch.raw_batch_path
        )

        # 添加增强信息
        if enhanced_metadata.get("context7_related_libraries"):
            logger.info(f"✅ 发现 {len(enhanced_metadata['context7_related_libraries'])} 个相关库:")
            for lib in enhanced_metadata["context7_related_libraries"]:
                logger.info(f"   - {lib['name']} ({lib.get('library_id', 'N/A')})")

        context7_client.close()
        return enhanced_batch

    except Exception as e:
        logger.error(f"Context7 增强失败: {e}")
        logger.info("继续返回原始批次（不影响主流程）")
        return batch


def main():
    # 检查 uv 环境
    ensure_uv_environment()

    parser = argparse.ArgumentParser(description='Context7 元数据增强（Step 2.5: 可选）')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件路径（Clean Artifact，如 artifacts/02_clean/fastapi_clean.json）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（增强后的 Clean Artifact，如 artifacts/02_clean/fastapi_enhanced.json）'
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        default=None,
        help='仓库名称（如 "fastapi/fastapi"，如不指定则从 source_url 提取）'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Pipeline Step 2.5: Context7 增强（可选）")
    logger.info("=" * 60)

    # 1. 加载 Clean Artifact
    logger.info(f"\n1. 加载 Clean Artifact: {args.input}")
    try:
        batch = CleanBatch.load_from_file(args.input)
        logger.info(f"   ✅ 成功加载 {len(batch.docs)} 个文档")
    except Exception as e:
        logger.error(f"   ❌ 加载失败: {e}")
        sys.exit(1)

    # 2. 提取仓库名称
    repo_name = args.repo_name
    if not repo_name:
        # 从 source_url 提取
        url_parts = batch.source_url.strip('/').split('/')
        if len(url_parts) >= 2:
            repo_name = f"{url_parts[-2]}/{url_parts[-1]}"
        else:
            repo_name = url_parts[-1] if url_parts else "unknown"
    logger.info(f"\n2. 仓库名称: {repo_name}")

    # 3. Context7 增强
    logger.info(f"\n3. Context7 增强")
    logger.info("   " + "-" * 50)
    enhanced_batch = enhance_batch(batch, repo_name)

    # 4. 保存增强后的批次
    logger.info(f"\n4. 保存增强后的批次: {args.output}")
    try:
        enhanced_batch.save_to_file(args.output)
        logger.info(f"   ✅ 成功保存")
    except Exception as e:
        logger.error(f"   ❌ 保存失败: {e}")
        sys.exit(1)

    # 5. 总结
    logger.info("\n" + "=" * 60)
    logger.info("✅ Context7 增强完成")
    logger.info(f"   输入: {args.input}")
    logger.info(f"   输出: {args.output}")
    logger.info(f"   文档数: {len(enhanced_batch.docs)}")
    logger.info("=" * 60)

    logger.info("\n💡 下一步:")
    logger.info(f"   uv run scripts/pipeline_ingest.py --input {args.output}")


if __name__ == "__main__":
    main()
