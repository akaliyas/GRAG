"""
Pipeline Step 2: Clean（清洗数据）
读取 Raw Artifact，应用清洗逻辑，保存为 Clean Artifact（artifacts/02_clean/）

清洗逻辑：
- HTML 标签清理（已在 GitHubIngestor 中完成）
- Notebook 清理（已在 GitHubIngestor 中完成）
- 链接修复（已在 GitHubIngestor 中完成）
- 可选的额外清洗步骤

使用方式：
    uv run scripts/pipeline_clean.py --input artifacts/01_raw/openai_v1_raw.json --output artifacts/02_clean/openai_v1_clean.json
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_uv_environment():
    """检查是否在 uv 环境中运行"""
    # 检查环境变量（某些情况下 uv 会设置此变量）
    if os.getenv('UV_PROJECT_ENVIRONMENT'):
        return True
    
    # 检查是否在项目虚拟环境中（.venv 目录存在）
    project_root = Path(__file__).parent.parent
    venv_path = project_root / '.venv'
    if venv_path.exists() and sys.prefix and str(venv_path) in sys.prefix:
        return True
    
    # 检查是否通过 uv run 启动（检查 sys.executable 路径）
    if sys.executable:
        executable_path = Path(sys.executable)
        # 如果在 .venv 目录中，说明可能是通过 uv 管理的
        if '.venv' in str(executable_path):
            return True
    
    # 如果项目根目录有 .venv，假设使用 uv 管理
    if venv_path.exists():
        return True
    
    return False


def ensure_uv_environment():
    """确保在 uv 环境中运行，否则提示用户"""
    if not check_uv_environment():
        logger = logging.getLogger(__name__)
        logger.warning("⚠️  建议使用 'uv run' 来执行此脚本")
        logger.warning("   示例: uv run scripts/pipeline_clean.py --input <input-path>")
        logger.warning("   当前将继续执行，但可能缺少依赖...")
        logger.warning("")

from utils.schema import IngestionBatch, CleanBatch, CleanDoc, RawDoc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_document(raw_doc: RawDoc, ingestor) -> CleanDoc:
    """
    清洗单个文档
    
    核心逻辑：
    1. Frontmatter 剥离：将 YAML Frontmatter 从内容中提取出来，存入 metadata
    2. HTML 标签清理：移除所有 HTML 标签
    3. Notebook 清理：移除 Notebook 输出，保留 Markdown 和 Code 输入
    4. 链接修复：修复相对链接为 GitHub Raw URL
    
    Args:
        raw_doc: RawDoc 对象（包含原始内容）
        ingestor: GitHubIngestor 实例（复用，避免重复初始化）
        
    Returns:
        CleanDoc 对象（清洗后的内容）
    """
    content = raw_doc.content
    original_length = len(content)
    cleaning_log = []
    
    # 步骤 1: Frontmatter 剥离（仅对 Markdown 文件）
    frontmatter = {}
    if raw_doc.file_type == 'markdown':
        frontmatter, body_content = ingestor._extract_frontmatter(content)
        if frontmatter:
            content = body_content
            cleaning_log.append(f"剥离了 Frontmatter: {list(frontmatter.keys())}")
    
    # 步骤 2: 根据文件类型应用清洗逻辑
    if raw_doc.file_type == 'notebook':
        # Notebook 清洗：移除输出，保留 Markdown 和 Code 输入
        content = ingestor._clean_notebook(content, raw_doc.source_url or '')
        cleaning_log.append("清洗了 Notebook（移除输出，保留 Markdown 和 Code 输入）")
    elif raw_doc.file_type == 'markdown':
        # Markdown 清洗：移除 HTML 标签，修复链接
        content = ingestor._clean_html_tags(content)
        content = ingestor._fix_relative_links(content, raw_doc.source_url or '')
        cleaning_log.append("清理了 HTML 标签，修复了相对链接")
    else:
        # 其他文件类型：仅清理 HTML 标签
        content = ingestor._clean_html_tags(content)
        cleaning_log.append("清理了 HTML 标签")
    
    # 步骤 3: 清理多余的空白行
    import re
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 记录清洗统计
    if len(content) != original_length:
        cleaning_log.append(f"清理了空白行（{original_length} -> {len(content)} 字符）")
    
    # 合并 metadata：原始 metadata + Frontmatter
    enhanced_metadata = raw_doc.metadata.copy()
    if frontmatter:
        enhanced_metadata['frontmatter'] = frontmatter
    
    return CleanDoc(
        doc_id=raw_doc.doc_id,  # 继承 doc_id，确保幂等性
        content=content,  # 清洗后的内容（Frontmatter 已剥离）
        source_url=raw_doc.source_url or '',  # 继承 source_url
        file_path=raw_doc.path,  # 继承 file_path
        file_type=raw_doc.file_type,  # 继承 file_type
        metadata=enhanced_metadata,  # 增强后的 metadata（包含 Frontmatter）
        cleaning_log=cleaning_log  # 清洗日志
    )


def main():
    # 检查 uv 环境
    ensure_uv_environment()
    
    parser = argparse.ArgumentParser(description='清洗文档数据（Step 2: Clean）')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件路径（Raw Artifact，如 artifacts/01_raw/openai_v1_raw.json）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（Clean Artifact，如 artifacts/02_clean/openai_v1_clean.json）'
    )
    
    args = parser.parse_args()
    
    # 读取 Raw Artifact
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    logger.info(f"读取 Raw Artifact: {input_path}")
    try:
        raw_batch = IngestionBatch.load_from_file(str(input_path))
        logger.info(f"✅ 成功读取 {len(raw_batch.docs)} 个文档")
    except Exception as e:
        logger.error(f"❌ 读取失败: {e}")
        sys.exit(1)
    
    # 创建 GitHubIngestor 实例（复用，避免重复初始化）
    from agent.tools.github_ingestor import GitHubIngestor
    ingestor = GitHubIngestor()
    
    # 清洗文档
    logger.info("开始清洗文档...")
    logger.info("  清洗步骤：1. Frontmatter 剥离 2. HTML 标签清理 3. Notebook 清理 4. 链接修复")
    clean_docs = []
    
    for i, raw_doc in enumerate(raw_batch.docs, 1):
        try:
            clean_doc = clean_document(raw_doc, ingestor)
            clean_docs.append(clean_doc)
            
            if (i % 10 == 0) or (i == len(raw_batch.docs)):
                logger.info(f"  已处理 {i}/{len(raw_batch.docs)} 个文档")
                if clean_doc.cleaning_log:
                    logger.debug(f"    清洗日志: {', '.join(clean_doc.cleaning_log)}")
        except Exception as e:
            logger.warning(f"清洗文档失败 ({raw_doc.path}): {e}")
            import traceback
            traceback.print_exc()
            # 即使清洗失败，也保留原始内容（但标记为失败）
            clean_docs.append(CleanDoc(
                doc_id=raw_doc.doc_id,
                content=raw_doc.content,  # 保留原始内容
                source_url=raw_doc.source_url or '',
                file_path=raw_doc.path,
                file_type=raw_doc.file_type,
                metadata=raw_doc.metadata,
                cleaning_log=[f"清洗失败: {str(e)}"]
            ))
    
    # 创建 CleanBatch
    clean_batch = CleanBatch(
        source_url=raw_batch.repo_url,  # 使用 repo_url 字段
        docs=clean_docs,
        raw_batch_path=str(input_path)
    )
    
    # 保存到文件
    output_path = Path(args.output)
    clean_batch.save_to_file(str(output_path))
    
    logger.info(f"✅ 已保存到: {output_path}")
    logger.info(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 打印统计信息
    total_chars = sum(len(doc.content) for doc in clean_docs)
    logger.info("\n📊 统计信息:")
    logger.info(f"  总文档数: {len(clean_docs)}")
    logger.info(f"  总字符数: {total_chars:,}")
    
    logger.info("\n✅ Step 2 (Clean) 完成！")
    logger.info(f"下一步: 运行 pipeline_ingest.py 导入到 LightRAG")
    logger.info(f"  uv run scripts/pipeline_ingest.py --input {output_path}")


if __name__ == '__main__':
    main()


