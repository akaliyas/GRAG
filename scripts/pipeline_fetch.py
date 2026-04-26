"""
Pipeline Step 1: Fetch（提取原始数据）
从 GitHub 仓库提取文档，保存为 Raw Artifact（artifacts/01_raw/）

使用方式：
    uv run scripts/pipeline_fetch.py --repo https://github.com/openai/openai-python --output artifacts/01_raw/openai_v1_raw.json
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件）
load_dotenv()

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
        logger.warning("   示例: uv run scripts/pipeline_fetch.py --repo <repo-url>")
        logger.warning("   当前将继续执行，但可能缺少依赖...")
        logger.warning("")

from agent.tools.github_ingestor import GitHubIngestor
from utils.schema import IngestionBatch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # 检查 uv 环境
    ensure_uv_environment()
    
    parser = argparse.ArgumentParser(description='从 GitHub 仓库提取文档（Step 1: Fetch）')
    parser.add_argument(
        '--repo',
        type=str,
        required=True,
        help='GitHub 仓库 URL（如 https://github.com/openai/openai-python）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（Raw Artifact，如 artifacts/01_raw/openai_v1_raw.json）'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.md', '.mdx', '.ipynb'],
        help='文件扩展名列表（默认: .md .mdx .ipynb）'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='最大文件数（可选，默认无限制）'
    )
    parser.add_argument(
        '--include-paths',
        type=str,
        nargs='+',
        default=None,
        help='包含的路径前缀列表（可选）'
    )
    parser.add_argument(
        '--exclude-paths',
        type=str,
        nargs='+',
        default=None,
        help='排除的路径前缀列表（可选）'
    )
    
    args = parser.parse_args()
    
    # 初始化 GitHubIngestor
    logger.info("初始化 GitHubIngestor...")
    try:
        ingestor = GitHubIngestor()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)
    
    # 提取文档
    logger.info(f"开始提取文档: {args.repo}")
    logger.info(f"文件扩展名: {args.extensions}")
    if args.max_files:
        logger.info(f"最大文件数: {args.max_files}")
    if args.include_paths:
        logger.info(f"包含路径: {args.include_paths}")
    if args.exclude_paths:
        logger.info(f"排除路径: {args.exclude_paths}")
    
    try:
        batch = ingestor.extract_repo_docs(
            repo_url=args.repo,
            file_extensions=args.extensions,
            max_files=args.max_files,
            include_paths=args.include_paths,
            exclude_paths=args.exclude_paths
        )
        
        logger.info(f"✅ 成功提取 {len(batch.docs)} 个文档")
        
        # 保存到文件
        output_path = Path(args.output)
        batch.save_to_file(str(output_path))
        
        logger.info(f"✅ 已保存到: {output_path}")
        logger.info(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")
        
        # 打印统计信息
        type_counts = {}
        total_chars = 0
        for doc in batch.docs:
            doc_type = doc.metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            total_chars += len(doc.content)
        
        logger.info("\n📊 统计信息:")
        logger.info(f"  总文档数: {len(batch.docs)}")
        logger.info(f"  总字符数: {total_chars:,}")
        logger.info(f"  文件类型分布:")
        for doc_type, count in type_counts.items():
            logger.info(f"    - {doc_type}: {count} 个")
        
        logger.info("\n✅ Step 1 (Fetch) 完成！")
        logger.info(f"下一步: 运行 pipeline_clean.py 清洗数据")
        logger.info(f"  uv run scripts/pipeline_clean.py --input {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

