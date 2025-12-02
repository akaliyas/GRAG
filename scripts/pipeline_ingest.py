"""
Pipeline Step 3: Ingest（导入到 LightRAG）
读取 Clean Artifact，导入到 LightRAG 知识库

使用方式：
    uv run scripts/pipeline_ingest.py --input artifacts/02_clean/openai_v1_clean.json
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
        logger.warning("   示例: uv run scripts/pipeline_ingest.py --input <input-path>")
        logger.warning("   当前将继续执行，但可能缺少依赖...")
        logger.warning("")

from knowledge.lightrag_wrapper import LightRAGWrapper
from models.model_manager import ModelManager
from utils.schema import CleanBatch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # 检查 uv 环境
    ensure_uv_environment()
    
    parser = argparse.ArgumentParser(description='导入文档到 LightRAG（Step 3: Ingest）')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件路径（Clean Artifact，如 artifacts/02_clean/openai_v1_clean.json）'
    )
    
    args = parser.parse_args()
    
    # 读取 Clean Artifact
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    logger.info(f"读取 Clean Artifact: {input_path}")
    try:
        clean_batch = CleanBatch.load_from_file(str(input_path))
        logger.info(f"✅ 成功读取 {len(clean_batch.docs)} 个文档")
    except Exception as e:
        logger.error(f"❌ 读取失败: {e}")
        sys.exit(1)
    
    # 初始化 LightRAG
    logger.info("初始化 LightRAG...")
    try:
        model_manager = ModelManager()
        lightrag = LightRAGWrapper(model_manager)
        logger.info("✅ LightRAG 初始化完成")
    except Exception as e:
        logger.error(f"❌ LightRAG 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 导入到 LightRAG
    logger.info("开始导入文档到 LightRAG...")
    logger.info(f"  数据源: {clean_batch.source_url}")
    logger.info(f"  文档数: {len(clean_batch.docs)}")
    
    try:
        result = lightrag.ingest_batch(clean_batch)
        
        if result.get('success'):
            logger.info("\n✅ Step 3 (Ingest) 完成！")
            logger.info(f"  成功导入 {result.get('total_documents', 0)} 个文档")
            logger.info(f"  数据源: {result.get('source_url', 'unknown')}")
            logger.info("\n📊 导入结果:")
            logger.info(f"  - 总文档数: {result.get('total_documents', 0)}")
            logger.info(f"  - Metadata 更新数: {result.get('metadata_updated', 0)}")
            logger.info(f"  - 状态: {'成功' if result.get('success') else '失败'}")
            
            logger.info("\n🎉 知识构建管道完成！")
            logger.info("下一步: 启动 API 服务进行查询")
            logger.info("  uvicorn api.main:app --reload")
        else:
            logger.error(f"❌ 导入失败: {result.get('error', result.get('message', '未知错误'))}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


