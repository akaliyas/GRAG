"""
测试脚本：测试文档导入到 LightRAG 知识库

使用方式：
    uv run scripts/test_ingest.py [--input <json_file_path>]
    
示例：
    uv run scripts/test_ingest.py --input agent/tools/samples/google_generative-ai-docs_cleaned_docs.json
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 加载环境变量（从 .env 文件）
load_dotenv()

# 如果环境变量未设置，设置默认值（仅用于测试）
if not os.getenv('POSTGRES_PASSWORD'):
    os.environ['POSTGRES_PASSWORD'] = 'grag_password'
if not os.getenv('API_PASSWORD'):
    os.environ['API_PASSWORD'] = 'admin123'
if not os.getenv('DEEPSEEK_API_KEY'):
    os.environ['DEEPSEEK_API_KEY'] = 'your_deepseek_api_key_here'

from knowledge.lightrag_wrapper import LightRAGWrapper
from models.model_manager import ModelManager
from utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)


def test_ingest(json_file_path: str):
    """
    测试文档导入功能
    
    Args:
        json_file_path: JSON 文件路径（清洗后的文档）
    """
    logger.info("=" * 60)
    logger.info("开始测试文档导入功能")
    logger.info("=" * 60)
    
    # 初始化（需要先配置好数据库和模型）
    # 注意：模型配置已从 deepseek/local 改为 api/local
    # - api: API 模型（DeepSeek），当前使用
    # - local: 本地模型（Ollama），暂时禁用
    logger.info("初始化 ModelManager...")
    model_manager = ModelManager()
    
    # 验证模型配置
    current_model_type = model_manager.get_current_model_type()
    priority = model_manager.priority
    
    logger.info(f"当前模型类型: {current_model_type}")
    logger.info(f"模型优先级配置: {priority}")
    
    if current_model_type == 'none':
        logger.warning("⚠️  警告: 当前没有可用的模型")
        logger.warning("   请检查 DEEPSEEK_API_KEY 环境变量或 config.yaml 中的 models.api.api_key")
        return False
    
    # 初始化 LightRAGWrapper
    logger.info("初始化 LightRAGWrapper...")
    try:
        lightrag = LightRAGWrapper(model_manager)
        logger.info("✅ LightRAGWrapper 初始化成功")
    except Exception as e:
        logger.error(f"❌ LightRAGWrapper 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 导入清洗后的文档
    logger.info(f"开始导入文档: {json_file_path}")
    try:
        result = lightrag.ingest_from_json_file(json_file_path)
        
        if result.get('success', False):
            logger.info("=" * 60)
            logger.info("✅ 文档导入成功")
            logger.info("=" * 60)
            logger.info(f"成功导入文档数: {result.get('total_documents', 0)}")
            logger.info(f"数据源: {result.get('source', 'unknown')}")
            logger.info(f"仓库: {result.get('repo_url', 'unknown')}")
            return True
        else:
            logger.error("=" * 60)
            logger.error("❌ 文档导入失败")
            logger.error("=" * 60)
            logger.error(f"错误信息: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 导入文档时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="测试文档导入到 LightRAG 知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认测试文件
  uv run scripts/test_ingest.py
  
  # 指定自定义文件
  uv run scripts/test_ingest.py --input artifacts/02_clean/test_clean_v2.json
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='agent/tools/samples/google_generative-ai-docs_cleaned_docs.json',
        help='要导入的 JSON 文件路径（清洗后的文档）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 文件不存在: {input_path}")
        logger.error(f"   请检查路径是否正确")
        sys.exit(1)
    
    # 运行测试
    success = test_ingest(str(input_path))
    
    if success:
        logger.info("\n✅ 测试完成")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败")
        sys.exit(1)


if __name__ == '__main__':
    main()

