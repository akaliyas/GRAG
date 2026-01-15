"""
测试 Agent 模块
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_agent_import():
    """测试 Agent 模块导入"""
    try:
        from agent.grag_agent import GRAGAgent
        logger.info("✅ Agent 模块导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ Agent 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_initialization():
    """测试 Agent 初始化"""
    try:
        from models.model_manager import ModelManager
        from knowledge.lightrag_wrapper import LightRAGWrapper
        from agent.grag_agent import GRAGAgent
        
        logger.info("正在初始化模型管理器...")
        model_manager = ModelManager()
        logger.info("✅ 模型管理器初始化成功")
        
        logger.info("正在初始化 LightRAG...")
        lightrag = LightRAGWrapper(model_manager)
        logger.info("✅ LightRAG 初始化成功")
        
        logger.info("正在初始化 Agent...")
        agent = GRAGAgent(model_manager, lightrag)
        logger.info("✅ Agent 初始化成功")
        
        return agent
    except Exception as e:
        logger.error(f"❌ Agent 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_query(agent):
    """测试 Agent 查询"""
    if agent is None:
        logger.error("❌ Agent 未初始化，无法测试查询")
        return False
    
    try:
        logger.info("正在测试 Agent 查询...")
        test_query = "什么是 Python？"
        result = agent.query(test_query, stream=False)
        
        if result.get("success"):
            logger.info(f"✅ Agent 查询成功")
            logger.info(f"答案: {result.get('answer', '')[:100]}...")
            logger.info(f"上下文 IDs: {result.get('context_ids', [])}")
            return True
        else:
            logger.error(f"❌ Agent 查询失败: {result.get('error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Agent 查询异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("开始测试 Agent 模块")
    logger.info("=" * 60)
    
    # 测试 1: 导入
    if not test_agent_import():
        logger.error("测试终止：导入失败")
        return
    
    # 测试 2: 初始化
    agent = test_agent_initialization()
    if agent is None:
        logger.error("测试终止：初始化失败")
        return
    
    # 测试 3: 查询
    if not test_agent_query(agent):
        logger.error("测试终止：查询失败")
        return
    
    logger.info("=" * 60)
    logger.info("✅ 所有测试通过！")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

