"""
测试 Agent 模块通过 API 调用
"""
import sys
import logging
import asyncio
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

def test_agent_direct_call():
    """直接测试 Agent 调用（不通过 API）"""
    try:
        from models.model_manager import ModelManager
        from knowledge.lightrag_wrapper import LightRAGWrapper
        from agent.grag_agent import GRAGAgent
        
        logger.info("=" * 60)
        logger.info("测试 1: 直接调用 Agent")
        logger.info("=" * 60)
        
        # 初始化组件
        logger.info("正在初始化组件...")
        model_manager = ModelManager()
        lightrag = LightRAGWrapper(model_manager)
        agent = GRAGAgent(model_manager, lightrag)
        logger.info("✅ 组件初始化成功")
        
        # 测试查询
        test_queries = [
            "什么是 Python？",
            "如何使用 OpenAI API？"
        ]
        
        for query in test_queries:
            logger.info(f"\n测试查询: {query}")
            result = agent.query(query, stream=False)
            
            if result.get("success"):
                answer = result.get("answer", "")
                context_ids = result.get("context_ids", [])
                logger.info(f"✅ 查询成功")
                logger.info(f"答案长度: {len(answer)} 字符")
                logger.info(f"上下文数量: {len(context_ids)}")
                logger.info(f"答案预览: {answer[:150]}...")
            else:
                logger.error(f"❌ 查询失败: {result.get('error')}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 直接调用测试完成")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_workflow():
    """测试 Agent 工作流（LangGraph）"""
    try:
        from models.model_manager import ModelManager
        from knowledge.lightrag_wrapper import LightRAGWrapper
        from agent.grag_agent import GRAGAgent
        
        logger.info("\n" + "=" * 60)
        logger.info("测试 2: Agent 工作流（LangGraph）")
        logger.info("=" * 60)
        
        # 初始化组件
        model_manager = ModelManager()
        lightrag = LightRAGWrapper(model_manager)
        agent = GRAGAgent(model_manager, lightrag)
        
        # 测试工作流
        test_query = "Python 是什么？"
        logger.info(f"测试查询: {test_query}")
        
        result = agent.query(test_query, stream=False)
        
        if result.get("success"):
            logger.info(f"✅ 工作流执行成功")
            logger.info(f"意图: {result.get('intent', 'unknown')}")
            logger.info(f"答案: {result.get('answer', '')[:200]}...")
            logger.info(f"上下文 IDs: {len(result.get('context_ids', []))} 个")
        else:
            logger.error(f"❌ 工作流执行失败: {result.get('error')}")
            return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 工作流测试完成")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ 工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("\n" + "=" * 60)
    logger.info("开始测试 Agent 模块（完整流程）")
    logger.info("=" * 60)
    
    # 测试 1: 直接调用
    if not test_agent_direct_call():
        logger.error("测试终止：直接调用失败")
        return
    
    # 测试 2: 工作流
    if not test_agent_workflow():
        logger.error("测试终止：工作流失败")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 所有测试通过！Agent 模块运行正常")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

