"""
Debug script to inspect what prompts LightRAG is sending to the LLM
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def debug_prompts():
    """Debug what prompts are being sent to the LLM"""
    from knowledge.lightrag_wrapper import LightRAGWrapper
    from config.config_manager import get_config

    # Monkey-patch the llm_func to log the prompts
    config = get_config()
    wrapper = LightRAGWrapper(config)

    # Store the original llm_func
    original_llm_func = wrapper.llm_func

    async def logged_llm_func(messages, **kwargs):
        """Wrapper to log LLM calls"""
        logger.debug(f"\n{'='*80}")
        logger.debug(f"LLM CALLED WITH {len(messages)} MESSAGES")
        logger.debug(f"{'='*80}")

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            logger.debug(f"\n--- Message {i+1} [{role}] ---")
            logger.debug(content)

        logger.debug(f"\n{'='*80}")
        logger.debug(f"KWARGS: {kwargs}")
        logger.debug(f"{'='*80}\n")

        # Call original
        result = await original_llm_func(messages, **kwargs)

        logger.debug(f"\n{'='*80}")
        logger.debug(f"LLM RESPONSE:")
        logger.debug(f"{'='*80}")
        logger.debug(result)
        logger.debug(f"{'='*80}\n")

        return result

    # Replace the llm_func
    wrapper.llm_func = logged_llm_func

    # Simple test text
    test_text = """
    FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

    Install FastAPI using pip:
    pip install fastapi
    pip install uvicorn[standard]
    """

    print("\n" + "="*80)
    print("Testing entity extraction with debug logging")
    print("="*80 + "\n")

    result = await wrapper.ainsert(test_text, id="debug_test_doc")
    print(f"\nResult: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_prompts())
