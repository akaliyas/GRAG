"""
模型管理模块
支持 API 模型（DeepSeek）和本地模型（Ollama）的动态切换
注意：本地模型（Ollama）暂时禁用，优先使用 API
"""
import logging
from typing import Optional, Dict, Any, Iterator
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from config.config_manager import get_config

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """LLM 基类"""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Any:
        """聊天补全"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass


class DeepSeekLLM(BaseLLM):
    """DeepSeek API 模型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DeepSeek API 客户端

        Args:
            config: 模型配置字典
        """
        if OpenAI is None:
            raise ImportError("需要安装 openai 库: pip install openai")

        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("DeepSeek API key 未设置")

        self.base_url = config.get('base_url', 'https://api.deepseek.com')
        self.model_name = config.get('model_name', 'deepseek-chat')
        self.temperature = config.get('temperature', 0.3)  # ✅ 优化：较低温度提升指令遵循
        self.max_tokens = config.get('max_tokens', 2000)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"DeepSeek API 客户端已初始化: {self.base_url}")

    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        调用 DeepSeek API 进行聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式返回

        Returns:
            API 响应对象或流式迭代器
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            raise

    def is_available(self) -> bool:
        """检查 DeepSeek API 是否可用"""
        try:
            # 简单的健康检查：发送一个短消息
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return test_response is not None
        except Exception:
            return False


class QwenLLM(BaseLLM):
    """Qwen API 模型 (Alibaba DashScope)"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Qwen API 客户端

        Args:
            config: 模型配置字典，包含:
                - api_key: DashScope API Key
                - base_url: API 基础 URL (默认: https://dashscope.aliyuncs.com/compatible-mode/v1)
                - model_name: 模型名称 (qwen-plus, qwen-turbo, qwen-max, qwen-max-latest)
                - temperature: 温度参数
                - max_tokens: 最大 token 数
        """
        if OpenAI is None:
            raise ImportError("需要安装 openai 库: pip install openai")

        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Qwen API key 未设置，请在 .env 中配置 QWEN_API_KEY")

        # DashScope OpenAI 兼容模式
        self.base_url = config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.model_name = config.get('model_name', 'qwen-plus')
        self.temperature = config.get('temperature', 0.3)  # Qwen 建议较低温度以提升指令遵循
        self.max_tokens = config.get('max_tokens', 2000)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"Qwen API 客户端已初始化: {self.base_url}")
        logger.info(f"Qwen 模型: {self.model_name}")

    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        调用 Qwen API 进行聊天补全

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数 (Qwen 建议使用较低温度以获得更好的结构化输出)
            max_tokens: 最大 token 数
            stream: 是否流式返回

        Returns:
            API 响应对象或流式迭代器

        Raises:
            Exception: API 调用失败时抛出异常
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"Qwen API 调用失败: {e}")
            raise

    def is_available(self) -> bool:
        """检查 Qwen API 是否可用"""
        try:
            # 简单的健康检查：发送一个短消息
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return test_response is not None
        except Exception as e:
            logger.warning(f"Qwen API 健康检查失败: {e}")
            return False


# ============================================================================
# Ollama 本地模型支持已弃用
# ============================================================================
class LocalLLM(BaseLLM):
    """本地模型（Ollama/vLLM，兼容 OpenAI API）- 已弃用"""
    
    def __init__(self, config: Dict[str, Any]):
        """已弃用：Ollama 本地模型支持"""
        logger.warning("LocalLLM 已弃用，请使用 DeepSeek API")
        pass
    
    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """已弃用：Ollama 本地模型支持"""
        raise NotImplementedError("LocalLLM 已弃用，请使用 DeepSeek API")
    
    def is_available(self) -> bool:
        """已弃用：Ollama 本地模型支持"""
        return False


class ModelManager:
    """模型管理器，支持动态切换 (DeepSeek/Qwen/本地)"""

    def __init__(self):
        """初始化模型管理器"""
        config = get_config()
        model_switch_config = config.get("model_switch", {})

        # 初始化模型
        self.api_llm = None  # API 模型（DeepSeek）
        self.qwen_llm = None  # Qwen 模型 (Alibaba DashScope)
        self.local_llm = None  # 本地模型（Ollama）- 暂时禁用

        # 初始化 DeepSeek API 模型
        try:
            api_config = config.get_model_config("ds")  # ds = DeepSeek
            if api_config.get('api_key'):
                self.api_llm = DeepSeekLLM(api_config)
                logger.info("DeepSeek API 模型初始化成功")
        except Exception as e:
            logger.warning(f"DeepSeek API 模型初始化失败: {e}")

        # 初始化 Qwen API 模型
        try:
            qwen_config = config.get_model_config("qwen")
            if qwen_config.get('api_key'):
                self.qwen_llm = QwenLLM(qwen_config)
                logger.info("Qwen API 模型初始化成功")
        except Exception as e:
            logger.warning(f"Qwen API 模型初始化失败: {e}")

        # ========================================================================
        # Ollama 本地模型暂时禁用（配置保留，但不初始化）
        # ========================================================================
        # try:
        #     local_config = config.get_model_config("local")
        #     self.local_llm = LocalLLM(local_config)
        # except Exception as e:
        #     logger.warning(f"本地模型初始化失败: {e}")

        # 切换策略
        self.priority = model_switch_config.get('priority', 'api')  # 默认使用 API
        self.fallback_to_api = model_switch_config.get('fallback_to_api', False)  # local 失败时回退到 api

        # 当前使用的模型
        self._current_model: Optional[BaseLLM] = None
        self._select_model()

        logger.info(f"模型管理器已初始化，当前模型: {self._current_model.__class__.__name__ if self._current_model else 'None'}")
    
    def _select_model(self):
        """根据优先级选择模型"""
        # ========================================================================
        # 模型选择逻辑：支持 ds=DeepSeek, qwen, local (已弃用)
        # ========================================================================
        if self.priority == 'local':
            # local 暂时禁用
            logger.warning("⚠️ 本地模型（Ollama）暂时禁用，切换到 ds (DeepSeek)")
            if self.api_llm:
                if self.api_llm.is_available():
                    self._current_model = self.api_llm
                    self._wrap_chat_completion(self._current_model)
                    logger.info("✅ 已选择 ds 模型（DeepSeek）")
                else:
                    self._current_model = None
                    logger.warning("⚠️ ds 模型不可用（健康检查失败）")
                    logger.info("💡 提示: 请检查 DEEPSEEK_API_KEY 环境变量是否正确设置")
            else:
                self._current_model = None
                logger.warning("⚠️ ds 模型未初始化")

        elif self.priority in ('ds', 'deepseek', 'api'):
            # 支持三种写法：ds, deepseek, api(向后兼容)
            if self.api_llm:
                if self.api_llm.is_available():
                    self._wrap_chat_completion(self.api_llm)
                    self._current_model = self.api_llm
                    logger.info("✅ 已选择 ds 模型（DeepSeek）")
                else:
                    self._current_model = None
                    logger.warning("⚠️ ds 模型不可用（健康检查失败）")
                    logger.info("💡 提示: 请检查 DEEPSEEK_API_KEY 环境变量是否正确设置")
            else:
                self._current_model = None
                logger.warning("⚠️ ds 模型未初始化")

        elif self.priority == 'qwen':
            if self.qwen_llm:
                if self.qwen_llm.is_available():
                    self._wrap_chat_completion(self.qwen_llm)
                    self._current_model = self.qwen_llm
                    logger.info("✅ 已选择 qwen 模型")
                else:
                    self._current_model = None
                    logger.warning("⚠️ qwen 模型不可用（健康检查失败）")
                    logger.info("💡 提示: 请检查 QWEN_API_KEY 环境变量是否正确设置")
            else:
                self._current_model = None
                logger.warning("⚠️ qwen 模型未初始化")

        else:
            # 未知优先级，默认使用 ds
            logger.warning(f"未知的优先级配置: {self.priority}，使用 ds (DeepSeek)")
            if self.api_llm and self.api_llm.is_available():
                self._wrap_chat_completion(self.api_llm)
                self._current_model = self.api_llm
                logger.info("已选择 ds 模型（DeepSeek）")
            else:
                self._current_model = None
                logger.warning("没有可用的模型")

    def _wrap_chat_completion(self, model: BaseLLM):
        """
        包装 chat_completion 方法，兼容字符串输入

        Args:
            model: 要包装的模型实例
        """
        original_chat_completion = model.chat_completion

        def wrapped_chat_completion(messages, *args, **kwargs):
            """
            包装 chat_completion 调用，若传入字符串 prompt，则转换为标准 messages 列表
            """
            # API 需要 messages 为列表
            if isinstance(messages, str):
                # 将字符串 prompt 包装为标准 user 消息
                messages = [{"role": "user", "content": messages}]
            return original_chat_completion(messages, *args, **kwargs)

        # 替换原有方法，仅用于当前管理器实例
        model.chat_completion = wrapped_chat_completion
    
    def get_model(self) -> Optional[BaseLLM]:
        """
        获取当前模型
        
        Returns:
            当前可用的模型实例
        """
        # 检查当前模型是否仍然可用
        if self._current_model and self._current_model.is_available():
            return self._current_model
        
        # 如果当前模型不可用，尝试切换
        self._select_model()
        return self._current_model
    
    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        调用模型进行聊天补全，支持自动回退
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式返回
            
        Returns:
            API 响应对象或流式迭代器
            
        Raises:
            RuntimeError: 所有模型都不可用时抛出异常
        """
        model = self.get_model()
        
        if model is None:
            raise RuntimeError("没有可用的模型")
        
        # 兼容传入字符串 prompt（部分上游可能直接给字符串）
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif messages and isinstance(messages, list) and isinstance(messages[0], str):
            # 兼容 List[str]
            messages = [{"role": "user", "content": m} for m in messages]
        
        try:
            return model.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            
            # ====================================================================
            # 回退逻辑：local 失败时回退到 api（如果启用）
            # ====================================================================
            if self.fallback_to_api and isinstance(model, LocalLLM) and self.api_llm:
                logger.info("尝试回退到 API 模型")
                try:
                    self._current_model = self.api_llm
                    return self.api_llm.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream
                    )
                except Exception as e2:
                    logger.error(f"API 模型回退也失败: {e2}")
            
            raise
    
    def switch_model(self, model_type: str) -> bool:
        """
        手动切换模型

        Args:
            model_type: 模型类型 ('ds'=DeepSeek, 'qwen'=Qwen, 'local'=Ollama已弃用)

        Returns:
            是否切换成功
        """
        if model_type == 'local':
            # local 暂时禁用
            logger.warning("本地模型（Ollama）暂时禁用，无法切换")
            return False

        if model_type in ('ds', 'deepseek', 'api'):
            # 支持三种写法：ds, deepseek, api(向后兼容)
            if self.api_llm and self.api_llm.is_available():
                self._current_model = self.api_llm
                logger.info("已切换到 DeepSeek 模型")
                return True
            else:
                logger.warning("无法切换到 DeepSeek，模型不可用")
                return False

        if model_type == 'qwen':
            if self.qwen_llm and self.qwen_llm.is_available():
                self._current_model = self.qwen_llm
                logger.info("已切换到 Qwen 模型")
                return True
            else:
                logger.warning("无法切换到 Qwen，模型不可用或未配置")
                return False

        logger.warning(f"未知的模型类型: {model_type}，支持的选项: 'ds'(DeepSeek), 'qwen', 'local'(已弃用)")
        return False

    def get_current_model_type(self) -> str:
        """获取当前模型类型"""
        if isinstance(self._current_model, LocalLLM):
            return 'local'
        elif isinstance(self._current_model, DeepSeekLLM):
            return 'ds'  # DeepSeekLLM 对应 ds/deepseek
        elif isinstance(self._current_model, QwenLLM):
            return 'qwen'  # QwenLLM 对应 qwen
        else:
            return 'none'

