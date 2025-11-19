"""
模型管理模块
支持 DeepSeek API 和本地模型（Ollama/vLLM）的动态切换
优先使用本地模型，失败时自动回退到 API
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
        self.temperature = config.get('temperature', 0.7)
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


class LocalLLM(BaseLLM):
    """本地模型（Ollama/vLLM，兼容 OpenAI API）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地模型客户端
        
        Args:
            config: 模型配置字典
        """
        if OpenAI is None:
            raise ImportError("需要安装 openai 库: pip install openai")
        
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'qwen2.5:7b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.timeout = config.get('timeout', 60)
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed"  # 本地模型通常不需要 API key
        )
        logger.info(f"本地模型客户端已初始化: {self.base_url}, 模型: {self.model_name}")
    
    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        调用本地模型进行聊天补全
        
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
                stream=stream,
                timeout=self.timeout
            )
            return response
        except Exception as e:
            logger.error(f"本地模型调用失败: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查本地模型是否可用"""
        try:
            # 简单的健康检查
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5
            )
            return test_response is not None
        except Exception:
            return False


class ModelManager:
    """模型管理器，支持动态切换"""
    
    def __init__(self):
        """初始化模型管理器"""
        config = get_config()
        model_switch_config = config.get("model_switch", {})
        
        # 初始化模型
        self.deepseek_llm = None
        self.local_llm = None
        
        try:
            deepseek_config = config.get_model_config("deepseek")
            if deepseek_config.get('api_key'):
                self.deepseek_llm = DeepSeekLLM(deepseek_config)
        except Exception as e:
            logger.warning(f"DeepSeek API 初始化失败: {e}")
        
        try:
            local_config = config.get_model_config("local")
            self.local_llm = LocalLLM(local_config)
        except Exception as e:
            logger.warning(f"本地模型初始化失败: {e}")
        
        # 切换策略
        self.priority = model_switch_config.get('priority', 'local')  # 优先使用本地模型
        self.fallback_to_api = model_switch_config.get('fallback_to_api', True)
        
        # 当前使用的模型
        self._current_model: Optional[BaseLLM] = None
        self._select_model()
        
        logger.info(f"模型管理器已初始化，当前模型: {self._current_model.__class__.__name__ if self._current_model else 'None'}")
    
    def _select_model(self):
        """根据优先级选择模型"""
        if self.priority == 'local' and self.local_llm and self.local_llm.is_available():
            self._current_model = self.local_llm
            logger.info("已选择本地模型")
        elif self.deepseek_llm and self.deepseek_llm.is_available():
            self._current_model = self.deepseek_llm
            logger.info("已选择 DeepSeek API")
        else:
            self._current_model = None
            logger.warning("没有可用的模型")
    
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
        
        try:
            return model.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            
            # 如果启用回退且当前是本地模型，尝试切换到 API
            if self.fallback_to_api and isinstance(model, LocalLLM) and self.deepseek_llm:
                logger.info("尝试回退到 DeepSeek API")
                try:
                    self._current_model = self.deepseek_llm
                    return self.deepseek_llm.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream
                    )
                except Exception as e2:
                    logger.error(f"DeepSeek API 回退也失败: {e2}")
            
            raise
    
    def switch_model(self, model_type: str) -> bool:
        """
        手动切换模型
        
        Args:
            model_type: 模型类型 ('local' 或 'deepseek')
            
        Returns:
            是否切换成功
        """
        if model_type == 'local' and self.local_llm and self.local_llm.is_available():
            self._current_model = self.local_llm
            logger.info("已切换到本地模型")
            return True
        elif model_type == 'deepseek' and self.deepseek_llm and self.deepseek_llm.is_available():
            self._current_model = self.deepseek_llm
            logger.info("已切换到 DeepSeek API")
            return True
        else:
            logger.warning(f"无法切换到 {model_type}，模型不可用")
            return False
    
    def get_current_model_type(self) -> str:
        """获取当前模型类型"""
        if isinstance(self._current_model, LocalLLM):
            return 'local'
        elif isinstance(self._current_model, DeepSeekLLM):
            return 'deepseek'
        else:
            return 'none'

