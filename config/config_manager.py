"""
配置管理模块
支持从 config.yaml 和环境变量读取配置
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的 config/config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 替换环境变量
        self._resolve_env_vars(self._config)
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        递归解析环境变量
        支持格式: ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            # 解析 ${VAR_NAME:default_value} 格式
            var_expr = obj[2:-1]
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_value = os.getenv(var_expr.strip())
                if var_value is None:
                    raise ValueError(f"环境变量 {var_expr} 未设置且无默认值")
                return var_value
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，如 "database.postgresql.host"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.get("database.postgresql", {})
    
    def get_lightrag_config(self) -> Dict[str, Any]:
        """获取 LightRAG 配置"""
        return self.get("lightrag", {})
    
    def get_model_config(self, model_type: str = "deepseek") -> Dict[str, Any]:
        """获取模型配置"""
        return self.get(f"models.{model_type}", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取 API 配置"""
        return self.get("api", {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self.get("cache", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get("logging", {})


@lru_cache()
def get_config() -> ConfigManager:
    """获取全局配置实例（单例）"""
    return ConfigManager()

