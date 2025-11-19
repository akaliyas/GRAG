"""
日志模块
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.config_manager import get_config


def setup_logger():
    """设置日志配置"""
    config = get_config()
    logging_config = config.get_logging_config()
    
    # 创建日志目录
    log_file = logging_config.get('file', 'logs/grag.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 日志级别
    log_level = getattr(logging, logging_config.get('level', 'DEBUG').upper())
    
    # 日志格式
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=logging_config.get('max_bytes', 10485760),  # 10MB
        backupCount=logging_config.get('backup_count', 5),
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    return root_logger

