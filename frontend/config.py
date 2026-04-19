"""
前端配置文件
从环境变量读取 API 配置
"""
import os

# API 配置
API_BASE_URL = os.getenv(
    "GRAG_API_URL",
    f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}/api/v1"
)
API_HEALTH_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}/health"

# API 认证配置
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "")
