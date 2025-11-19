"""
API 认证模块
"""
import hashlib
import secrets
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from config.config_manager import get_config

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Security(security)) -> str:
    """
    验证用户凭证
    
    Args:
        credentials: HTTP Basic 认证凭证
        
    Returns:
        用户名
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    config = get_config()
    api_config = config.get_api_config()
    auth_config = api_config.get("auth", {})
    
    if not auth_config.get("enabled", True):
        return credentials.username
    
    correct_username = auth_config.get("username", "admin")
    correct_password = auth_config.get("password", "")
    
    if not correct_password:
        raise HTTPException(
            status_code=500,
            detail="API 密码未配置，请设置环境变量 API_PASSWORD"
        )
    
    # 使用 secrets.compare_digest 防止时序攻击
    is_correct_username = secrets.compare_digest(credentials.username, correct_username)
    is_correct_password = secrets.compare_digest(credentials.password, correct_password)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

