# GRAG 项目开发规则

## Python 脚本执行规则

### 1. 使用 `uv run` 执行所有 Python 脚本

**原因**: GRAG 项目使用 `uv` 作为包管理工具，必须通过 `uv run` 确保依赖正确加载。

**正确方式**:
```bash
# 执行 pipeline 脚本
uv run scripts/pipeline_fetch.py --repo <url> --output <path>
uv run scripts/pipeline_clean.py --input <path> --output <path>
uv run scripts/pipeline_ingest.py --input <path>

# 执行测试脚本
uv run scripts/test_agent.py
uv run scripts/test_context7_api.py

# 执行任何 Python 脚本
uv run python <script>.py
```

**错误方式**:
```bash
# 直接使用 python (可能导致依赖缺失)
python scripts/pipeline_fetch.py

# 使用 pip 安装的 python (可能使用错误的环境)
python3 scripts/pipeline_fetch.py
```

### 2. UTF-8 编码设置（Windows 环境）

**原因**: Windows 默认使用 GBK 编码，会导致 emoji 和中文输出乱码。

**脚本头部必须添加**:
```python
import sys

# 设置标准输出编码为 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
```

**使用 uv run 时自动启用 UTF-8**:
```bash
# 方式 1: 使用 -X 参数
uv run -X utf8 python script.py

# 方式 2: 在脚本中设置编码（推荐）
# 在所有脚本头部添加上述编码设置代码
```

## 代码规范

### 3. 环境变量和配置

**敏感信息**:
- API Keys 必须放在 `.env` 文件中
- `.env` 已在 `.gitignore` 中，不会被提交
- 非敏感配置放在 `config/config.yaml` 中

**配置读取**:
```python
# 使用 dotenv 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 读取配置
import os
api_key = os.getenv("API_KEY")

# 或者使用配置管理器
from config.config_manager import get_config
config = get_config()
```

### 4. 日志规范

**使用标准 logging 模块**:
```python
import logging

logger = logging.getLogger(__name__)

logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
logger.exception("异常日志（包含堆栈）")
```

**避免使用 print 输出日志**（调试时除外）

### 5. 错误处理

**静默失败策略**（针对可选功能）:
```python
# Context7 等可选功能应该静默失败
client = get_context7_client()
if not client:
    logger.info("Context7 不可用，跳过增强")
    return  # 不阻塞主流程
```

**明确抛出异常**（针对核心功能）:
```python
# 核心 API 调用应该明确处理错误
try:
    result = api_call()
except APIError as e:
    logger.error(f"API 调用失败: {e}")
    raise
```

### 6. 类型注解

**所有公共函数必须添加类型注解**:
```python
from typing import List, Optional

def fetch_docs(repo_url: str, max_files: Optional[int] = None) -> List[Document]:
    """获取仓库文档

    Args:
        repo_url: 仓库 URL
        max_files: 最大文件数（可选）

    Returns:
        文档列表
    """
    ...
```

## Git 提交规范

### 7. .gitignore 规则

**必须忽略的文件/目录**:
- `.env` - 环境变量（包含敏感信息）
- `.venv/` - 虚拟环境
- `__pycache__/` - Python 缓存
- `*.pyc` - 编译的 Python 文件
- `logs/` - 日志文件
- `artifacts/` - 中间产物（可选，根据项目需求）
- `.vscode/`, `.idea/` - IDE 配置

**配置文件**:
- 当前 `.gitignore` 已正确配置
- 添加新文件时检查是否应被忽略

## 开发工作流

### 8. Pipeline 脚本执行顺序

**标准顺序**:
```bash
# Step 1: Fetch（GitHub API）
uv run scripts/pipeline_fetch.py --repo <url> --output artifacts/01_raw/<name>_raw.json

# Step 2: Clean（数据清洗）
uv run scripts/pipeline_clean.py --input artifacts/01_raw/<name>_raw.json --output artifacts/02_clean/<name>_clean.json

# Step 2.5: Context7 增强（可选）
uv run scripts/pipeline_context7_enhance.py --input artifacts/02_clean/<name>_clean.json --output artifacts/02_clean/<name>_enhanced.json

# Step 3: Ingest（LightRAG）
uv run scripts/pipeline_ingest.py --input artifacts/02_clean/<name>_clean.json  # 或 <name>_enhanced.json
```

### 9. 测试规范

**运行测试前检查**:
```bash
# 1. 确认在项目根目录
pwd  # 应显示 /path/to/GRAG

# 2. 使用 uv run
uv run pytest tests/

# 3. 或者运行特定测试
uv run python scripts/test_agent.py
```

## 文档规范

### 10. Docstring 格式

**使用 Google-style docstrings**:
```python
def process_document(content: str) -> dict:
    """处理文档内容

    Args:
        content: 原始文档内容

    Returns:
        处理后的文档字典，包含以下字段：
            - title: 文档标题
            - content: 处理后的内容
            - metadata: 元数据字典

    Raises:
        ValueError: 当内容为空时

    Example:
        >>> process_document("Hello World")
        {'title': '', 'content': 'Hello World', 'metadata': {}}
    """
    ...
```

## 依赖管理

### 11. 依赖安装

**使用 uv 管理依赖**:
```bash
# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 同步依赖
uv sync
```

**不要使用 pip**:
```bash
# 错误方式（可能导致依赖不一致）
pip install <package-name>

# 正确方式
uv add <package-name>
```

## Context7 集成规范

### 12. Context7 使用规则

**生态位定位**:
- Context7 是**补充工具**，不是主要数据源
- 主要数据源仍是 GitHub API（零爬虫策略）
- Context7 用于：发现相关库、增强元数据、补充时效性

**使用原则**:
```python
# 1. 静默失败设计
client = get_context7_client()
if not client:
    logger.info("Context7 不可用，跳过")
    return  # 不阻塞主流程

# 2. 限流保护
# Context7 API 有速率限制，批量操作时注意

# 3. 缓存结果
# 文档更新不频繁，可以缓存几小时到几天
```

### 13. 网络和代理配置

**问题背景**:
GRAG 系统使用 OpenAI 兼容 API（SiliconFlow）进行嵌入操作，Python 的 httpx 库（OpenAI 库底层使用）会自动检测并使用 Windows 系统代理，即使代理被禁用。

**症状**:
- 查询请求超时（60秒）
- 日志显示 `ConnectTimeout` 或 `Failed to connect to 127.0.0.1:7897`
- Health 端点正常但 Query 端点超时

**根本原因**:
Windows 注册表中存在代理服务器配置：
```
ProxyEnable    = 0x0 (禁用)
ProxyServer    = 127.0.0.1:7897  ← httpx 仍会尝试使用
```

**诊断命令**:
```bash
# 检查 Windows 代理配置
reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" | findstr -i "proxy"

# 检查环境变量
echo HTTP_PROXY=%HTTP_PROXY%
echo HTTPS_PROXY=%HTTPS_PROXY%

# 测试代理连通性
curl -x http://127.0.0.1:7897 --connect-timeout 5 https://www.google.com
```

**解决方案（三选一）**:

**方案 A: 清除注册表代理配置**（推荐）
```bash
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings" /v ProxyServer /f
```

**方案 B: 在 .env 中设置 NO_PROXY**
```bash
# 在 .env 文件中添加
NO_PROXY=*
```

**方案 C: 修改代码显式禁用代理**
```python
# 在 knowledge/lightrag_wrapper.py 的 embedding_func 中
import httpx
client = httpx.Client(proxy=None)  # 显式禁用代理
```

**检查清单**（当遇到 API 超时时）:
1. ✅ 检查注册表代理配置
2. ✅ 检查环境变量（HTTP_PROXY, HTTPS_PROXY）
3. ✅ 测试外部 API 连通性
4. ✅ 查看日志中的 `ConnectTimeout` 错误
5. ✅ 确认代理服务是否运行

## 总结

**最重要的规则**:
1. ✅ 始终使用 `uv run` 执行脚本
2. ✅ 设置 UTF-8 编码（Windows）
3. ✅ 敏感信息放在 `.env`
4. ✅ 使用类型注解和 docstrings
5. ✅ 可选功能静默失败
6. ✅ Context7 作为补充工具
7. ✅ API 超时首先检查代理配置

**触发时机**: 每次执行 Python 脚本、创建新脚本、添加依赖时，都需要遵循这些规则。
