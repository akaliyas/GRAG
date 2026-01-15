"""
诊断脚本：检查环境和配置

用于诊断以下问题：
- 环境变量配置
- 数据库连接
- LightRAG 包导入
- 模型配置

使用方式：
    uv run scripts/test_diagnostics.py [--check <check_type>]
    
示例：
    uv run scripts/test_diagnostics.py                    # 运行所有检查
    uv run scripts/test_diagnostics.py --check env         # 只检查环境变量
    uv run scripts/test_diagnostics.py --check db         # 只检查数据库
    uv run scripts/test_diagnostics.py --check lightrag    # 只检查 LightRAG
    uv run scripts/test_diagnostics.py --check model       # 只检查模型配置
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def check_env_vars():
    """检查环境变量配置"""
    print("=" * 60)
    print("1. 环境变量检查")
    print("=" * 60)
    
    env_vars = {
        'HTTP_PROXY': os.environ.get('HTTP_PROXY', '未设置'),
        'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', '未设置'),
        'POSTGRES_HOST': os.environ.get('POSTGRES_HOST', '未设置'),
        'POSTGRES_PORT': os.environ.get('POSTGRES_PORT', '未设置'),
        'POSTGRES_DATABASE': os.getenv('POSTGRES_DATABASE', '未设置'),
        'POSTGRES_DB': os.getenv('POSTGRES_DB', '未设置'),
        'POSTGRES_USER': os.getenv('POSTGRES_USER', '未设置'),
        'POSTGRES_PASSWORD': '***已设置***' if os.getenv('POSTGRES_PASSWORD') else '未设置',
        'DEEPSEEK_API_KEY': '***已设置***' if os.getenv('DEEPSEEK_API_KEY') else '未设置',
        'EMBEDDING_API_KEY': '***已设置***' if os.getenv('EMBEDDING_API_KEY') else '未设置',
    }
    
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
    
    print()


def check_db_config():
    """检查数据库配置"""
    print("=" * 60)
    print("2. 数据库配置检查")
    print("=" * 60)
    
    try:
        from config.config_manager import get_config
        
        config = get_config()
        db_config = config.get_database_config()
        
        print("配置中的数据库信息:")
        print(f"  - Host: {db_config.get('host', '未设置')}")
        print(f"  - Port: {db_config.get('port', '未设置')}")
        print(f"  - Database: {db_config.get('database', '未设置')}")
        print(f"  - User: {db_config.get('user', '未设置')}")
        print(f"  - Password: {'***已设置***' if db_config.get('password') else '未设置'}")
        print(f"  - Pool Size: {db_config.get('pool_size', '未设置')}")
        print()
        
    except Exception as e:
        print(f"❌ 无法读取数据库配置: {e}")
        print()


def check_db_connection():
    """检查数据库连接"""
    print("=" * 60)
    print("3. 数据库连接测试")
    print("=" * 60)
    
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            try:
                from config.config_manager import get_config
                config = get_config()
                db_config = config.get_database_config()
                
                conn = await asyncpg.connect(
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    user=db_config.get('user'),
                    password=db_config.get('password'),
                    database=db_config.get('database')
                )
                
                # 测试查询
                result = await conn.fetchval('SELECT version()')
                print(f"✅ 数据库连接成功")
                print(f"   PostgreSQL 版本: {result.split(',')[0]}")
                await conn.close()
                return True
                
            except Exception as e:
                print(f"❌ 数据库连接失败: {e}")
                print(f"   请检查:")
                print(f"   1. PostgreSQL 服务是否运行")
                print(f"   2. 数据库配置是否正确")
                print(f"   3. 数据库实例是否已创建")
                return False
        
        success = asyncio.run(test_connection())
        print()
        return success
        
    except ImportError:
        print("⚠️  asyncpg 未安装，跳过数据库连接测试")
        print("   安装命令: pip install asyncpg")
        print()
        return None
    except Exception as e:
        print(f"❌ 数据库连接测试失败: {e}")
        print()
        return False


def check_lightrag_import():
    """检查 LightRAG 包导入"""
    print("=" * 60)
    print("4. LightRAG 包检查")
    print("=" * 60)
    
    # 方法 1: 直接尝试导入
    try:
        import lightrag
        print(f"✅ import lightrag 成功")
        print(f"   模块位置: {lightrag.__file__}")
        print(f"   模块属性: {', '.join(dir(lightrag)[:10])}...")
    except ImportError as e:
        print(f"❌ import lightrag 失败: {e}")
        print()
        return False
    
    # 方法 2: 检查包信息
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution('lightrag-hku')
        print(f"\n✅ 包信息:")
        print(f"   包名: {dist.project_name}")
        print(f"   版本: {dist.version}")
        print(f"   位置: {dist.location}")
    except Exception as e:
        print(f"\n⚠️  无法获取包信息: {e}")
    
    # 方法 3: 检查实际导入
    try:
        from lightrag import LightRAG
        print(f"\n✅ from lightrag import LightRAG 成功")
    except ImportError as e:
        print(f"\n❌ from lightrag import LightRAG 失败: {e}")
        print(f"   错误详情: {type(e).__name__}: {e}")
        print()
        return False
    
    # 方法 4: 检查 Python 环境
    print(f"\n✅ Python 环境:")
    print(f"   当前 Python: {sys.executable}")
    print(f"   是否在 .venv: {'.venv' in sys.executable}")
    
    # 检查 lightrag-hku 是否在当前环境中
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "lightrag-hku"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"\n✅ pip show lightrag-hku:")
            print(result.stdout)
        else:
            print(f"\n⚠️  pip show lightrag-hku 失败:")
            print(result.stderr)
    except Exception as e:
        print(f"\n⚠️  无法检查包信息: {e}")
    
    print()
    return True


def check_model_config():
    """检查模型配置"""
    print("=" * 60)
    print("5. 模型配置检查")
    print("=" * 60)
    
    try:
        from config.config_manager import get_config
        
        config = get_config()
        api_config = config.get_model_config("api")
        local_config = config.get_model_config("local")
        
        print("API 模型配置:")
        print(f"  - API Key 已设置: {'是' if api_config.get('api_key') else '否'}")
        print(f"  - Base URL: {api_config.get('base_url', '未设置')}")
        print(f"  - Model Name: {api_config.get('model_name', '未设置')}")
        print(f"  - Temperature: {api_config.get('temperature', '未设置')}")
        print(f"  - Max Tokens: {api_config.get('max_tokens', '未设置')}")
        
        print("\n本地模型配置（暂时禁用）:")
        print(f"  - Base URL: {local_config.get('base_url', '未设置')}")
        print(f"  - Model Name: {local_config.get('model_name', '未设置')}")
        
        print("\n模型切换策略:")
        model_switch = config.get("model_switch", {})
        print(f"  - Priority: {model_switch.get('priority', '未设置')}")
        print(f"  - Fallback to API: {model_switch.get('fallback_to_api', False)}")
        
        # 检查 LightRAG 配置
        lightrag_config = config.get_lightrag_config()
        print("\nLightRAG 配置:")
        print(f"  - LLM Model: {lightrag_config.get('llm_model', '未设置')}")
        print(f"  - Embedding Model: {lightrag_config.get('embedding_model', '未设置')}")
        print(f"  - Embedding Provider: {lightrag_config.get('embedding_provider', '未设置')}")
        print(f"  - Storage Type: {lightrag_config.get('storage_type', '未设置')}")
        print()
        
    except Exception as e:
        print(f"❌ 无法读取模型配置: {e}")
        import traceback
        traceback.print_exc()
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="诊断脚本：检查环境和配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
检查类型:
  env       - 检查环境变量
  db        - 检查数据库配置和连接
  lightrag  - 检查 LightRAG 包导入
  model     - 检查模型配置
  all       - 运行所有检查（默认）

示例:
  uv run scripts/test_diagnostics.py                    # 运行所有检查
  uv run scripts/test_diagnostics.py --check env       # 只检查环境变量
  uv run scripts/test_diagnostics.py --check db         # 只检查数据库
        """
    )
    
    parser.add_argument(
        '--check',
        type=str,
        choices=['env', 'db', 'lightrag', 'model', 'all'],
        default='all',
        help='要运行的检查类型'
    )
    
    args = parser.parse_args()
    
    check_type = args.check
    
    if check_type == 'all' or check_type == 'env':
        check_env_vars()
    
    if check_type == 'all' or check_type == 'db':
        check_db_config()
        check_db_connection()
    
    if check_type == 'all' or check_type == 'lightrag':
        check_lightrag_import()
    
    if check_type == 'all' or check_type == 'model':
        check_model_config()
    
    print("=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

