"""
数据库连接和配置检查脚本
用于诊断 PostgreSQL 连接问题
"""
import os
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
    
    required_vars = {
        'POSTGRES_HOST': os.getenv('POSTGRES_HOST', '未设置'),
        'POSTGRES_PORT': os.getenv('POSTGRES_PORT', '未设置'),
        'POSTGRES_DB': os.getenv('POSTGRES_DB', '未设置'),
        'POSTGRES_USER': os.getenv('POSTGRES_USER', '未设置'),
        'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD', '未设置（已隐藏）'),
        'EMBEDDING_DIM': os.getenv('EMBEDDING_DIM', '未设置'),
    }
    
    for key, value in required_vars.items():
        if key == 'POSTGRES_PASSWORD' and value != '未设置（已隐藏）':
            value = '***已设置***'
        print(f"  {key}: {value}")
    
    print()

def check_db_connection():
    """检查数据库连接"""
    print("=" * 60)
    print("2. 数据库连接测试")
    print("=" * 60)
    
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = int(os.getenv('POSTGRES_PORT', 5432))
            database = os.getenv('POSTGRES_DB', 'grag_db')
            user = os.getenv('POSTGRES_USER', 'grag_user')
            password = os.getenv('POSTGRES_PASSWORD', '')
            
            try:
                conn = await asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
                
                # 检查扩展
                extensions = await conn.fetch("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname IN ('vector', 'plpgsql')
                    ORDER BY extname
                """)
                
                print(f"  ✅ 数据库连接成功")
                print(f"    主机: {host}:{port}")
                print(f"    数据库: {database}")
                print(f"    用户: {user}")
                print(f"  已安装的扩展:")
                for ext in extensions:
                    print(f"    - {ext['extname']}: {ext['extversion']}")
                
                # 检查表
                tables = await conn.fetch("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename LIKE 'LIGHTRAG%'
                    ORDER BY tablename
                """)
                
                if tables:
                    print(f"  LightRAG 表 ({len(tables)} 个):")
                    for table in tables:
                        print(f"    - {table['tablename']}")
                else:
                    print(f"  ⚠️  未找到 LightRAG 表（首次运行时会自动创建）")
                
                await conn.close()
                return True
                
            except Exception as e:
                print(f"  ❌ 数据库连接失败: {e}")
                return False
        
        result = asyncio.run(test_connection())
        print()
        return result
        
    except ImportError:
        print("  ⚠️  asyncpg 未安装，跳过连接测试")
        print("  安装命令: pip install asyncpg")
        print()
        return False

def check_lightrag_config():
    """检查 LightRAG 配置"""
    print("=" * 60)
    print("3. LightRAG 配置检查")
    print("=" * 60)
    
    try:
        from config.config_manager import get_config
        
        config = get_config()
        db_config = config.get_database_config()
        lightrag_config = config.get_lightrag_config()
        
        print(f"  数据库配置:")
        print(f"    Host: {db_config.get('host')}")
        print(f"    Port: {db_config.get('port')}")
        print(f"    Database: {db_config.get('database')}")
        print(f"    User: {db_config.get('user')}")
        
        print(f"  LightRAG 配置:")
        print(f"    存储类型: {lightrag_config.get('storage_type')}")
        print(f"    嵌入模型: {lightrag_config.get('embedding_model')}")
        print(f"    嵌入提供商: {lightrag_config.get('embedding_provider')}")
        print()
        
    except Exception as e:
        print(f"  ❌ 配置读取失败: {e}")
        import traceback
        traceback.print_exc()
        print()

def main():
    print("\n" + "=" * 60)
    print("PostgreSQL 数据库配置检查")
    print("=" * 60 + "\n")
    
    check_env_vars()
    check_db_connection()
    check_lightrag_config()
    
    print("=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == '__main__':
    main()

