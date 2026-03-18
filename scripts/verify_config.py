"""
验证存储配置

快速验证存储配置是否正确设置。
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_storage_config():
    """验证存储配置"""
    print("=" * 60)
    print("存储配置验证".center(60))
    print("=" * 60)

    # 检查环境变量
    print("\n1. 环境变量:")
    storage_mode = os.environ.get('STORAGE_MODE')
    lightrag_storage = os.environ.get('LIGHTRAG_STORAGE_TYPE')
    deployment_mode = os.environ.get('DEPLOYMENT_MODE')

    print(f"   STORAGE_MODE: {storage_mode}")
    print(f"   LIGHTRAG_STORAGE_TYPE: {lightrag_storage}")
    print(f"   DEPLOYMENT_MODE: {deployment_mode}")

    # 检查配置文件
    print("\n2. 配置文件:")
    try:
        from config.config_manager import get_config
        config = get_config()
        lightrag_config = config.get_lightrag_config()

        configured_storage = lightrag_config.get('storage_type')
        kv_storage = lightrag_config.get('kv_storage')
        vector_storage = lightrag_config.get('vector_storage')
        graph_storage = lightrag_config.get('graph_storage')

        print(f"   storage_type: {configured_storage}")
        print(f"   kv_storage: {kv_storage}")
        print(f"   vector_storage: {vector_storage}")
        print(f"   graph_storage: {graph_storage}")
    except Exception as e:
        print(f"   错误: {e}")
        return False

    # 验证最终选择
    print("\n3. 存储模式决定:")

    # 按优先级选择
    final_storage = storage_mode or lightrag_storage or configured_storage or 'file'
    if final_storage.lower() in ['json', 'file']:
        final_storage = 'file'

    print(f"   最终存储类型: {final_storage}")

    if final_storage == 'file':
        print("   ✓ 使用本地文件存储（正确）")
        print("   ✓ 不需要 Neo4j 或 PostgreSQL")
        return True
    elif final_storage == 'postgresql':
        print("   ✗ 使用 PostgreSQL（需要数据库服务）")
        print("   💡 请设置 STORAGE_MODE=file 或 LIGHTRAG_STORAGE_TYPE=file")
        return False
    elif final_storage == 'neo4j':
        print("   ✗ 使用 Neo4j（需要图数据库服务）")
        print("   💡 请设置 STORAGE_MODE=file 或 LIGHTRAG_STORAGE_TYPE=file")
        return False
    else:
        print(f"   ? 未知存储类型: {final_storage}")
        return False

    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    success = verify_storage_config()

    if not success:
        print("\n请设置以下环境变量:")
        print("  PowerShell: $env:STORAGE_MODE='file'")
        print("  CMD:       set STORAGE_MODE=file")
        print("  Linux/Mac: export STORAGE_MODE=file")
        sys.exit(1)
    else:
        print("\n✓ 配置验证通过")
        print("\n可以启动服务:")
        print("  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
