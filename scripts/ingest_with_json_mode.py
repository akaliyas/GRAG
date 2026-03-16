"""
使用JSON模式摄取数据到知识库
"""
import sys
import os
from pathlib import Path

# 强制使用JSON文件存储模式
os.environ['STORAGE_MODE'] = 'json'
os.environ['LIGHTRAG_STORAGE_TYPE'] = 'file'
os.environ['LIGHTRAG_GRAPH_STORAGE'] = 'NetworkXStorage'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
from scripts.pipeline_ingest import main as ingest_main

if __name__ == "__main__":
    # 设置命令行参数
    sys.argv = ['pipeline_ingest.py', '--input', 'artifacts/02_clean/openai_cookbook_clean.json']

    print("=" * 60)
    print("OpenAI Cookbook Data Ingestion")
    print("Storage mode: JSON (no external database)")
    print("=" * 60)

    # 运行摄取
    ingest_main()
