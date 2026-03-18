#!/bin/bash
# GRAG 后端启动脚本 (Linux/Mac)
# 自动设置正确的环境变量并启动服务

echo "========================================"
echo " GRAG Backend Startup"
echo "========================================"
echo ""

# 设置环境变量
export DEPLOYMENT_MODE=local
export LIGHTRAG_STORAGE_TYPE=file
export STORAGE_MODE=json

echo "Environment Variables:"
echo "  DEPLOYMENT_MODE=$DEPLOYMENT_MODE"
echo "  LIGHTRAG_STORAGE_TYPE=$LIGHTRAG_STORAGE_TYPE"
echo "  STORAGE_MODE=$STORAGE_MODE"
echo ""

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "Please run: uv sync"
    exit 1
fi

echo ""
echo "Starting FastAPI server..."
echo "URL: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# 启动服务
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
