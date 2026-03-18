#!/bin/bash
# GRAG 前端启动脚本 (Linux/Mac)
# 自动设置正确的环境变量并启动Streamlit

echo "========================================"
echo " GRAG Frontend Startup"
echo "========================================"
echo ""

# 设置环境变量
export BACKEND_URL=http://localhost:8000

echo "Configuration:"
echo "  Backend URL: $BACKEND_URL"
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
echo "Starting Streamlit frontend..."
echo "URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the frontend"
echo "========================================"
echo ""

# 启动前端
streamlit run frontend/app.py --server.port 8501
