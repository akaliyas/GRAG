@echo off
REM GRAG 前端启动脚本 (Windows)
REM 自动设置正确的环境变量并启动Streamlit

echo ========================================
echo  GRAG Frontend Startup
echo ========================================
echo.

REM 设置环境变量
set BACKEND_URL=http://localhost:8000

echo Configuration:
echo   Backend URL: %BACKEND_URL%
echo.

REM 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Please run: uv sync
    pause
    exit /b 1
)

echo.
echo Starting Streamlit frontend...
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the frontend
echo ========================================
echo.

REM 启动前端
streamlit run frontend/app.py --server.port 8501
