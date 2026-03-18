@echo off
REM GRAG 后端启动脚本 (Windows)
REM 自动设置正确的环境变量并启动服务

echo ========================================
echo  GRAG Backend Startup
echo ========================================
echo.

REM 设置环境变量（关键：必须设置为 file）
set DEPLOYMENT_MODE=local
set LIGHTRAG_STORAGE_TYPE=file
set STORAGE_MODE=file

echo Environment Variables:
echo   DEPLOYMENT_MODE=%DEPLOYMENT_MODE%
echo   LIGHTRAG_STORAGE_TYPE=%LIGHTRAG_STORAGE_TYPE%
echo   STORAGE_MODE=%STORAGE_MODE%
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

REM 验证配置
echo.
echo Verifying storage configuration...
uv run python scripts/verify_config.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Configuration verification failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo Starting FastAPI server...
echo URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM 启动服务
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
