@echo off
REM OpenAI Cookbook数据摄取脚本 - JSON模式
setlocal enabledelayedexpansion

REM 设置环境变量强制使用JSON存储
set STORAGE_MODE=json
set LIGHTRAG_STORAGE_TYPE=file
set LIGHTRAG_GRAPH_STORAGE=NetworkXStorage

REM 运行摄取
echo ============================================================
echo OpenAI Cookbook Data Ingestion
echo Storage mode: JSON (no external database)
echo ============================================================
echo.

uv run scripts/pipeline_ingest.py --input artifacts/02_clean/openai_cookbook_clean.json

endlocal
