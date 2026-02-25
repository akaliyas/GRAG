FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖和 uv
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# 设置 uv 的 PATH
ENV PATH="/root/.local/bin:${PATH}"

# 复制依赖文件（先复制这些以利用Docker层缓存）
COPY uv.lock pyproject.toml ./

# 使用 uv sync 安装项目依赖
RUN uv sync --frozen

# 安装 Playwright 浏览器（如果需要）
RUN uv run playwright install chromium
RUN uv run playwright install-deps chromium

# 复制项目文件
COPY . .

# 创建日志目录
RUN mkdir -p logs

# 暴露端口
EXPOSE 8000

# 启动命令（使用 uv run 执行）
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
