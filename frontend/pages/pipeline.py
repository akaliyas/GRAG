"""
数据管道管理页面
支持 Fetch、Clean、Ingest 三个步骤的可视化管理和执行
"""
import streamlit as st
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def init_session_state():
    """初始化会话状态"""
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = {
            "fetch": {"status": "idle", "last_run": None, "output": None},
            "clean": {"status": "idle", "last_run": None, "output": None},
            "ingest": {"status": "idle", "last_run": None, "output": None}
        }
    if "pipeline_logs" not in st.session_state:
        st.session_state.pipeline_logs = []


def run_pipeline_script(script_name: str, args: list) -> tuple[bool, str]:
    """运行管道脚本"""
    try:
        script_path = PROJECT_ROOT / "scripts" / script_name

        if not script_path.exists():
            return False, f"脚本不存在: {script_path}"

        # 使用 uv run 执行脚本
        cmd = ["uv", "run", str(script_path)] + args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=str(PROJECT_ROOT)
        )

        success = result.returncode == 0
        output = result.stdout if result.stdout else result.stderr
        return success, output

    except subprocess.TimeoutExpired:
        return False, "脚本执行超时（10分钟）"
    except Exception as e:
        return False, f"执行失败: {str(e)}"


def get_artifact_files(stage: str) -> list:
    """获取指定阶段的文件列表"""
    artifacts_dir = PROJECT_ROOT / "artifacts" / stage

    if not artifacts_dir.exists():
        return []

    files = list(artifacts_dir.glob("*.json"))
    return [
        {
            "name": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
        for f in files
    ]


def show():
    """显示管道管理页面"""
    init_session_state()

    st.subheader("🔄 数据管道管理")

    # 三列布局显示管道步骤
    col1, col2, col3 = st.columns(3)

    # Step 1: Fetch
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📥 Step 1: Fetch</h3>
            <p>从 GitHub 获取文档数据</p>
        </div>
        """, unsafe_allow_html=True)

        fetch_status = st.session_state.pipeline_status["fetch"]["status"]

        # 状态指示器
        if fetch_status == "idle":
            st.markdown('<p class="status-success">● 待机</p>', unsafe_allow_html=True)
        elif fetch_status == "running":
            st.markdown('<p class="status-warning">● 运行中</p>', unsafe_allow_html=True)
        elif fetch_status == "success":
            st.markdown('<p class="status-success">● 完成</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">● 失败</p>', unsafe_allow_html=True)

        st.markdown("---")

        # 配置表单
        st.markdown("**配置参数**")
        repo_url = st.text_input(
            "GitHub 仓库 URL",
            placeholder="https://github.com/owner/repo",
            key="fetch_repo"
        )

        output_file = st.text_input(
            "输出文件",
            value="artifacts/01_raw/data.json",
            key="fetch_output"
        )

        # 执行按钮
        if st.button("▶️ 开始 Fetch", use_container_width=True, type="primary"):
            if not repo_url:
                st.error("请输入 GitHub 仓库 URL")
            else:
                st.session_state.pipeline_status["fetch"]["status"] = "running"
                st.rerun()

                success, output = run_pipeline_script(
                    "pipeline_fetch.py",
                    ["--repo", repo_url, "--output", output_file]
                )

                st.session_state.pipeline_status["fetch"].update({
                    "status": "success" if success else "error",
                    "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "output": output
                })
                st.rerun()

        # 显示输出
        if st.session_state.pipeline_status["fetch"]["output"]:
            with st.expander("📋 查看输出"):
                st.text(st.session_state.pipeline_status["fetch"]["output"])

    # Step 2: Clean
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧹 Step 2: Clean</h3>
            <p>清洗和预处理数据</p>
        </div>
        """, unsafe_allow_html=True)

        clean_status = st.session_state.pipeline_status["clean"]["status"]

        # 状态指示器
        if clean_status == "idle":
            st.markdown('<p class="status-success">● 待机</p>', unsafe_allow_html=True)
        elif clean_status == "running":
            st.markdown('<p class="status-warning">● 运行中</p>', unsafe_allow_html=True)
        elif clean_status == "success":
            st.markdown('<p class="status-success">● 完成</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">● 失败</p>', unsafe_allow_html=True)

        st.markdown("---")

        # 配置表单
        st.markdown("**配置参数**")
        input_file = st.text_input(
            "输入文件 (Raw)",
            value="artifacts/01_raw/data.json",
            key="clean_input"
        )

        output_file = st.text_input(
            "输出文件 (Clean)",
            value="artifacts/02_clean/data.json",
            key="clean_output"
        )

        # 执行按钮
        if st.button("▶️ 开始 Clean", use_container_width=True, type="primary"):
            # 检查输入文件是否存在
            input_path = PROJECT_ROOT / input_file
            if not input_path.exists():
                st.error(f"输入文件不存在: {input_file}")
            else:
                st.session_state.pipeline_status["clean"]["status"] = "running"
                st.rerun()

                success, output = run_pipeline_script(
                    "pipeline_clean.py",
                    ["--input", input_file, "--output", output_file]
                )

                st.session_state.pipeline_status["clean"].update({
                    "status": "success" if success else "error",
                    "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "output": output
                })
                st.rerun()

        # 显示输出
        if st.session_state.pipeline_status["clean"]["output"]:
            with st.expander("📋 查看输出"):
                st.text(st.session_state.pipeline_status["clean"]["output"])

    # Step 3: Ingest
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Step 3: Ingest</h3>
            <p>导入到 LightRAG 知识库</p>
        </div>
        """, unsafe_allow_html=True)

        ingest_status = st.session_state.pipeline_status["ingest"]["status"]

        # 状态指示器
        if ingest_status == "idle":
            st.markdown('<p class="status-success">● 待机</p>', unsafe_allow_html=True)
        elif ingest_status == "running":
            st.markdown('<p class="status-warning">● 运行中</p>', unsafe_allow_html=True)
        elif ingest_status == "success":
            st.markdown('<p class="status-success">● 完成</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">● 失败</p>', unsafe_allow_html=True)

        st.markdown("---")

        # 配置表单
        st.markdown("**配置参数**")
        input_file = st.text_input(
            "输入文件 (Clean)",
            value="artifacts/02_clean/data.json",
            key="ingest_input"
        )

        # 执行按钮
        if st.button("▶️ 开始 Ingest", use_container_width=True, type="primary"):
            # 检查输入文件是否存在
            input_path = PROJECT_ROOT / input_file
            if not input_path.exists():
                st.error(f"输入文件不存在: {input_file}")
            else:
                st.session_state.pipeline_status["ingest"]["status"] = "running"
                st.rerun()

                success, output = run_pipeline_script(
                    "pipeline_ingest.py",
                    ["--input", input_file]
                )

                st.session_state.pipeline_status["ingest"].update({
                    "status": "success" if success else "error",
                    "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "output": output
                })
                st.rerun()

        # 显示输出
        if st.session_state.pipeline_status["ingest"]["output"]:
            with st.expander("📋 查看输出"):
                st.text(st.session_state.pipeline_status["ingest"]["output"])

    # 底部：文件浏览器和快速操作
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📁 Artifacts 文件浏览器")

        stage = st.selectbox(
            "选择阶段",
            ["01_raw", "02_clean"],
            label_visibility="collapsed"
        )

        files = get_artifact_files(stage)

        if files:
            st.write(f"找到 {len(files)} 个文件")

            for file_info in files:
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**大小**: {file_info['size'] / 1024:.2f} KB")
                    st.write(f"**修改时间**: {file_info['modified']}")
                    st.write(f"**路径**: `{file_info['path']}`")
        else:
            st.info(f"未找到 {stage} 阶段的文件")

    with col_right:
        st.subheader("⚡ 快速操作")

        # 一键执行全部
        st.markdown("**🚀 一键执行**")

        repo_url = st.text_input(
            "GitHub 仓库 URL",
            placeholder="https://github.com/owner/repo",
            key="run_all_repo"
        )

        if st.button("执行完整管道 (Fetch → Clean → Ingest)", use_container_width=True):
            if not repo_url:
                st.error("请输入 GitHub 仓库 URL")
            else:
                st.info("开始执行完整管道...")

                # Step 1: Fetch
                st.info("📥 执行 Step 1: Fetch...")
                success1, output1 = run_pipeline_script(
                    "pipeline_fetch.py",
                    ["--repo", repo_url, "--output", "artifacts/01_raw/data.json"]
                )

                if success1:
                    st.success("✅ Fetch 完成")

                    # Step 2: Clean
                    st.info("🧹 执行 Step 2: Clean...")
                    success2, output2 = run_pipeline_script(
                        "pipeline_clean.py",
                        ["--input", "artifacts/01_raw/data.json", "--output", "artifacts/02_clean/data.json"]
                    )

                    if success2:
                        st.success("✅ Clean 完成")

                        # Step 3: Ingest
                        st.info("📊 执行 Step 3: Ingest...")
                        success3, output3 = run_pipeline_script(
                            "pipeline_ingest.py",
                            ["--input", "artifacts/02_clean/data.json"]
                        )

                        if success3:
                            st.success("✅ Ingest 完成")
                            st.success("🎉 管道执行完成！")
                        else:
                            st.error(f"❌ Ingest 失败: {output3}")
                    else:
                        st.error(f"❌ Clean 失败: {output2}")
                else:
                    st.error(f"❌ Fetch 失败: {output1}")

        st.markdown("---")

        # 清空状态
        st.markdown("**🔄 重置状态**")

        if st.button("重置所有状态", use_container_width=True, type="secondary"):
            st.session_state.pipeline_status = {
                "fetch": {"status": "idle", "last_run": None, "output": None},
                "clean": {"status": "idle", "last_run": None, "output": None},
                "ingest": {"status": "idle", "last_run": None, "output": None}
            }
            st.success("状态已重置")
            st.rerun()


# Streamlit 多页面应用会自动执行此文件
show()
