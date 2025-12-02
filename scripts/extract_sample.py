"""
GitHub 仓库文档提取脚本
从 GitHub 仓库获取文档（Python、Markdown 等），保存为 JSON
支持 OpenAI Python SDK 和 Cookbook 仓库
使用 PyGithub 库进行更高效的 GitHub API 交互
"""
import json
import logging
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from tqdm import tqdm
from github import Github
from github.GithubException import (
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
    BadCredentialsException
)

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置简单日志（不依赖完整配置系统）
def setup_simple_logger():
    """设置简单日志配置，不依赖数据库等配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 设置日志
setup_simple_logger()
logger = logging.getLogger(__name__)


class GitHubDocExtractor:
    """GitHub 仓库文档提取器（使用 PyGithub）"""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        初始化提取器
        
        Args:
            github_token: GitHub Personal Access Token（可选，用于提高速率限制）
        """
        try:
            # PyGithub 会自动处理重试，默认会重试多次
            self.github = Github(github_token) if github_token else Github()
            self.github_token = github_token
            
            # 检查速率限制状态（这里可能会遇到网络问题，PyGithub 会自动重试）
            logger.info("正在连接 GitHub API...")
            self._check_rate_limit_status()
        except BadCredentialsException:
            logger.error("GitHub 认证失败，请检查 token 是否有效")
            raise
        except Exception as e:
            logger.error(f"初始化 GitHub 客户端失败: {e}")
            logger.error("如果遇到网络连接问题，请检查：")
            logger.error("1. 网络连接是否正常（可以访问 https://api.github.com）")
            logger.error("2. 是否使用了代理（可能需要配置环境变量）")
            logger.error("3. 防火墙是否阻止了连接")
            logger.error("4. 等待一段时间后重试（PyGithub 会自动重试）")
            raise
    
    def _check_rate_limit_status(self):
        """检查并显示速率限制状态"""
        try:
            rate_limit = self.github.get_rate_limit()
            core = rate_limit.core
            reset_ts = core.reset.timestamp() if hasattr(core.reset, "timestamp") else int(core.reset)
            logger.info(
                f"GitHub API 速率限制: 剩余 {core.remaining}/{core.limit} 请求, "
                f"重置时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reset_ts))}"
            )
            
            if core.remaining < 10:
                logger.warning(f"速率限制即将用完，剩余 {core.remaining} 次请求")
        except Exception as e:
            logger.warning(f"无法获取速率限制状态: {e}")
    
    def _parse_github_url(self, url: str) -> tuple:
        """
        解析 GitHub URL
        
        Args:
            url: GitHub 仓库 URL，如 https://github.com/openai/openai-python
            
        Returns:
            (owner, repo, path) 元组
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"无效的 GitHub URL: {url}")
        
        owner = path_parts[0]
        repo = path_parts[1]
        path = '/'.join(path_parts[2:]) if len(path_parts) > 2 else ''
        
        return owner, repo, path
    
    def get_repo_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        file_extensions: List[str] = None
    ) -> List[Dict]:
        """
        获取仓库内容（使用 PyGithub）
        
        Args:
            owner: 仓库所有者
            repo: 仓库名
            path: 路径（空字符串表示根目录）
            file_extensions: 要提取的文件扩展名（如 ['.py', '.md', '.mdx']）
            
        Returns:
            文件列表
        """
        if file_extensions is None:
            file_extensions = ['.py', '.md', '.mdx', '.txt']
        
        try:
            # 获取仓库对象
            repo_obj = self.github.get_repo(f"{owner}/{repo}")
            
            # 获取目录内容
            try:
                if path:
                    contents = repo_obj.get_contents(path)
                    count = len(contents) if isinstance(contents, list) else 1
                    logger.info(f"访问路径: {path}, 获取到 {count} 个项目")
                else:
                    contents = repo_obj.get_contents("")
            except UnknownObjectException:
                logger.warning(f"路径不存在: {path}")
                return []
            
            # 如果是单个文件，转换为列表
            if not isinstance(contents, list):
                contents = [contents]
            
            # 调试：显示路径下的内容类型（前10个）
            if contents:
                content_preview = [f"{c.type}:{c.name}" for c in contents[:10]]
                logger.info(f"路径 {path} 下的内容: {content_preview}")
            
            files = []
            
            for content in contents:
                if content.type == "file":
                    # 检查文件扩展名
                    if any(content.name.endswith(ext) for ext in file_extensions):
                        files.append({
                            'path': content.path,
                            'name': content.name,
                            'type': content.type,
                            'size': content.size,
                            'sha': content.sha,
                            'url': content.html_url,
                            'download_url': content.download_url
                        })
                elif content.type == "dir":
                    # 递归获取子目录内容
                    logger.info(f"递归访问目录: {content.path}")
                    sub_files = self.get_repo_contents(
                        owner, repo, content.path, file_extensions
                    )
                    files.extend(sub_files)
                    if sub_files:
                        logger.info(f"从目录 {content.path} 获取到 {len(sub_files)} 个文件")
            
            return files
            
        except RateLimitExceededException:
            # PyGithub 会自动等待，但我们可以添加日志
            rate_limit = self.github.get_rate_limit()
            logger.warning(
                f"速率限制，剩余: {rate_limit.core.remaining}, "
                f"重置时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_limit.core.reset))}"
            )
            # PyGithub 会自动重试，这里可以等待后继续
            time.sleep(60)
            return self.get_repo_contents(owner, repo, path, file_extensions)
        except UnknownObjectException:
            logger.error(f"仓库不存在: {owner}/{repo}")
            return []
        except GithubException as e:
            logger.error(f"获取仓库内容失败: {e}")
            return []
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """
        获取文件内容（使用 PyGithub，自动处理 base64 解码）
        
        Args:
            owner: 仓库所有者
            repo: 仓库名
            path: 文件路径
            
        Returns:
            文件内容（已自动解码）
        """
        try:
            repo_obj = self.github.get_repo(f"{owner}/{repo}")
            content = repo_obj.get_contents(path)
            
            # PyGithub 自动处理 base64 解码
            # decoded_content 返回 bytes，需要解码为字符串
            try:
                if content.encoding == "base64":
                    return content.decoded_content.decode('utf-8')
                else:
                    # 对于非 base64 编码的文件（如文本文件）
                    return content.decoded_content.decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                # 如果 UTF-8 解码失败，尝试其他编码
                logger.warning(f"UTF-8 解码失败，尝试其他编码: {path}")
                try:
                    return content.decoded_content.decode('latin-1', errors='ignore')
                except Exception as e:
                    logger.error(f"无法解码文件内容 ({path}): {e}")
                    return None
                
        except RateLimitExceededException:
            rate_limit = self.github.get_rate_limit()
            logger.warning(f"速率限制，等待中... 剩余: {rate_limit.core.remaining}")
            time.sleep(60)
            return self.get_file_content(owner, repo, path)
        except UnknownObjectException:
            logger.error(f"文件不存在: {path}")
            return None
        except GithubException as e:
            logger.error(f"获取文件内容失败 ({path}): {e}")
            return None
    
    def extract_repo_docs(
        self,
        repo_url: str,
        output_file: str,
        file_extensions: List[str] = None,
        max_files: Optional[int] = None,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None
    ) -> Dict:
        """
        提取仓库文档
        
        Args:
            repo_url: 仓库 URL
            output_file: 输出 JSON 文件路径
            file_extensions: 文件扩展名列表
            max_files: 最大文件数（None 表示全部）
            include_paths: 包含的路径前缀（None 表示全部）
            exclude_paths: 排除的路径前缀（如 ['tests/', '.github/']）
            
        Returns:
            提取结果字典
        """
        if file_extensions is None:
            # 默认包含 Python 源代码、Markdown 等文档文件
            file_extensions = ['.py', '.md', '.mdx', '.txt']
        if exclude_paths is None:
            # 排除构建和配置文件，但保留测试文件（包含有用的使用示例）
            exclude_paths = [
                '.github/', '.git/', '__pycache__/', 'node_modules/',
                '.vscode/', '.devcontainer/', '.inline-snapshot/', 'bin/',
                # 可选：如果测试配置文件包含太多测试框架代码，可以排除
                # 'tests/conftest.py',  # pytest 配置
            ]
        
        # 解析 URL
        owner, repo, path = self._parse_github_url(repo_url)
        logger.info(f"提取仓库文档: {owner}/{repo}")
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "source": "GitHub Repository",
            "repo_url": repo_url,
            "owner": owner,
            "repo": repo,
            "files": []
        }
        
        # 如果指定了 include_paths，优先使用它们作为起始路径
        # 否则使用 URL 中解析的 path，如果都没有则从根目录开始
        search_paths = []
        if include_paths:
            # 使用 include_paths 作为起始路径
            search_paths = include_paths
            logger.info(f"从指定路径开始搜索: {search_paths}")
        elif path:
            # 使用 URL 中的路径
            search_paths = [path]
        else:
            # 从根目录开始
            search_paths = [""]
        
        # 从多个路径获取文件（如果指定了多个 include_paths）
        all_files = []
        for search_path in search_paths:
            files = self.get_repo_contents(owner, repo, search_path, file_extensions)
            all_files.extend(files)
        
        logger.info(f"找到 {len(all_files)} 个文档文件")
        
        # 过滤文件
        filtered_files = []
        
        # 调试：显示过滤条件
        if exclude_paths:
            logger.info(f"排除路径过滤条件: {exclude_paths}")
        
        # 调试：显示前几个文件路径示例
        if all_files:
            logger.info(f"前 5 个文件路径示例:")
            for file_info in all_files[:5]:
                logger.info(f"  - {file_info.get('path', '')}")
        
        for file_info in all_files:
            file_path = file_info.get('path', '')
            
            # 检查排除路径
            if any(file_path.startswith(exclude) for exclude in exclude_paths):
                continue
            
            # 注意：include_paths 已经作为起始路径使用，不需要再次过滤
            # 但如果用户同时指定了 URL 路径和 include_paths，这里可以进一步细化过滤
            # 目前简化处理：如果 include_paths 已使用，就不再过滤
            
            filtered_files.append(file_info)
            
            if max_files and len(filtered_files) >= max_files:
                break
        
        logger.info(f"过滤后剩余 {len(filtered_files)} 个文件")
        
        # 提取文件内容（使用进度条）
        with tqdm(total=len(filtered_files), desc="获取文件内容", unit="文件", ncols=100) as pbar:
            for file_info in filtered_files:
                file_path = file_info['path']
                # 显示当前处理的文件名（截断过长的路径）
                display_name = Path(file_path).name
                if len(display_name) > 40:
                    display_name = display_name[:37] + "..."
                pbar.set_description(f"处理: {display_name}")
                
                content = self.get_file_content(owner, repo, file_path)
                
                if content:
                    results["files"].append({
                        "path": file_path,
                        "name": file_info['name'],
                        "url": file_info.get('html_url', ''),
                        "size": file_info.get('size', 0),
                        "content": content,
                        "content_length": len(content)
                    })
                
                pbar.update(1)
                # PyGithub 会自动处理速率限制，但可以添加小延迟避免过于频繁
                time.sleep(0.1)
        
        # 保存结果
        logger.info(f"正在保存结果到: {output_file}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        total_files = len(results["files"])
        total_content = sum(f["content_length"] for f in results["files"])
        
        logger.info("=" * 60)
        logger.info("文档提取完成！")
        logger.info(f"总文件数: {total_files}")
        logger.info(f"总内容长度: {total_content:,} 字符")
        logger.info(f"输出文件: {output_file}")
        logger.info("=" * 60)
        
        return results


def extract_github_repos(
    repos: List[str],
    output_dir: str = "samples",
    max_files_per_repo: Optional[int] = None
):
    """
    批量提取多个 GitHub 仓库的文档
    
    Args:
        repos: 仓库 URL 列表
        output_dir: 输出目录
        max_files_per_repo: 每个仓库的最大文件数
    """
    extractor = GitHubDocExtractor()
    
    for repo_url in repos:
        # 生成输出文件名
        owner, repo, _ = extractor._parse_github_url(repo_url)
        output_file = f"{output_dir}/{owner}_{repo}_docs.json"
        
        # 提取文档
        extractor.extract_repo_docs(
            repo_url=repo_url,
            output_file=output_file,
            max_files=max_files_per_repo
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从 GitHub 仓库提取文档")
    parser.add_argument(
        "--repo",
        default="https://github.com/openai/openai-python",
        help="GitHub 仓库 URL"
    )
    parser.add_argument(
        "--output",
        default="samples/github_docs_sample.json",
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="最大文件数（默认 10，用于测试）"
    )
    parser.add_argument(
        "--github-token",
        default=GITHUB_TOKEN,
        help="GitHub Personal Access Token（默认读取环境变量 GITHUB_TOKEN）"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="包含的路径前缀（如 examples/ docs/）"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[
            ".github/", ".git/", "__pycache__/", "node_modules/",
            ".vscode/", ".devcontainer/", ".inline-snapshot/", "bin/"
        ],
        help="排除的路径前缀（默认保留测试文件，因为它们包含有用的使用示例）"
    )
    
    args = parser.parse_args()
    
    try:
        extractor = GitHubDocExtractor(github_token=args.github_token)
        extractor.extract_repo_docs(
            repo_url=args.repo,
            output_file=args.output,
            max_files=args.max_files,
            include_paths=args.include,
            exclude_paths=args.exclude
        )
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"提取失败: {e}", exc_info=True)
        sys.exit(1)
