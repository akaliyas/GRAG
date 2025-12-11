"""
GitHub 文档提取工具
基于 GitHub API 提取仓库文档，支持 Markdown 和 Jupyter Notebook
完全去爬虫化，仅使用结构化数据源（Source Code is Truth）
"""
import logging
import os
import re
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse

try:
    import nbformat
    from nbformat import NotebookNode
except ImportError:
    nbformat = None
    NotebookNode = None

from dotenv import load_dotenv
from github import Github
from github.GithubException import (
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
    BadCredentialsException
)

from utils.schema import IngestionBatch, RawDoc

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class GitHubIngestor:
    """
    GitHub 文档提取工具
    
    功能：
    - 从 GitHub 仓库提取 Markdown 和 Jupyter Notebook 文件
    - 清洗 .ipynb 文件（仅保留 Markdown 和 Code 输入，丢弃 Output）
    - 提取 .md 文件的 Frontmatter
    - 修复相对链接为 GitHub Raw URL
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        初始化 GitHub 提取工具
        
        Args:
            github_token: GitHub Personal Access Token（从环境变量读取，可选）
        """
        token = github_token or os.getenv("GITHUB_TOKEN")
        if not token:
            logger.warning("GITHUB_TOKEN 未设置，将使用匿名访问（速率限制较低）")
        
        try:
            self.github = Github(token) if token else Github()
            self.github_token = token
            logger.info("GitHub 提取工具已初始化")
        except BadCredentialsException:
            logger.error("GitHub 认证失败，请检查 token 是否有效")
            raise
        except Exception as e:
            logger.error(f"初始化 GitHub 客户端失败: {e}")
            raise
    
    def _parse_github_url(self, url: str) -> Tuple[str, str, str]:
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
    
    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, str], str]:
        """
        提取 Markdown 文件的 Frontmatter（YAML 格式）
        
        Args:
            content: Markdown 文件内容
            
        Returns:
            (frontmatter_dict, body_content) 元组
        """
        frontmatter = {}
        body = content
        
        # 检查是否有 Frontmatter（以 --- 开头）
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                body = parts[2].strip()
                
                # 简单解析 YAML（仅支持 key: value 格式）
                for line in frontmatter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
        
        return frontmatter, body
    
    def _clean_notebook(self, notebook_content: str, repo_url: str) -> str:
        """
        清洗 Jupyter Notebook 文件
        
        仅保留：
        - Markdown 单元格
        - Code 单元格的输入部分
        
        丢弃：
        - Code 单元格的输出（包含 Base64 图片、错误信息等噪音）
        - HTML 标签（特别是 tfo-notebook-buttons 等导航元素）
        - 其他元数据
        
        Args:
            notebook_content: Notebook JSON 字符串
            repo_url: 仓库 URL（用于修复链接）
            
        Returns:
            清洗后的纯文本内容
        """
        if nbformat is None:
            logger.warning("nbformat 未安装，无法处理 .ipynb 文件")
            return notebook_content
        
        try:
            notebook = nbformat.reads(notebook_content, as_version=4)
            # 规范化 Notebook（添加缺失的 id 字段，消除警告）
            # nbformat 5.1.4+ 支持 normalize，旧版本会忽略
            try:
                # 尝试使用 normalize 方法（nbformat 5.1.4+）
                if hasattr(nbformat, 'normalize'):
                    nbformat.normalize(notebook)
                else:
                    # 旧版本手动添加 id 字段
                    import uuid
                    for cell in notebook.cells:
                        if not hasattr(cell, 'id') or not cell.id:
                            cell.id = str(uuid.uuid4())
            except (AttributeError, TypeError, ImportError) as e:
                # 如果 normalize 不可用或出错，记录警告，不影响功能
                logger.warning(f"nbformat.normalize 不可用或处理 notebook 规范化出错: {e}")
        except Exception as e:
            logger.error(f"解析 Notebook 失败: {e}")
            return notebook_content
        
        cleaned_parts = []
        
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                # 保留 Markdown 单元格
                cell_content = cell.source
                # 修复相对链接
                cell_content = self._fix_relative_links(cell_content, repo_url)
                # 清理 HTML 标签
                cell_content = self._clean_html_tags(cell_content)
                cleaned_parts.append(cell_content)
            elif cell.cell_type == 'code':
                # 仅保留代码输入，丢弃输出
                code_input = cell.source
                # 添加代码块标记
                cleaned_parts.append(f"```python\n{code_input}\n```")
        
        cleaned_content = '\n\n'.join(cleaned_parts)
        
        # 最终清理：移除残留的 HTML 标签
        cleaned_content = self._clean_html_tags(cleaned_content)
        
        return cleaned_content
    
    def _clean_html_tags(self, content: str) -> str:
        """
        清理 HTML 标签，保留文本内容
        
        策略：
        1. 完全移除 tfo-notebook-buttons 表格（Google Colab 导航按钮）
        2. 移除其他 HTML 标签，保留文本内容
        3. 保留 Markdown 代码块（避免误删）
        
        Args:
            content: 包含 HTML 的文本内容
            
        Returns:
            清理后的纯文本内容
        """
        # 步骤 1: 移除 tfo-notebook-buttons 表格（完全移除，包括内容）
        # 使用非贪婪匹配，匹配整个表格
        tfo_table_pattern = r'<table[^>]*class=["\']tfo-notebook-buttons["\'][^>]*>.*?</table>'
        content = re.sub(tfo_table_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # 步骤 2: 保护 Markdown 代码块（避免误删代码块内的 HTML）
        # 临时替换代码块为占位符
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'
        
        def replace_code_block(match):
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append(match.group(0))
            return placeholder
        
        content = re.sub(code_block_pattern, replace_code_block, content)
        
        # 步骤 3: 移除所有 HTML 标签（保留文本内容）
        # 匹配 <tag> 或 <tag attr="..."> 格式
        html_tag_pattern = r'<[^>]+>'
        content = re.sub(html_tag_pattern, '', content)
        
        # 步骤 4: 恢复代码块
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", code_block)
        
        # 步骤 5: 清理多余的空白行（HTML 标签移除后可能产生）
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 步骤 6: 清理行首行尾空白
        content = content.strip()
        
        return content
    
    def _fix_relative_links(self, content: str, repo_url: str) -> str:
        """
        修复 Markdown 中的相对链接为 GitHub Raw URL
        
        Args:
            content: Markdown 内容
            repo_url: 仓库 URL
            
        Returns:
            修复后的内容
        """
        # 解析仓库信息
        owner, repo, _ = self._parse_github_url(repo_url)
        base_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
        
        # 匹配相对链接模式 [text](../path/to/file)
        def replace_link(match):
            link_text = match.group(1)
            relative_path = match.group(2)
            # 简化处理：将相对路径转换为绝对路径
            absolute_path = relative_path.lstrip('./')
            return f"[{link_text}]({base_raw_url}/{absolute_path})"
        
        # 替换相对链接
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        content = re.sub(pattern, replace_link, content)
        
        return content
    
    def fetch_repo_structure(
        self,
        repo_url: str,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
        current_path: str = ""
    ) -> List[Dict[str, str]]:
        """
        EDA 第一步：获取仓库结构（不下载内容）
        
        Args:
            repo_url: GitHub 仓库 URL
            include_paths: 包含的路径前缀列表
            exclude_paths: 排除的路径前缀列表
            file_extensions: 文件扩展名列表（默认 ['.md', '.ipynb']）
            
        Returns:
            文件元数据列表，包含 path, name, size, url 等
        """
        if file_extensions is None:
            file_extensions = ['.md', '.mdx', '.ipynb', '.txt']
        
        if exclude_paths is None:
            exclude_paths = [
                '.github/', '.git/', '__pycache__/', 'node_modules/',
                '.vscode/', '.devcontainer/', '.inline-snapshot/', 'bin/',
                '_build/', 'legacy/', 'translations/'
            ]
        
        owner, repo, url_path = self._parse_github_url(repo_url)
        
        # 使用 current_path 作为搜索路径（递归时使用）
        # 如果是首次调用且 URL 中包含路径，使用 URL 中的路径
        if current_path == "" and url_path:
            search_path = url_path
        else:
            search_path = current_path
        
        try:
            repo_obj = self.github.get_repo(f"{owner}/{repo}")
            
            # try to get tqdm
            if current_path == "":
                logger.info(f"开始扫描仓库结构：{owner}/{repo}")
                start_time = time.time()
            # 获取目录内容
            try:
                contents = repo_obj.get_contents(search_path)
            except UnknownObjectException:
                logger.warning(f"路径不存在: {search_path}")
                return []
            
            if not isinstance(contents, list):
                contents = [contents]
            
            files = []
            
            # try to get tqdm
            contents_iter = tqdm(contents, desc="扫描仓库文件",unit="项",disable=(current_path != ""))if current_path == "" else contents

            for content in contents_iter:
                if content.type == "file":
                    file_path = content.path
                    
                    # 检查文件扩展名
                    if not any(file_path.endswith(ext) for ext in file_extensions):
                        continue
                    
                    # 检查排除路径
                    if any(file_path.startswith(exclude) for exclude in exclude_paths):
                        continue
                    
                    # 检查包含路径
                    if include_paths and not any(
                        file_path.startswith(include) for include in include_paths
                    ):
                        continue
                    
                    # 过滤文件大小（1KB - 100KB）
                    if content.size < 1024 or content.size > 102400:
                        logger.debug(f"文件 {file_path} 大小 {content.size} 超出范围，跳过")
                        continue
                    
                    files.append({
                        'path': file_path,
                        'name': content.name,
                        'size': content.size,
                        'sha': content.sha,
                        'url': content.html_url,
                        'download_url': content.download_url
                    })
                elif content.type == "dir":
                    # 递归获取子目录
                    # 检查是否需要递归（如果设置了 include_paths，只递归匹配的目录）
                    should_recurse = True
                    if include_paths:
                        # 检查子目录路径是否匹配任何 include_paths
                        should_recurse = any(
                            content.path.startswith(inc) or inc.startswith(content.path)
                            for inc in include_paths
                        )
                    
                    if should_recurse:
                        sub_files = self.fetch_repo_structure(
                            repo_url,
                            include_paths,
                            exclude_paths,
                            file_extensions,
                            current_path=content.path  # 使用子目录路径作为新的搜索路径
                        )
                        files.extend(sub_files)
            
            return files
            
        except GithubException as e:
            logger.error(f"获取仓库结构失败: {e}")
            return []
    
    def _download_raw_content(
        self,
        repo_url: str,
        file_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        下载原始文件内容（不做任何清洗）
        
        Args:
            repo_url: GitHub 仓库 URL
            file_paths: 文件路径列表
            
        Returns:
            原始文档列表，每个文档包含：path, content, source_url, file_type, metadata
        """
        owner, repo, _ = self._parse_github_url(repo_url)
        documents = []
        
        for file_path in file_paths:
            try:
                # 获取文件内容
                repo_obj = self.github.get_repo(f"{owner}/{repo}")
                content_obj = repo_obj.get_contents(file_path)
                
                # 解码内容（原始，不做清洗）
                if content_obj.encoding == "base64":
                    raw_content = content_obj.decoded_content.decode('utf-8')
                else:
                    raw_content = content_obj.decoded_content.decode('utf-8', errors='ignore')
                
                # 确定文件类型
                if file_path.endswith('.ipynb'):
                    file_type = 'notebook'
                elif file_path.endswith(('.md', '.mdx')):
                    file_type = 'markdown'
                else:
                    file_type = 'text'
                
                # 提取 Frontmatter（仅提取，不修改内容）
                frontmatter = {}
                if file_type == 'markdown':
                    frontmatter, _ = self._extract_frontmatter(raw_content)
                
                # 构建 GitHub Raw URL
                source_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
                
                documents.append({
                    'path': file_path,
                    'content': raw_content,  # 原始内容，不做清洗
                    'source_url': source_url,
                    'file_type': file_type,
                    'metadata': {
                        'type': file_type,
                        'url': content_obj.html_url,
                        'frontmatter': frontmatter
                    }
                })
                
                logger.debug(f"已下载原始文件: {file_path}")
                
            except Exception as e:
                logger.error(f"下载文件失败 ({file_path}): {e}")
                continue
        
        logger.info(f"成功下载 {len(documents)} 个原始文件")
        return documents
    
    def extract_repo_docs(
        self,
        repo_url: str,
        file_extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None
    ) -> IngestionBatch:
        """
        从 GitHub 仓库提取文档，返回 IngestionBatch 对象（Raw Artifact）
        
        注意：此方法返回原始内容，不做任何清洗。清洗应在后续步骤完成。
        
        Args:
            repo_url: GitHub 仓库 URL
            file_extensions: 文件扩展名列表（默认 ['.md', '.ipynb']）
            max_files: 最大文件数（可选，默认无限制）
            include_paths: 包含的路径前缀列表
            exclude_paths: 排除的路径前缀列表
            
        Returns:
            IngestionBatch 对象（Raw Artifact）
        """
        if file_extensions is None:
            file_extensions = ['.md', '.mdx', '.ipynb']
        
        # 获取仓库结构
        file_metadata = self.fetch_repo_structure(
            repo_url,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions
        )
        
        # 限制文件数量
        if max_files:
            file_metadata = file_metadata[:max_files]
        
        # 下载原始内容（不做清洗）
        file_paths = [f['path'] for f in file_metadata]
        raw_documents = self._download_raw_content(repo_url, file_paths)
        
        # 转换为 RawDoc 对象
        raw_docs = []
        for doc in raw_documents:
            raw_doc = RawDoc(
                path=doc['path'],
                content=doc['content'],  # 原始内容
                source_url=doc['source_url'],
                file_type=doc['file_type'],
                metadata=doc['metadata']
            )
            raw_docs.append(raw_doc)
        
        # 生成 batch_id（基于 repo_url 和提取时间）
        import hashlib
        from datetime import datetime
        batch_id = hashlib.md5(f"{repo_url}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # 创建 IngestionBatch
        batch = IngestionBatch(
            batch_id=batch_id,
            repo_url=repo_url,
            docs=raw_docs
        )
        
        logger.info(f"成功提取 {len(raw_docs)} 个文档，返回 IngestionBatch 对象")
        return batch
    
    def download_and_clean(
        self,
        repo_url: str,
        file_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        EDA 第三步：下载并清洗文件为 LightRAG 可用的文本
        
        Args:
            repo_url: GitHub 仓库 URL
            file_paths: 要下载的文件路径列表（如果为 None，则自动发现）
            include_paths: 包含的路径前缀列表
            exclude_paths: 排除的路径前缀列表
            
        Returns:
            清洗后的文档列表，每个文档包含：
            - content: 清洗后的文本内容
            - path: 文件路径
            - metadata: 元数据（Frontmatter、文件信息等）
        """
        owner, repo, _ = self._parse_github_url(repo_url)
        
        # 如果没有指定文件路径，先获取仓库结构
        if file_paths is None:
            file_metadata = self.fetch_repo_structure(
                repo_url,
                include_paths=include_paths,
                exclude_paths=exclude_paths
            )
            file_paths = [f['path'] for f in file_metadata]
        
        documents = []
        
        for file_path in file_paths:
            try:
                # 获取文件内容
                repo_obj = self.github.get_repo(f"{owner}/{repo}")
                content_obj = repo_obj.get_contents(file_path)
                
                # 解码内容
                if content_obj.encoding == "base64":
                    raw_content = content_obj.decoded_content.decode('utf-8')
                else:
                    raw_content = content_obj.decoded_content.decode('utf-8', errors='ignore')
                
                # 根据文件类型清洗
                if file_path.endswith('.ipynb'):
                    cleaned_content = self._clean_notebook(raw_content, repo_url)
                    metadata = {
                        'type': 'notebook',
                        'path': file_path,
                        'url': content_obj.html_url
                    }
                elif file_path.endswith(('.md', '.mdx')):
                    frontmatter, body = self._extract_frontmatter(raw_content)
                    cleaned_content = self._fix_relative_links(body, repo_url)
                    metadata = {
                        'type': 'markdown',
                        'path': file_path,
                        'url': content_obj.html_url,
                        'frontmatter': frontmatter
                    }
                else:
                    # 其他文本文件直接使用
                    cleaned_content = raw_content
                    metadata = {
                        'type': 'text',
                        'path': file_path,
                        'url': content_obj.html_url
                    }
                
                documents.append({
                    'content': cleaned_content,
                    'path': file_path,
                    'metadata': metadata
                })
                
                logger.debug(f"已处理文件: {file_path}")
                
            except Exception as e:
                logger.error(f"处理文件失败 ({file_path}): {e}")
                continue
        
        logger.info(f"成功处理 {len(documents)} 个文件")
        return documents

if __name__ == "__main__":
    """
    测试 GitHub 提取工具
    
    测试场景：
    1. 获取仓库结构（不下载内容）
    2. 下载并清洗文档
    3. 测试不同文件类型（.md, .ipynb）
    4. 测试路径过滤
    """
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    ingestor = GitHubIngestor()
    
    # 测试仓库 URL（可以根据需要修改）
    test_repo = "https://github.com/google/generative-ai-docs"
    test_path = "site/en/gemini-api"
    
    print("=" * 60)
    print("测试 1: 获取仓库结构（不下载内容）")
    print("=" * 60)
    
    try:
        files = ingestor.fetch_repo_structure(
            test_repo,
            include_paths=[test_path],
            file_extensions=['.md', '.ipynb']
        )
        print(f"✅ 找到 {len(files)} 个文件")
        print(f"\n前 5 个文件示例:")
        for f in files[:5]:
            print(f"  - {f['path']} ({f['size']} bytes)")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("测试 2: 下载并清洗文档（限制数量）")
    print("=" * 60)
    
    if not files:
        print("⚠️  没有找到文件，跳过下载测试")
        print("=" * 60)
        print("✅ 测试完成（部分跳过）")
        print("=" * 60)
        sys.exit(0)
    
    try:
        # 只处理前 3 个文件（快速测试）
        file_paths = [f['path'] for f in files[:3]]
        print(f"准备下载 {len(file_paths)} 个文件: {file_paths}")
        
        documents = ingestor.download_and_clean(
            test_repo,
            file_paths=file_paths
        )
        
        print(f"✅ 成功处理 {len(documents)} 个文档\n")
        
        # 统计信息
        total_chars = 0
        type_counts = {}
        
        for i, doc in enumerate(documents, 1):
            doc_type = doc['metadata']['type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            total_chars += len(doc['content'])
            
            print(f"文档 {i}: {doc['path']}")
            print(f"  类型: {doc_type}")
            print(f"  内容长度: {len(doc['content']):,} 字符")
            print(f"  GitHub URL: {doc['metadata']['url']}")
            
            if doc_type == 'markdown' and 'frontmatter' in doc['metadata']:
                frontmatter = doc['metadata']['frontmatter']
                if frontmatter:
                    print(f"  Frontmatter: {frontmatter}")
            
            # 显示内容预览（前 150 字符）
            content_preview = doc['content'][:150].replace('\n', ' ').strip()
            print(f"  内容预览: {content_preview}...")
            print()
        
        # 打印统计信息
        print("=" * 60)
        print("📊 处理结果统计")
        print("=" * 60)
        print(f"总文档数: {len(documents)}")
        print(f"总字符数: {total_chars:,}")
        print(f"文件类型分布:")
        for doc_type, count in type_counts.items():
            print(f"  - {doc_type}: {count} 个")
        print()
        
        # 说明数据格式和位置
        print("=" * 60)
        print("📦 数据格式说明")
        print("=" * 60)
        print("处理后的数据格式：")
        print("  - 数据类型: Python List[Dict]")
        print("  - 每个文档包含:")
        print("    • content: str - 清洗后的文本内容（纯文本）")
        print("    • path: str - 文件在仓库中的路径")
        print("    • metadata: Dict - 元数据（类型、URL、Frontmatter 等）")
        print()
        print("当前状态: 数据仅存在于内存中，未保存到文件")
        print()
        
        # 保存到 JSON 文件（可选）
        import json
        from pathlib import Path
        
        output_dir = Path("samples")
        output_dir.mkdir(exist_ok=True)
        
        # 生成输出文件名
        owner, repo, _ = ingestor._parse_github_url(test_repo)
        output_file = output_dir / f"{owner}_{repo}_cleaned_docs.json"
        
        # 保存数据
        output_data = {
            "source": "GitHub Repository",
            "repo_url": test_repo,
            "extracted_at": __import__('datetime').datetime.now().isoformat(),
            "total_documents": len(documents),
            "total_characters": total_chars,
            "type_distribution": type_counts,
            "documents": documents
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 数据已保存到: {output_file}")
        print(f"   文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        print()
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print()
    print("📋 下一步操作建议：")
    print("=" * 60)
    print("1. 【集成到 LightRAG】")
    print("   将清洗后的文档添加到知识库：")
    print("   ```python")
    print("   from knowledge.lightrag_wrapper import LightRAGWrapper")
    print("   from models.model_manager import ModelManager")
    print("   ")
    print("   model_manager = ModelManager()")
    print("   lightrag = LightRAGWrapper(model_manager)")
    print("   ")
    print("   # 提取文档内容列表")
    print("   doc_contents = [doc['content'] for doc in documents]")
    print("   ")
    print("   # 添加到知识库")
    print("   lightrag.add_documents(doc_contents)")
    print("   ```")
    print()
    print("2. 【通过 Agent 调用】")
    print("   在 Agent 工作流中使用 GitHubIngestor：")
    print("   ```python")
    print("   from agent.tools.github_ingestor import GitHubIngestor")
    print("   ")
    print("   ingestor = GitHubIngestor()")
    print("   documents = ingestor.download_and_clean(repo_url)")
    print("   # Agent 会自动调用 lightrag.add_documents()")
    print("   ```")
    print()
    print("3. 【批量处理多个仓库】")
    print("   可以循环处理多个 GitHub 仓库，统一添加到知识库")
    print()
    print("4. 【验证数据质量】")
    print("   检查保存的 JSON 文件，确认清洗效果是否符合预期")
    print("=" * 60)
