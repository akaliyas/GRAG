"""
引用处理工具模块

提供引用格式化、验证、修复等功能，支持多种引用格式。
"""
import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class CitationFormatter:
    """引用格式化器"""

    @staticmethod
    def build_context_blocks(
        context_metadata: List[Dict[str, Any]],
        format_type: str = "number"
    ) -> str:
        """
        构建带引用标注的上下文块

        Args:
            context_metadata: 上下文元数据列表
            format_type: 引用格式类型 ("number", "file", "mixed")

        Returns:
            格式化后的上下文字符串
        """
        blocks = []

        for meta in context_metadata:
            idx = meta.get('index', 0)
            content = meta.get('content', '')
            source = meta.get('source', {})

            if format_type == "number":
                # [1] content
                blocks.append(f"[{idx}] {content}")

            elif format_type == "file":
                # [docs/api.md] content
                file_path = source.get('file_path', '未知来源')
                file_name = file_path.split('/')[-1] if file_path else '未知'
                blocks.append(f"[{file_name}] {content}")

            elif format_type == "mixed":
                # [1: api.md] content
                file_path = source.get('file_path', '')
                file_name = file_path.split('/')[-1] if file_path else ''
                citation = f"[{idx}" + (f": {file_name}" if file_name else "]")
                blocks.append(f"{citation} {content}")

        return "\n\n".join(blocks)

    @staticmethod
    def format_citation_list(context_metadata: List[Dict[str, Any]]) -> str:
        """
        格式化引用列表（用于答案末尾）

        Args:
            context_metadata: 上下文元数据列表

        Returns:
            格式化后的引用列表字符串
        """
        lines = ["参考来源：", ""]

        for meta in context_metadata:
            idx = meta.get('index', 0)
            source = meta.get('source', {})
            retrieval_info = meta.get('retrieval_info', {})

            file_path = source.get('file_path', '未知来源')
            source_url = source.get('source_url', '')

            # 构建引用信息
            citation = f"[{idx}] {file_path}"
            if source_url:
                citation += f" - {source_url}"

            # 添加检索方法
            method = retrieval_info.get('method', 'unknown')
            if method == 'vector':
                citation += " (向量检索)"
            elif method == 'bm25':
                citation += " (关键词检索)"
            elif method == 'hybrid_bm25':
                citation += " (混合检索)"

            lines.append(citation)

        return "\n".join(lines)


class CitationValidator:
    """引用验证器"""

    # 引用格式正则表达式
    CITATION_PATTERNS = {
        'number': r'\[(\d+)\]',  # [1], [2]
        'file': r'\[([^\]]+\.(md|txt|py|js))\]',  # [file.md]
        'mixed': r'\[(\d+)(?::[^\]]+)?\]'  # [1], [1: file.md]
    }

    @staticmethod
    def has_citations(text: str, format_type: str = "number") -> bool:
        """
        检查文本中是否包含引用

        Args:
            text: 待检查文本
            format_type: 引用格式类型

        Returns:
            是否包含引用
        """
        pattern = CitationValidator.CITATION_PATTERNS.get(format_type)
        if not pattern:
            return False

        matches = re.findall(pattern, text)
        return len(matches) > 0

    @staticmethod
    def extract_citations(text: str, format_type: str = "number") -> List[str]:
        """
        提取文本中的所有引用

        Args:
            text: 待提取文本
            format_type: 引用格式类型

        Returns:
            引用列表
        """
        pattern = CitationValidator.CITATION_PATTERNS.get(format_type)
        if not pattern:
            return []

        matches = re.findall(pattern, text)
        if format_type == "number":
            # 返回匹配的完整字符串
            return re.findall(r'\[\d+\]', text)
        elif format_type == "mixed":
            return re.findall(r'\[\d+(?::[^\]]+)?\]', text)
        else:
            return matches

    @staticmethod
    def validate_citations(
        answer: str,
        expected_count: int,
        format_type: str = "number"
    ) -> Dict[str, Any]:
        """
        验证答案中的引用

        Args:
            answer: 答案文本
            expected_count: 预期的引用数量
            format_type: 引用格式类型

        Returns:
            验证结果字典
        """
        has_cit = CitationValidator.has_citations(answer, format_type)
        citations = CitationValidator.extract_citations(answer, format_type)

        # 检查引用编号是否在合理范围内
        valid_numbers = []
        for cit in citations:
            match = re.search(r'\[(\d+)\]', cit)
            if match:
                num = int(match.group(1))
                if 1 <= num <= expected_count:
                    valid_numbers.append(num)

        return {
            'has_citations': has_cit,
            'citation_count': len(citations),
            'valid_count': len(valid_numbers),
            'is_valid': has_cit and len(valid_numbers) > 0,
            'citations': citations
        }


class CitationFixer:
    """引用修复器 - 容错机制"""

    @staticmethod
    def ensure_citations(
        answer: str,
        context_metadata: List[Dict[str, Any]],
        format_type: str = "number"
    ) -> str:
        """
        确保答案包含引用（容错机制）

        如果 LLM 没有添加引用，则在答案末尾添加引用列表。

        Args:
            answer: 原始答案
            context_metadata: 上下文元数据
            format_type: 引用格式类型

        Returns:
            包含引用的答案
        """
        # 验证是否已有引用
        validation = CitationValidator.validate_citations(
            answer,
            len(context_metadata),
            format_type
        )

        if validation['is_valid']:
            # 已有有效引用，直接返回
            return answer

        # 没有引用或引用无效，添加引用列表
        logger.info(f"答案缺少有效引用，添加引用列表。原引用数: {validation['citation_count']}")

        citation_list = CitationFormatter.format_citation_list(context_metadata)

        return f"{answer}\n\n{citation_list}"

    @staticmethod
    def fix_citation_format(
        answer: str,
        context_metadata: List[Dict[str, Any]],
        from_format: str = "file",
        to_format: str = "number"
    ) -> str:
        """
        转换引用格式

        Args:
            answer: 原始答案
            context_metadata: 上下文元数据
            from_format: 源格式
            to_format: 目标格式

        Returns:
            格式转换后的答案
        """
        # 构建文件名到编号的映射
        file_to_index = {}
        for meta in context_metadata:
            idx = meta.get('index', 0)
            source = meta.get('source', {})
            file_path = source.get('file_path', '')
            file_name = file_path.split('/')[-1] if file_path else f"source_{idx}"
            file_to_index[file_name] = idx

        # 替换引用格式
        if from_format == "file" and to_format == "number":
            # [docs/api.md] -> [1]
            for file_name, idx in file_to_index.items():
                # 替换多种可能的格式
                answer = re.sub(
                    rf'\[{re.escape(file_name)}\]',
                    f'[{idx}]',
                    answer
                )

        return answer


def build_citation_prompt(
    context_metadata: List[Dict[str, Any]],
    query: str,
    format_type: str = "number"
) -> str:
    """
    构建带引用指令的 prompt

    Args:
        context_metadata: 上下文元数据
        query: 用户查询
        format_type: 引用格式类型

    Returns:
        完整的 prompt
    """
    # 构建上下文块
    context_blocks = CitationFormatter.build_context_blocks(
        context_metadata,
        format_type
    )

    # 引用指令
    if format_type == "number":
        citation_instruction = """
**强制要求**：你必须在答案中使用引用格式！

每当你使用上下文中的信息时，必须在对应位置标注引用编号 [1]、[2]、[3] 等。

正确示例：
- 根据文档说明[1]，这个功能用于...
- 配置方法包括以下步骤[2]：首先...其次...
- 如果没有[3]配置，系统将...

错误示例（不要这样）：
- ❌ 根据文档说明，这个功能用于... （缺少引用）
- ❌ 配置方法包括以下步骤：首先...其次... （缺少引用）

记住：答案中的每个主要观点都必须有对应的引用编号！
"""
    else:
        citation_instruction = """
**重要**：在答案中引用信息时，请标注来源。
"""

    prompt = f"""基于以下上下文回答用户问题。

{citation_instruction}

上下文：
{context_blocks}

问题：{query}

答案："""

    return prompt


def post_process_answer(
    answer: str,
    context_metadata: List[Dict[str, Any]],
    format_type: str = "number",
    enable_fix: bool = True
) -> Dict[str, Any]:
    """
    后处理答案：验证和修复引用

    Args:
        answer: 原始答案
        context_metadata: 上下文元数据
        format_type: 引用格式类型
        enable_fix: 是否启用修复

    Returns:
        处理结果字典
        """
    # 验证引用
    validation = CitationValidator.validate_citations(
        answer,
        len(context_metadata),
        format_type
    )

    result = {
        'original_answer': answer,
        'has_citations': validation['has_citations'],
        'citation_count': validation['citation_count'],
        'is_valid': validation['is_valid'],
        'citations': validation['citations']
    }

    # 如果启用修复且答案无效
    if enable_fix and not validation['is_valid']:
        fixed_answer = CitationFixer.ensure_citations(
            answer,
            context_metadata,
            format_type
        )
        result['fixed_answer'] = fixed_answer
        result['was_fixed'] = True
    else:
        result['fixed_answer'] = answer
        result['was_fixed'] = False

    return result
