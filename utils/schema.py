"""
Pydantic Schema 定义
作为模块间通信的数据契约（Protocol）

设计原则：
- 所有中间产物必须定义为 Pydantic Model
- 开发模式：Step A -> Pydantic Model -> JSON File -> Step B
- 生产模式：Step A -> Pydantic Model -> Memory/Queue -> Step B

模块结构：
- Pipeline Models: RawDoc, IngestionBatch, CleanDoc, CleanBatch
- API Models: QueryRequest, QueryResponse, etc.
- Agent Models: AgentState, IntentType
- Cache Models: QueryCacheEntry, CacheStats
- Metadata Models: DocMetadata
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, computed_field, model_validator, field_validator


class DocMetadata(BaseModel):
    """文档元数据模型

    标准化元数据结构，确保类型安全。
    """
    type: str = Field(..., description="文件类型：markdown, notebook, text")
    url: Optional[str] = Field(None, description="文档 URL（GitHub 或其他）")
    frontmatter: Dict[str, str] = Field(default_factory=dict, description="Markdown Frontmatter")
    tags: List[str] = Field(default_factory=list, description="文档标签")
    version: str = Field(default="1.0", description="元数据版本")
    extracted_at: Optional[datetime] = Field(None, description="提取时间")


class RawDoc(BaseModel):
    """原始文档模型（Raw Artifact）

    注意：此模型存储的是从 GitHub API 获取的原始内容，不做任何清洗。
    清洗逻辑应在 CleanDoc 阶段完成。
    """

    path: str = Field(..., description="文件路径（仓库内的相对路径）")
    content: str = Field(..., description="文档内容（原始，未清洗）")
    source_url: Optional[str] = Field(None, description="GitHub Raw URL（可选，建议提供）")
    file_type: str = Field(..., description="文件类型：markdown 或 notebook")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据（Frontmatter + 其他）")

    @computed_field
    @property
    def doc_id(self) -> str:
        """生成确定性 ID，保证幂等性

        使用 source_url + path 生成 hash，确保跨仓库文档 ID 唯一。
        如果 source_url 不可用，则降级为仅使用 path。

        改进：修复跨仓库 ID 冲突问题。
        """
        if self.source_url:
            unique_key = f"{self.source_url}:{self.path}"
        else:
            unique_key = self.path
        return hashlib.md5(unique_key.encode()).hexdigest()

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: str) -> str:
        """验证文件类型"""
        allowed_types = {"markdown", "notebook", "text", ".md", ".mdx", ".ipynb"}
        if v not in allowed_types:
            # 标准化文件类型
            if v.endswith((".md", ".mdx")):
                return "markdown"
            elif v.endswith(".ipynb"):
                return "notebook"
            else:
                return "text"
        return v


class IngestionBatch(BaseModel):
    """批量导入批次（Raw Artifact）

    用于存储从 GitHub API 提取的原始数据。
    作为"不可变源数据"，如果清洗逻辑改了，不需要重新请求 GitHub API。
    """

    batch_id: str = Field(..., description="批次 ID（用于标识此批次）")
    repo_url: str = Field(..., description="数据源 URL（GitHub 仓库 URL）")
    docs: List[RawDoc] = Field(..., description="文档列表")
    extracted_at: datetime = Field(default_factory=datetime.now, description="提取时间")

    def save_to_file(self, path: str) -> None:
        """保存到文件（artifacts/01_raw/）

        改进：添加异常处理和更详细的错误信息。

        Args:
            path: 文件路径

        Raises:
            ValueError: 文件保存失败
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.model_dump_json(indent=2))
        except (OSError, IOError) as e:
            raise ValueError(f"Failed to save IngestionBatch to {path}: {e}") from e

    @classmethod
    def load_from_file(cls, path: str) -> "IngestionBatch":
        """从文件加载

        改进：添加异常处理和更详细的错误信息。

        Args:
            path: 文件路径

        Returns:
            IngestionBatch 对象

        Raises:
            ValueError: 文件加载失败
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return cls.model_validate_json(f.read())
        except (OSError, IOError) as e:
            raise ValueError(f"Failed to load IngestionBatch from {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse IngestionBatch from {path}: {e}") from e


class CleanDoc(BaseModel):
    """清洗后的文档模型（Clean Artifact）

    LightRAG 的高质量输入，已清洗 HTML、去除 Notebook 噪音、修复链接。
    必须继承 RawDoc 的关键字段（doc_id, source_url, file_path），确保溯源能力。

    改进：添加清洗状态追踪。
    """

    doc_id: str = Field(..., description="文档 ID（继承自 RawDoc，保持不变，用于幂等性）")
    content: str = Field(..., description="清洗后的纯文本（Frontmatter 已剥离）")
    source_url: str = Field(..., description="GitHub Raw URL（用于前端展示溯源）")
    file_path: str = Field(..., description="文件路径（用于调试和溯源）")
    file_type: str = Field(..., description="文件类型：markdown 或 notebook")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="增强后的 Metadata（Frontmatter + Tags）")
    cleaning_log: List[str] = Field(default_factory=list, description="清洗日志（方便 Debug）")
    cleaning_status: Literal["success", "failed", "partial"] = Field(
        default="success", description="清洗状态"
    )
    cleaning_error: Optional[str] = Field(None, description="清洗错误信息（如果失败）")


class CleanBatch(BaseModel):
    """清洗批次（Clean Artifact）

    用于存储清洗后的文档，作为 LightRAG 的输入。
    开发阶段可以在这里人工介入，打开 JSON 文件修补错误。
    """

    source_url: str = Field(..., description="数据源 URL（GitHub 仓库 URL）")
    docs: List[CleanDoc] = Field(..., description="清洗后的文档列表")
    cleaned_at: datetime = Field(default_factory=datetime.now, description="清洗时间")
    raw_batch_path: Optional[str] = Field(None, description="原始批次文件路径（可选）")

    def save_to_file(self, path: str) -> None:
        """保存到文件（artifacts/02_clean/）

        改进：添加异常处理和更详细的错误信息。

        Args:
            path: 文件路径

        Raises:
            ValueError: 文件保存失败
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.model_dump_json(indent=2))
        except (OSError, IOError) as e:
            raise ValueError(f"Failed to save CleanBatch to {path}: {e}") from e

    @classmethod
    def load_from_file(cls, path: str) -> "CleanBatch":
        """从文件加载

        改进：添加异常处理和更详细的错误信息。

        Args:
            path: 文件路径

        Returns:
            CleanBatch 对象

        Raises:
            ValueError: 文件加载失败
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return cls.model_validate_json(f.read())
        except (OSError, IOError) as e:
            raise ValueError(f"Failed to load CleanBatch from {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse CleanBatch from {path}: {e}") from e


# ==================== API 协议层 ====================

class QueryRequest(BaseModel):
    """查询请求（API 层）"""
    query: str = Field(..., description="查询文本", min_length=1)
    use_cache: bool = Field(default=True, description="是否使用缓存")
    stream: bool = Field(default=False, description="是否流式返回")


class QueryResponse(BaseModel):
    """查询响应（API 层）"""
    success: bool = Field(..., description="请求是否成功")
    answer: str = Field(..., description="生成的答案")
    context_ids: List[str] = Field(default_factory=list, description="上下文 ID 列表")
    response_time: float = Field(..., description="响应时间（秒）")
    model_type: str = Field(..., description="使用的模型类型")
    from_cache: bool = Field(default=False, description="是否来自缓存")
    error: Optional[str] = Field(None, description="错误信息（如果失败）")
    # 引用相关字段
    citations: List[str] = Field(default_factory=list, description="引用列表")
    citation_info: Optional[Dict[str, Any]] = Field(None, description="引用统计信息")
    context_metadata: List[Dict[str, Any]] = Field(default_factory=list, description="完整的上下文元数据")


class FeedbackRequest(BaseModel):
    """反馈请求（API 层）"""
    query: str = Field(..., description="查询文本", min_length=1)
    is_positive: bool = Field(..., description="True 为正面反馈，False 为负面反馈")


class FeedbackResponse(BaseModel):
    """反馈响应（API 层）"""
    success: bool = Field(..., description="反馈是否成功")
    message: str = Field(..., description="响应消息")


class ModelSwitchRequest(BaseModel):
    """模型切换请求（API 层）"""
    model_type: str = Field(..., description="模型类型：api 或 local（local 暂时禁用）")


class ModelSwitchResponse(BaseModel):
    """模型切换响应（API 层）"""
    success: bool = Field(..., description="切换是否成功")
    current_model: str = Field(..., description="当前模型类型")
    message: str = Field(..., description="响应消息")


# ==================== Agent 协议层 ====================

class IntentType(str, Enum):
    """意图类型枚举"""
    QUERY = "query"  # 直接查询
    GITHUB_INGEST = "github_ingest"  # 需要从 GitHub 获取数据（已弃用）
    PREPROCESS = "preprocess"  # 需要数据预处理（已弃用）
    UNKNOWN = "unknown"  # 未知意图


class AgentState(BaseModel):
    """
    Agent 状态模型（Pydantic 版本）

    改进：从 TypedDict 迁移到 Pydantic，提供运行时类型验证。
    用于 LangGraph 状态管理。

    注意：LangGraph 需要 Annotated 类型，使用时需要适配。
    """
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="消息列表")
    intent: str = Field(default="", description="意图类型")
    query: str = Field(default="", description="查询文本")
    need_github_ingest: bool = Field(default=False, description="是否需要从 GitHub 获取数据")
    need_preprocess: bool = Field(default=False, description="是否需要预处理")
    github_repo_urls: List[str] = Field(default_factory=list, description="GitHub 仓库 URL 列表")
    documents: List[str] = Field(default_factory=list, description="文档列表")
    context_ids: List[str] = Field(default_factory=list, description="检索到的上下文 ID")
    context_metadata: List[Dict[str, Any]] = Field(default_factory=list, description="上下文元数据")
    answer: str = Field(default="", description="生成的答案")
    citations: List[str] = Field(default_factory=list, description="引用列表")
    citation_info: Dict[str, Any] = Field(default_factory=dict, description="引用信息")
    error: Optional[str] = Field(None, description="错误信息")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 LangGraph 兼容性）"""
        return self.model_dump()


# ==================== Cache 协议层 ====================

class QueryCacheEntry(BaseModel):
    """查询缓存条目模型

    标准化缓存数据结构，提供类型安全。
    """
    query_hash: str = Field(..., description="查询的 SHA256 哈希值")
    query_text: str = Field(..., description="查询文本")
    answer: str = Field(..., description="生成的答案")
    context_ids: List[str] = Field(default_factory=list, description="上下文 ID 列表")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="质量评分（0-1）")
    feedback_count: int = Field(default=0, ge=0, description="反馈次数")
    positive_feedback: int = Field(default=0, ge=0, description="正面反馈数")
    negative_feedback: int = Field(default=0, ge=0, description="负面反馈数")
    model_type: str = Field(..., description="使用的模型类型")
    response_time: float = Field(..., ge=0.0, description="响应时间（秒）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    last_accessed_at: datetime = Field(default_factory=datetime.now, description="最后访问时间")
    access_count: int = Field(default=1, ge=1, description="访问次数")


class CacheStats(BaseModel):
    """缓存统计信息模型"""
    total_count: int = Field(..., ge=0, description="总缓存条目数")
    max_size: int = Field(..., ge=0, description="最大缓存条目数")
    backend: str = Field(..., description="存储后端类型：postgresql, json, redis")
    low_quality_count: int = Field(default=0, ge=0, description="低质量缓存数")
    high_quality_count: int = Field(default=0, ge=0, description="高质量缓存数")


# ==================== 导出辅助函数 ====================

def create_agent_state(**kwargs) -> AgentState:
    """
    创建 AgentState 实例的辅助函数

    用于替代旧的字典式状态初始化，提供类型安全。

    Args:
        **kwargs: 状态字段值

    Returns:
        AgentState 实例

    Example:
        state = create_agent_state(
            query="What is GRAG?",
            intent=IntentType.QUERY.value
        )
    """
    return AgentState(**kwargs)

