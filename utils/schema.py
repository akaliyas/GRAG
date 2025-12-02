"""
Pydantic Schema 定义
作为模块间通信的数据契约（Protocol）

设计原则：
- 所有中间产物必须定义为 Pydantic Model
- 开发模式：Step A -> Pydantic Model -> JSON File -> Step B
- 生产模式：Step A -> Pydantic Model -> Memory/Queue -> Step B
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from pydantic import BaseModel, Field, computed_field


class RawDoc(BaseModel):
    """原始文档模型（Raw Artifact）
    
    注意：此模型存储的是从 GitHub API 获取的原始内容，不做任何清洗。
    清洗逻辑应在 CleanDoc 阶段完成。
    """
    
    path: str = Field(..., description="文件路径（仓库内的相对路径）")
    content: str = Field(..., description="文档内容（原始，未清洗）")
    source_url: Optional[str] = Field(None, description="GitHub Raw URL（可选）")
    file_type: str = Field(..., description="文件类型：markdown 或 notebook")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据（Frontmatter + 其他）")
    
    @computed_field
    @property
    def doc_id(self) -> str:
        """生成确定性 ID，保证幂等性
        
        使用 file_path 生成 hash，确保同一文件多次处理 ID 不变。
        即使内容变了，ID 不变，LightRAG 会执行 Update 操作。
        """
        return hashlib.md5(self.path.encode()).hexdigest()


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
        
        Args:
            path: 文件路径
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load_from_file(cls, path: str) -> "IngestionBatch":
        """从文件加载
        
        Args:
            path: 文件路径
            
        Returns:
            IngestionBatch 对象
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())


class CleanDoc(BaseModel):
    """清洗后的文档模型（Clean Artifact）
    
    这是 LightRAG 的黄金输入，已清洗 HTML、去除 Notebook 噪音、修复链接。
    必须继承 RawDoc 的关键字段（doc_id, source_url, file_path），确保溯源能力。
    """
    
    doc_id: str = Field(..., description="文档 ID（继承自 RawDoc，保持不变，用于幂等性）")
    content: str = Field(..., description="清洗后的纯文本（Frontmatter 已剥离）")
    source_url: str = Field(..., description="GitHub Raw URL（用于前端展示溯源）")
    file_path: str = Field(..., description="文件路径（用于调试和溯源）")
    file_type: str = Field(..., description="文件类型：markdown 或 notebook")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="增强后的 Metadata（Frontmatter + Tags）")
    cleaning_log: List[str] = Field(default_factory=list, description="清洗日志（方便 Debug）")


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
        
        Args:
            path: 文件路径
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load_from_file(cls, path: str) -> "CleanBatch":
        """从文件加载
        
        Args:
            path: 文件路径
            
        Returns:
            CleanBatch 对象
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

