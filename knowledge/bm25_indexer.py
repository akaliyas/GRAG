"""
BM25 关键词检索模块

提供 BM25 索引构建和检索功能，与 LightRAG 向量检索互补。
使用 rank_bm25 库实现高效的稀疏检索。
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import jieba
import jieba.analyse

logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    BM25 索引器

    提供文档索引构建、关键词检索和结果排序功能。
    支持中文分词和增量更新。
    """

    def __init__(
        self,
        index_dir: str = "./rag_storage/bm25",
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        初始化 BM25 索引器

        Args:
            index_dir: 索引文件存储目录
            k1: BM25 参数，控制词频饱和度 (默认 1.5)
            b: BM25 参数，控制文档长度归一化 (默认 0.75)
            epsilon: BM25 参数，用于下界归一化 (默认 0.25)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # BM25 参数
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # 索引数据
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []  # 原始文档
        self.tokenized_docs: List[List[str]] = []  # 分词后的文档
        self.doc_ids: List[str] = []  # 文档 ID 列表

        # 索引文件路径
        self.index_file = self.index_dir / "bm25_index.pkl"
        self.meta_file = self.index_dir / "bm25_meta.pkl"

        # 尝试加载已有索引
        self._load_index()

        logger.info(f"BM25 索引器初始化完成，目录: {self.index_dir}")
        logger.info(f"  当前文档数: {len(self.documents)}")
        logger.info(f"  BM25 参数: k1={k1}, b={b}, epsilon={epsilon}")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        # 使用 jieba 分词
        tokens = jieba.lcut(text)

        # 过滤停用词和短词
        stopwords = self._get_stopwords()
        tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]

        return tokens

    def _get_stopwords(self) -> set:
        """获取中文停用词表"""
        # 常用中文停用词
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '有',
            '个', '之', '为', '或', '与', '及', '等', '以', '于', '对',
            '中', '而', '能', '由', '从', '被', '把', '向', '使', '因',
            '但', '则', '又', '若', '如', '虽', '即使', '既然', '因为',
            '所以', '因此', '于是', '不过', '但是', '然而', '而且', '并且',
            '接着', '于是', '从而', '进而', '故', '此', '彼', '其', '它',
            '什么', '怎么', '如何', '哪里', '哪些', '哪个', '多少', '几',
            '吗', '呢', '吧', '啊', '呀', '哦', '嗯', '哈', '啦', '呐',
            '这个', '那个', '这些', '那些', '这样', '那样', '这种', '那种'
        }
        return stopwords

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        添加文档到索引

        Args:
            documents: 文档列表，每个文档包含 content 和 doc_id

        Returns:
            成功添加的文档数量
        """
        added_count = 0

        for doc in documents:
            doc_id = doc.get('doc_id') or doc.get('id', '')
            content = doc.get('content', '')

            if not doc_id or not content:
                logger.warning(f"文档缺少 doc_id 或 content，跳过")
                continue

            # 检查是否已存在
            if doc_id in self.doc_ids:
                logger.debug(f"文档已存在，跳过: {doc_id}")
                continue

            # 分词
            tokens = self._tokenize(content)

            if not tokens:
                logger.warning(f"文档分词结果为空，跳过: {doc_id}")
                continue

            # 添加到索引
            self.documents.append(doc)
            self.doc_ids.append(doc_id)
            self.tokenized_docs.append(tokens)

            added_count += 1

        # 重建 BM25 索引
        if added_count > 0:
            self._rebuild_index()

        logger.info(f"添加 {added_count} 个文档到 BM25 索引")
        return added_count

    def _rebuild_index(self):
        """重建 BM25 索引"""
        if not self.tokenized_docs:
            logger.warning("没有文档，无法重建索引")
            return

        try:
            self.bm25 = BM25Okapi(
                self.tokenized_docs,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon
            )
            logger.debug(f"BM25 索引重建完成，文档数: {len(self.tokenized_docs)}")
        except Exception as e:
            logger.error(f"重建 BM25 索引失败: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表，每个结果包含 doc_id, score, content
        """
        if self.bm25 is None:
            logger.warning("BM25 索引未初始化，返回空结果")
            return []

        # 分词
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning(f"查询分词结果为空: {query}")
            return []

        # BM25 打分
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception as e:
            logger.error(f"BM25 打分失败: {e}")
            return []

        # 获取 top-k
        top_indices = scores.argsort()[-top_k:][::-1]

        # 构建结果
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:  # 过滤零分结果
                continue

            doc_id = self.doc_ids[idx]
            doc = self.documents[idx]

            results.append({
                'doc_id': doc_id,
                'score': float(score),
                'content': doc.get('content', ''),
                'metadata': doc.get('metadata', {})
            })

        logger.debug(f"BM25 检索完成，查询: {query}, 结果数: {len(results)}")
        return results

    def save_index(self) -> bool:
        """
        保存索引到磁盘

        Returns:
            保存成功返回 True，失败返回 False
        """
        try:
            # 保存 BM25 索引
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.bm25, f)

            # 保存元数据
            meta = {
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
                'doc_ids': self.doc_ids,
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
            with open(self.meta_file, 'wb') as f:
                pickle.dump(meta, f)

            logger.info(f"BM25 索引已保存到: {self.index_file}")
            return True

        except Exception as e:
            logger.error(f"保存 BM25 索引失败: {e}")
            return False

    def _load_index(self) -> bool:
        """
        从磁盘加载索引

        Returns:
            加载成功返回 True，失败返回 False
        """
        try:
            if not self.index_file.exists() or not self.meta_file.exists():
                logger.debug("BM25 索引文件不存在，将创建新索引")
                return False

            # 加载元数据
            with open(self.meta_file, 'rb') as f:
                meta = pickle.load(f)

            self.documents = meta.get('documents', [])
            self.tokenized_docs = meta.get('tokenized_docs', [])
            self.doc_ids = meta.get('doc_ids', [])

            # 加载 BM25 索引
            with open(self.index_file, 'rb') as f:
                self.bm25 = pickle.load(f)

            logger.info(f"BM25 索引已加载，文档数: {len(self.documents)}")
            return True

        except Exception as e:
            logger.error(f"加载 BM25 索引失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_documents': len(self.documents),
            'index_loaded': self.bm25 is not None,
            'index_file': str(self.index_file),
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }

    def clear_index(self) -> bool:
        """
        清空索引

        Returns:
            清空成功返回 True，失败返回 False
        """
        try:
            self.bm25 = None
            self.documents = []
            self.tokenized_docs = []
            self.doc_ids = []

            # 删除索引文件
            if self.index_file.exists():
                self.index_file.unlink()
            if self.meta_file.exists():
                self.meta_file.unlink()

            logger.info("BM25 索引已清空")
            return True

        except Exception as e:
            logger.error(f"清空 BM25 索引失败: {e}")
            return False


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) 算法

    融合多个检索结果列表，生成统一的排序结果。

    Args:
        results_list: 多个检索结果列表的列表
                      每个结果必须包含 doc_id 和 score
        k: RRF 常数，用于调整排名权重 (默认 60)

    Returns:
        融合后的结果列表，按 RRF 分数降序排列
    """
    # 存储 RRF 分数
    rrf_scores = {}

    # 遍历每个检索结果列表
    for results in results_list:
        # 按原始分数排序
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

        # 计算 RRF 分数
        for rank, result in enumerate(sorted_results, start=1):
            doc_id = result.get('doc_id')
            if not doc_id:
                continue

            # RRF 公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score

            # 保留第一个包含该 doc_id 的结果的完整信息
            if 'content' not in rrf_scores.get(doc_id, {}):
                rrf_scores[doc_id] = {
                    'doc_id': doc_id,
                    'rrf_score': rrf_score,
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'original_scores': result.get('original_scores', [])
                }
                result.setdefault('original_scores', []).append({
                    'source': result.get('source', 'unknown'),
                    'score': result.get('score', 0)
                })
            else:
                # 累加 RRF 分数
                rrf_scores[doc_id]['rrf_score'] += rrf_score
                if 'original_scores' in result:
                    rrf_scores[doc_id]['original_scores'].extend(result.get('original_scores', []))

    # 按 RRF 分数排序
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    return sorted_results
