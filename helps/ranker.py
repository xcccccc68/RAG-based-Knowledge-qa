# 重排序模块

from core.models.rerank import create_reranker
from langchain_core.documents import Document
from typing import List
from utils.logger import logger


class Ranker:
    """
    重排序模块，用于对检索结果进行重排序
    """
    
    def __init__(self):
        """
        初始化重排序器
        
        创建重排序模型实例
        """
        self.reranker = create_reranker()
        logger.info("重排序器初始化成功")

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        对文档列表进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果
            
        Returns:
            List[Document]: 重排序后的文档列表
        """
        try:
            if not documents:
                return []
            
            return self.reranker.rerank_documents(query, documents, top_k=top_k)
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents


# 单例实例
_ranker_instance = None

def create_ranker():
    """
    创建重排序器实例（单例模式）
    
    Returns:
        Ranker: 重排序器实例
    """
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = Ranker()
    return _ranker_instance
