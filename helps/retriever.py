# 检索模块

from utils.milvus_manager import create_milvus_manager
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
import re
import os
from utils.logger import logger


class MilvusRetriever(BaseRetriever):
    """
    Milvus检索器，实现langchain的BaseRetriever接口
    提供三种检索方式：向量检索、关键词检索、混合检索
    """
    model_config = ConfigDict(extra='allow')

    def __init__(self, retrieval_mode: str = "hybrid", **kwargs: Any):
        """
        初始化检索器

        Args:
            retrieval_mode: 检索模式，可选值：vector, keyword, hybrid
        """
        super().__init__(**kwargs)
        self.vector_store_manager = create_milvus_manager()
        self.retrieval_mode = retrieval_mode
        self.stop_words = set()
        self._load_stop_words()
        logger.info(f"检索器初始化成功，使用模式: {retrieval_mode}")

    def _load_stop_words(self):
        """
        从stopword.txt文件加载停用词
        """
        stopword_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'stopword.txt')

        if os.path.exists(stopword_path):
            try:
                with open(stopword_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            self.stop_words.add(word)
                logger.info(f"已加载 {len(self.stop_words)} 个停用词")
            except Exception as e:
                logger.error(f"加载停用词文件失败: {e}")
                self.stop_words = set()
        else:
            logger.warning(f"停用词文件不存在: {stopword_path}")

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词

        Args:
            query: 查询文本

        Returns:
            List[str]: 关键词列表
        """
        words = re.findall(r'\b\w+\b', query)
        keywords = [word for word in words if word not in self.stop_words and len(word) > 1]

        return keywords

    def _vector_retrieval(self, query: str, k: int = 8) -> List[Document]:
        """
        向量检索

        Args:
            query: 查询文本
            k: 返回前k个结果

        Returns:
            List[Document]: 检索结果
        """
        try:
            return self.vector_store_manager.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _keyword_retrieval(self, query: str, k: int = 3) -> List[Document]:
        """
        关键词检索

        Args:
            query: 查询文本
            k: 每个关键词返回前k个结果

        Returns:
            List[Document]: 检索结果
        """
        try:
            keywords = self._extract_keywords(query)
            keyword_docs = []

            if keywords:
                for keyword in keywords[:3]:
                    keyword_results = self.vector_store_manager.similarity_search(keyword, k=k)
                    keyword_docs.extend(keyword_results)

            return keyword_docs
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def _hybrid_retrieval(self, query: str, vector_k: int = 8, keyword_k: int = 3) -> List[Document]:
        """
        混合检索（向量检索 + 关键词检索）

        Args:
            query: 查询文本
            vector_k: 向量检索返回前k个结果
            keyword_k: 每个关键词返回前k个结果

        Returns:
            List[Document]: 检索结果
        """
        try:
            vector_docs = self._vector_retrieval(query, k=vector_k)
            keyword_docs = self._keyword_retrieval(query, k=keyword_k)

            combined_docs = vector_docs.copy()
            seen_contents = set([doc.page_content for doc in vector_docs])

            for doc in keyword_docs:
                if doc.page_content not in seen_contents:
                    combined_docs.append(doc)
                    seen_contents.add(doc.page_content)

            return combined_docs
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        实现BaseRetriever接口的方法（已弃用，但仍需实现）

        Args:
            query: 查询文本
            **kwargs: 额外参数

        Returns:
            List[Document]: 相关文档列表
        """
        k = kwargs.get('k', 8)

        if self.retrieval_mode == "vector":
            return self._vector_retrieval(query, k=k)
        elif self.retrieval_mode == "keyword":
            return self._keyword_retrieval(query, k=k)
        else:
            return self._hybrid_retrieval(query, vector_k=k)

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        实现BaseRetriever接口的invoke方法

        Args:
            input: 查询文本（字符串）
            config: 可选的配置参数

        Returns:
            List[Document]: 相关文档列表
        """
        k = 8
        if config and isinstance(config, dict):
            k = config.get('k', k)

        if self.retrieval_mode == "vector":
            return self._vector_retrieval(input, k=k)
        elif self.retrieval_mode == "keyword":
            return self._keyword_retrieval(input, k=k)
        else:
            return self._hybrid_retrieval(input, vector_k=k)


_retriever_instance = None

def create_retriever(retrieval_mode: str = "hybrid"):
    """
    创建检索器实例（单例模式）

    Args:
        retrieval_mode: 检索模式，可选值：vector, keyword, hybrid

    Returns:
        MilvusRetriever: 检索器实例
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MilvusRetriever(retrieval_mode=retrieval_mode)
    return _retriever_instance