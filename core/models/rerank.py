import requests
from openai import OpenAI
from typing import List, Tuple
from configs.config import settings
from utils.logger import logger


class Reranker:
    def __init__(self):
        self.api_url = settings.RERANK_API_URL
        self.model = settings.RERANK_MODEL_NAME
        self.timeout = settings.RERANK_TIMEOUT
        
        # 使用 OpenAI 原生客户端（但 rerank API 格式不同，仍用 requests）
        self.client = OpenAI(
            api_key="not-used",  # Rerank 不需要认证
            base_url=self.api_url,
            timeout=self.timeout * 1000
        )
        logger.info(f"初始化重排序器：{self.model}")

    def _call_api(self, query: str, documents: List[str]) -> List[dict]:
        """调用 rerank API"""
        response = requests.post(
            f"{self.api_url}/rerank",
            json={"model": self.model, "query": query, "documents": documents},
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        return result.get("results", [])

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        if not documents:
            return []

        try:
            logger.info(f"开始重排序 {len(documents)} 个文档")
            results = self._call_api(query, documents)
            ranked_results = [(item.get("index", 0), item.get("score", 0.0)) for item in results]
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"重排序完成，返回前 {top_k if top_k else len(ranked_results)} 个结果")
            return ranked_results[:top_k] if top_k else ranked_results
        except Exception as e:
            logger.error(f"重排序失败：{e}")
            return [(i, 0.0) for i in range(len(documents))]

    def rerank_documents(self, query: str, documents: List, top_k: int = None) -> List:
        if not documents:
            return []

        doc_texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                doc_texts.append(doc.page_content)
            elif isinstance(doc, str):
                doc_texts.append(doc)
            else:
                doc_texts.append(str(doc))

        ranked_indices = self.rerank(query, doc_texts, top_k)
        reranked_docs = []
        for index, score in ranked_indices:
            if 0 <= index < len(documents):
                doc = documents[index]
                if hasattr(doc, 'metadata'):
                    doc.metadata['rerank_score'] = score
                reranked_docs.append(doc)

        return reranked_docs


# 单例实例
_reranker_instance = None

def create_reranker():
    """
    创建重排序器实例（单例模式）
    
    Returns:
        Reranker: 重排序器实例
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
