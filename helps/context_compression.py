# 上下文压缩模块

from typing import List, Dict, Any
from configs.config import settings
from core.models.llm import create_llm_client
from utils.prompt_manager import create_prompt_manager
import tiktoken
from utils.logger import logger


class ContentDeduplicator:
    """内容去重器"""
    
    def deduplicate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去除重复文档
        
        Args:
            docs: 文档列表
            
        Returns:
            List[Dict[str, Any]]: 去重后的文档列表
        """
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            content = doc.get('text', '')
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        return unique_docs


class RelevanceRanker:
    """相关性排序器"""
    
    def rank_and_truncate(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        对文档进行排序并截断
        
        Args:
            docs: 文档列表
            query: 查询文本
            
        Returns:
            List[Dict[str, Any]]: 排序并截断后的文档列表
        """
        # 简单的相关性排序（基于关键词匹配）
        def calculate_relevance(doc, query):
            content = doc.get('text', '').lower()
            query_words = set(query.lower().split())
            doc_words = set(content.split())
            return len(query_words.intersection(doc_words))
        
        # 排序
        ranked_docs = sorted(docs, key=lambda x: calculate_relevance(x, query), reverse=True)
        
        # 截断到前10个文档
        return ranked_docs[:10]


class SimilarityMerger:
    """相似度合并器"""
    
    def merge_similar_chunks(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并相似的文档块
        
        Args:
            docs: 文档列表
            
        Returns:
            List[Dict[str, Any]]: 合并后的文档列表
        """
        if not docs:
            return []
        
        merged_docs = [docs[0]]
        
        for doc in docs[1:]:
            content = doc.get('text', '')
            merged = False
            
            # 检查是否与已合并的文档相似
            for i, merged_doc in enumerate(merged_docs):
                merged_content = merged_doc.get('text', '')
                # 简单的相似度检查（基于内容重叠）
                if self._calculate_similarity(content, merged_content) > 0.5:
                    # 合并内容
                    merged_docs[i]['text'] = merged_content + ' ' + content
                    merged = True
                    break
            
            if not merged:
                merged_docs.append(doc)
        
        return merged_docs
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union


class SummaryGenerator:
    """摘要生成器"""
    
    def __init__(self):
        """
        初始化摘要生成器
        
        创建大模型客户端用于生成摘要
        """
        self.llm_client = create_llm_client()
        self.prompt_manager = create_prompt_manager()
    
    def generate_summary(self, context: str, query: str) -> str:
        """
        生成上下文摘要
        
        Args:
            context: 原始上下文
            query: 查询文本
            
        Returns:
            str: 生成的摘要
        """
        # 使用prompt_manager构建摘要提示词
        prompt = f"请对以下上下文进行摘要，重点关注与查询 '{query}' 相关的内容。摘要应简洁明了，保留关键信息，去除冗余内容。\n\n上下文：\n{context}\n\n摘要："
        
        # 调用大模型生成摘要
        summary = self.llm_client.generate(prompt)
        
        return summary


class ContextCompression:
    """上下文压缩工具"""
    
    def __init__(self):
        # 上下文压缩策略1: 内容去重 - 去除重复的文档内容
        self.deduplicator = ContentDeduplicator()
        # 上下文压缩策略2: 相关性排序 - 根据与查询的相关性对文档进行排序并截断
        self.ranker = RelevanceRanker()
        # 上下文压缩策略3: 相似度合并 - 合并相似的文档块，减少冗余信息
        self.merger = SimilarityMerger()
        # 上下文压缩策略4: 摘要总结 - 使用大模型对上下文进行摘要，节省上下文空间
        self.summarizer = SummaryGenerator()
        # 从配置读取上下文token限制
        self.max_tokens = settings.MAX_CONTEXT_TOKENS  # 上下文token上限
        self.enable_compression = settings.ENABLE_CONTEXT_COMPRESSION  # 是否启用压缩
        # 初始化tiktoken
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            logger.info("tiktoken加载成功")
        except Exception as e:
            logger.error(f"tiktoken加载失败: {e}，使用降级方案")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 文本内容
            
        Returns:
            int: 计算的token数量
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.error(f"tiktoken计算失败: {e}，使用降级方案")
        
        # 降级方案：粗略估算
        return len(text.split()) * 1.3  # 粗略估算
    
    def compress_context(self, context: str, query: str, strategies: List[str] = None) -> str:
        """
        压缩上下文
        
        Args:
            context: 原始上下文
            query: 查询文本
            strategies: 要使用的压缩策略列表，可选值: ['deduplicate', 'rank', 'merge', 'summarize']
                      如果为None，则使用所有策略
            
        Returns:
            str: 压缩后的上下文
        """
        # 检查是否启用压缩
        if not self.enable_compression:
            return context
        
        # 检查是否需要压缩
        current_tokens = self.count_tokens(context)
        if current_tokens <= self.max_tokens:
            return context
        
        # 压缩策略1: 摘要总结（作为并列策略，与其他策略平级）
        if 'summarize' in strategies:
            try:
                summary = self.summarizer.generate_summary(context, query)
                final_context = f"[摘要]\n{summary}\n"
                if self.count_tokens(final_context) <= self.max_tokens:
                    return final_context
            except Exception as e:
                pass
        
        # 将上下文分割为文档块
        docs = []
        # 简单分割（实际应用中可能需要更复杂的分割逻辑）
        chunks = context.split('\n\n')
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append({'text': chunk.strip(), 'id': i})
        
        # 策略2: 内容去重
        if 'deduplicate' in strategies:
            docs = self.deduplicator.deduplicate(docs)
        
        # 策略3: 相关性排序
        if 'rank' in strategies:
            docs = self.ranker.rank(docs, query)
        
        # 策略4: 相似度合并
        if 'merge' in strategies:
            docs = self.merger.merge(docs)
        
        # 构建最终上下文
        final_context = self.build_context(docs)
        final_tokens = self.count_tokens(final_context)
        if final_tokens <= self.max_tokens:
            return final_context
        
        # 如果仍然超过限制，返回部分上下文
        return final_context[:int(self.max_tokens * 0.8)]  # 安全截断
    
    def build_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        构建最终上下文
        
        Args:
            docs: 文档列表
            
        Returns:
            str: 构建好的上下文
        """
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            # 添加文档标记
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc.get('text', ''))
            context_parts.append("")  # 空行分隔
        
        return '\n'.join(context_parts)


def create_context_compression() -> ContextCompression:
    """
    创建上下文压缩工具实例
    
    Returns:
        ContextCompression: 上下文压缩工具实例
    """
    return ContextCompression()