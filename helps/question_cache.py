"""
高频问题缓存模块
将高频问题及其答案缓存到 Redis 数据库（DB 3）中
通过相似度计算实现问题匹配，置信度阈值 0.9
"""

import json
import redis
import threading
from typing import Dict, List, Optional, Any, Tuple
from configs.config import settings
from utils.logger import logger
from core.models.embeddings import APIEmbeddings
import numpy as np


class QuestionCache:
    """
    高频问题缓存管理器（线程安全的单例模式）
    使用 Redis DB 3 存储问题缓存
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_cache()
        return cls._instance
    
    def _init_cache(self):
        """初始化缓存连接和组件"""
        self.redis_client = None
        self.embedding_model = None
        self.similarity_threshold = settings.QUESTION_CACHE_SIMILARITY_THRESHOLD
        self.db_index = settings.QUESTION_CACHE_DB
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=self.db_index,
                password=settings.REDIS_PASSWORD,
                decode_responses=settings.REDIS_DECODE_RESPONSES
            )
            self.redis_client.ping()
            logger.info(f"QUESTION_CACHE_INIT: Redis DB {self.db_index} 连接成功")
        except Exception as e:
            logger.error(f"QUESTION_CACHE_INIT: Redis DB {self.db_index} 连接失败：{e}")
            self.redis_client = None
        
        try:
            self.embedding_model = APIEmbeddings()
            logger.info("QUESTION_CACHE_INIT: 嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"QUESTION_CACHE_INIT: 嵌入模型初始化失败：{e}")
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """检查缓存是否可用"""
        if not self.redis_client or not self.embedding_model:
            return False
        try:
            return self.redis_client.ping()
        except Exception:
            return False
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 向量 1
            vector2: 向量 2
            
        Returns:
            float: 余弦相似度值 (0-1)
        """
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失败：{e}")
            return 0.0
    
    def _get_question_embedding(self, question: str) -> Optional[List[float]]:
        """
        获取问题的嵌入向量
        
        Args:
            question: 问题文本
            
        Returns:
            Optional[List[float]]: 嵌入向量，失败返回 None
        """
        try:
            embeddings = self.embedding_model.embed_query(question)
            return embeddings
        except Exception as e:
            logger.error(f"获取问题嵌入向量失败：{e}")
            return None
    
    def search_similar_question(self, question: str) -> Optional[Dict[str, Any]]:
        """
        搜索相似问题
        
        Args:
            question: 当前问题
            
        Returns:
            Optional[Dict[str, Any]]: 匹配的缓存问题及答案，未找到返回 None
        """
        if not self.is_available():
            logger.warning("问题缓存不可用")
            return None
        
        try:
            current_embedding = self._get_question_embedding(question)
            if not current_embedding:
                logger.warning("无法获取当前问题的嵌入向量")
                return None
            
            all_questions = self.redis_client.hkeys("question_cache:questions")
            
            if not all_questions:
                logger.debug("缓存中没有问题记录")
                return None
            
            best_match = None
            best_similarity = 0.0
            
            for cached_question in all_questions:
                cached_data = self.redis_client.hget("question_cache:questions", cached_question)
                if not cached_data:
                    continue
                
                cached_info = json.loads(cached_data)
                cached_embedding = cached_info.get('embedding')
                
                if not cached_embedding:
                    continue
                
                similarity = self._calculate_similarity(current_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'question': cached_question,
                        'answer': cached_info.get('answer'),
                        'similarity': similarity
                    }
            
            if best_similarity >= self.similarity_threshold:
                logger.info(f"找到匹配问题：'{best_match['question']}' (相似度：{best_similarity:.4f})")
                return best_match
            else:
                logger.debug(f"未找到匹配问题 (最高相似度：{best_similarity:.4f})")
                return None
                
        except Exception as e:
            logger.error(f"搜索相似问题失败：{e}")
            return None
    
    def add_question(self, question: str, answer: str) -> bool:
        """
        添加问题到缓存
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            bool: 添加成功返回 True，失败返回 False
        """
        if not self.is_available():
            logger.warning("问题缓存不可用，无法添加")
            return False
        
        try:
            embedding = self._get_question_embedding(question)
            if not embedding:
                logger.warning("无法获取问题嵌入向量，添加失败")
                return False
            
            cached_data = {
                'answer': answer,
                'embedding': embedding,
                'created_at': str(threading.current_thread().ident)
            }
            
            self.redis_client.hset("question_cache:questions", question, json.dumps(cached_data))
            logger.info(f"问题缓存添加成功：'{question}'")
            return True
            
        except Exception as e:
            logger.error(f"添加问题到缓存失败：{e}")
            return False
    
    def remove_question(self, question: str) -> bool:
        """
        从缓存中移除问题
        
        Args:
            question: 问题文本
            
        Returns:
            bool: 移除成功返回 True，失败返回 False
        """
        if not self.is_available():
            logger.warning("问题缓存不可用，无法移除")
            return False
        
        try:
            result = self.redis_client.hdel("question_cache:questions", question)
            if result:
                logger.info(f"问题缓存移除成功：'{question}'")
                return True
            else:
                logger.warning(f"问题不存在：'{question}'")
                return False
        except Exception as e:
            logger.error(f"移除问题失败：{e}")
            return False
    
    def get_all_questions(self) -> List[str]:
        """
        获取所有缓存的问题列表
        
        Returns:
            List[str]: 问题列表
        """
        if not self.is_available():
            return []
        
        try:
            questions = self.redis_client.hkeys("question_cache:questions")
            return questions if questions else []
        except Exception as e:
            logger.error(f"获取问题列表失败：{e}")
            return []
    
    def clear_cache(self) -> bool:
        """
        清空问题缓存
        
        Returns:
            bool: 清空成功返回 True，失败返回 False
        """
        if not self.is_available():
            logger.warning("问题缓存不可用，无法清空")
            return False
        
        try:
            self.redis_client.delete("question_cache:questions")
            logger.info("问题缓存已清空")
            return True
        except Exception as e:
            logger.error(f"清空缓存失败：{e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.is_available():
            return {"available": False}
        
        try:
            question_count = self.redis_client.hlen("question_cache:questions")
            return {
                "available": True,
                "question_count": question_count,
                "db_index": self.db_index,
                "similarity_threshold": self.similarity_threshold
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败：{e}")
            return {"available": False}


def create_question_cache() -> QuestionCache:
    """
    创建问题缓存实例
    
    Returns:
        QuestionCache: 问题缓存单例实例
    """
    return QuestionCache()
