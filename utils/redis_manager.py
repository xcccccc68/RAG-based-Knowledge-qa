"""
Redis管理器
统一管理Redis连接和操作，包括对话数据、黑白名单等
"""

import json
import threading
from typing import Dict, List, Set, Optional, Any
import redis
from configs.config import settings


class RedisManager:
    """Redis管理器，统一处理所有Redis操作（线程安全的单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_redis()
        return cls._instance
    
    def _init_redis(self):
        """初始化Redis连接"""
        self.redis_client = None
        self._logger = None
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=settings.REDIS_DECODE_RESPONSES == "True"
            )
            self.redis_client.ping()
            self.logger.info("REDIS_MANAGER_INIT: Redis连接成功")
        except Exception as e:
            self.logger.error(f"REDIS_MANAGER_INIT: Redis连接失败: {e}，将使用本地存储")
            self.redis_client = None
    
    @property
    def logger(self):
        if self._logger is None:
            from utils.logger import logger
            self._logger = logger
        return self._logger
    
    def is_available(self) -> bool:
        """检查Redis是否可用"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except Exception:
            return False
    
    # ========== 黑白名单管理 ==========
    
    def load_sensitive_words(self) -> List[str]:
        """加载敏感词列表"""
        if not self.is_available():
            return []
        
        try:
            words = self.redis_client.smembers("security:sensitive_words")
            return list(words) if words else []
        except Exception as e:
            self.logger.warning(f"从Redis加载敏感词失败: {e}")
            return []
    
    def save_sensitive_words(self, words: List[str]) -> bool:
        """保存敏感词列表"""
        if not self.is_available() or not words:
            return False
        
        try:
            self.redis_client.sadd("security:sensitive_words", *words)
            self.logger.info(f"将 {len(words)} 个敏感词保存到Redis")
            return True
        except Exception as e:
            self.logger.warning(f"保存敏感词到Redis失败: {e}")
            return False
    
    def add_sensitive_words(self, words: List[str]) -> bool:
        """添加敏感词"""
        if not self.is_available() or not words:
            return False
        
        try:
            self.redis_client.sadd("security:sensitive_words", *words)
            return True
        except Exception as e:
            self.logger.warning(f"添加敏感词到Redis失败: {e}")
            return False
    
    def load_whitelist(self) -> Set[str]:
        """加载白名单"""
        if not self.is_available():
            return set()
        
        try:
            words = self.redis_client.smembers("security:whitelist")
            return set(words) if words else set()
        except Exception as e:
            self.logger.warning(f"从Redis加载白名单失败: {e}")
            return set()
    
    def save_whitelist(self, words: Set[str]) -> bool:
        """保存白名单"""
        if not self.is_available() or not words:
            return False
        
        try:
            self.redis_client.sadd("security:whitelist", *words)
            self.logger.info(f"将 {len(words)} 个白名单词保存到Redis")
            return True
        except Exception as e:
            self.logger.warning(f"保存白名单到Redis失败: {e}")
            return False
    
    def add_whitelist_words(self, words: List[str]) -> bool:
        """添加白名单词"""
        if not self.is_available() or not words:
            return False
        
        try:
            self.redis_client.sadd("security:whitelist", *words)
            return True
        except Exception as e:
            self.logger.warning(f"添加白名单词到Redis失败: {e}")
            return False
    
    # ========== 对话数据管理 ==========
    
    def save_conversation(self, session_id: str, conversation_data: Dict[str, Any]) -> bool:
        """保存对话数据"""
        if not self.is_available() or not session_id or not conversation_data:
            return False
        
        try:
            key = f"conversation:{session_id}"
            self.redis_client.setex(key, settings.CONVERSATION_TIMEOUT, json.dumps(conversation_data))
            return True
        except Exception as e:
            self.logger.warning(f"保存对话数据到Redis失败: {e}")
            return False
    
    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载对话数据"""
        if not self.is_available() or not session_id:
            return None
        
        try:
            key = f"conversation:{session_id}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.warning(f"从Redis加载对话数据失败: {e}")
            return None
    
    def delete_conversation(self, session_id: str) -> bool:
        """删除对话数据"""
        if not self.is_available() or not session_id:
            return False
        
        try:
            key = f"conversation:{session_id}"
            self.redis_client.delete(key)
            return True
        except Exception as e:
            self.logger.warning(f"从Redis删除对话数据失败: {e}")
            return False
            return True
        except Exception as e:
            self.logger.warning(f"删除Redis键失败: {e}")
            return False


# 全局Redis管理器实例
redis_manager = RedisManager()