import time
import uuid
import json
import os
from typing import Dict, Optional, Any
from configs.config import settings
from utils.logger import logger
from utils.redis_manager import redis_manager


class ConversationManager:
    """
    会话管理器
    
    负责管理会话状态和超时判断，支持Redis存储和内存存储（带文件持久化）
    """
    
    def __init__(self):
        """
        初始化会话管理器
        """
        self.timeout_seconds = settings.CONVERSATION_TIMEOUT
        self.confidence_threshold = settings.INTENT_CONFIDENCE_THRESHOLD
        self.session_file = os.path.join(os.path.dirname(__file__), "../logs/sessions.json")
        
        # 检查Redis是否可用
        if redis_manager.is_available():
            logger.info("REDIS_INIT: Redis连接成功")
        else:
            logger.error(f"REDIS_INIT: Redis连接失败，将使用内存存储")
            # 加载持久化的会话数据
            self.memory_storage: Dict[str, Dict[str, Any]] = self._load_sessions()
            logger.info(f"SESSION_LOAD: 从文件加载会话数据，共 {len(self.memory_storage)} 个会话")
    
    def _load_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        从文件加载会话数据
        
        Returns:
            Dict[str, Dict[str, Any]]: 会话数据
        """
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"SESSION_LOAD: 加载会话数据失败: {e}")
        return {}
    
    def _save_sessions(self):
        """
        将会话数据保存到文件
        """
        if not redis_manager.is_available():
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
                # 只保存未过期的会话
                active_sessions = {}
                for session_id, session_data in self.memory_storage.items():
                    if time.time() - session_data["last_activity"] <= self.timeout_seconds:
                        active_sessions[session_id] = session_data
                
                with open(self.session_file, 'w', encoding='utf-8') as f:
                    json.dump(active_sessions, f, ensure_ascii=False, indent=2)
                logger.info(f"SESSION_SAVE: 保存会话数据成功，共 {len(active_sessions)} 个活跃会话")
            except Exception as e:
                logger.error(f"SESSION_SAVE: 保存会话数据失败: {e}")
    
    def create_session(self) -> str:
        """
        创建新会话
        
        Returns:
            str: 会话ID
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "start_time": time.time(),
            "last_activity": time.time(),
            "status": "PROCESSING",
            "messages": []
        }
        
        try:
            if redis_manager.is_available():
                # 使用Redis存储
                redis_manager.save_conversation(session_id, session_data)
                logger.info(f"SESSION_CREATE: 创建会话成功，session_id={session_id}")
            else:
                # 使用内存存储
                self.memory_storage[session_id] = session_data
                # 保存到文件
                self._save_sessions()
                logger.warning(f"SESSION_CREATE: 使用内存存储会话，session_id={session_id}")
        except Exception as e:
            logger.error(f"SESSION_CREATE: 创建会话失败: {e}，session_id={session_id}")
            raise
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Dict[str, Any]]: 会话信息，如果会话不存在或已超时返回None
        """
        session_data = None
        
        try:
            if redis_manager.is_available():
                # 从Redis获取
                session_data = redis_manager.load_conversation(session_id)
            else:
                # 从内存获取
                session_data = self.memory_storage.get(session_id)
            
            if not session_data:
                return None
            
            # 检查会话是否超时
            if time.time() - session_data["last_activity"] > self.timeout_seconds:
                self.end_session(session_id)
                logger.info(f"SESSION_TIMEOUT: 会话已过期，session_id={session_id}")
                return None
            
            return session_data
        except Exception as e:
            logger.error(f"SESSION_GET: 获取会话失败: {e}，session_id={session_id}")
            return None
    
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            role: 角色（user或assistant）
            content: 消息内容
        """
        try:
            session_data = self.get_session(session_id)
            if session_data:
                session_data["messages"].append({
                    "role": role,
                    "content": content,
                    "timestamp": time.time()
                })
                session_data["last_activity"] = time.time()
                
                if redis_manager.is_available():
                    redis_manager.save_conversation(session_id, session_data)
                else:
                    self.memory_storage[session_id] = session_data
                    # 保存到文件
                    self._save_sessions()
                
                logger.info(f"MESSAGE_ADD: 添加消息成功，session_id={session_id}, role={role}")
            else:
                logger.warning(f"MESSAGE_ADD: 会话不存在，无法添加消息，session_id={session_id}")
        except Exception as e:
            logger.error(f"MESSAGE_ADD: 添加消息失败: {e}，session_id={session_id}, role={role}")
    
    def add_history_messages(self, session_id: str, history_messages: list):
        """
        添加历史对话到会话
        
        Args:
            session_id: 会话ID
            history_messages: 历史对话列表
        """
        if not history_messages:
            return
            
        try:
            session_data = self.get_session(session_id)
            if session_data:
                # 保留现有消息，将历史对话添加到前面
                new_messages = []
                for msg in history_messages:
                    new_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", time.time())
                    })
                # 合并消息，去重
                existing_content = {msg["content"] for msg in session_data["messages"]}
                added_count = 0
                for msg in new_messages:
                    if msg["content"] not in existing_content:
                        session_data["messages"].insert(0, msg)
                        existing_content.add(msg["content"])
                        added_count += 1
                session_data["last_activity"] = time.time()
                
                if redis_manager.is_available():
                    redis_manager.save_conversation(session_id, session_data)
                else:
                    self.memory_storage[session_id] = session_data
                    # 保存到文件
                    self._save_sessions()
                
                logger.info(f"HISTORY_ADD: 添加历史消息成功，session_id={session_id}, count={added_count}")
            else:
                logger.warning(f"HISTORY_ADD: 会话不存在，无法添加历史消息，session_id={session_id}")
        except Exception as e:
            logger.error(f"HISTORY_ADD: 添加历史消息失败: {e}，session_id={session_id}")
    
    def end_session(self, session_id: str):
        """
        结束会话
        
        Args:
            session_id: 会话ID
        """
        # 会话状态现在只有 PROCESSING，不再设置为 FINISHED
        # 只需要从存储中删除会话
        if redis_manager.is_available():
            redis_manager.delete_conversation(session_id)
        else:
            if session_id in self.memory_storage:
                del self.memory_storage[session_id]
                # 保存到文件
                self._save_sessions()
        
        logger.info(f"SESSION_END: 结束会话，session_id={session_id}")
    
    def is_session_active(self, session_id: str) -> bool:
        """
        检查会话是否活跃
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 会话是否活跃
        """
        session = self.get_session(session_id)
        return session is not None
    
    def get_session_messages(self, session_id: str) -> list:
        """
        获取会话消息历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            list: 消息历史
        """
        session = self.get_session(session_id)
        return session["messages"] if session else []
    



# 单例实例
_conversation_manager_instance = None

def create_conversation_manager() -> ConversationManager:
    """
    创建会话管理器实例（单例模式）
    
    Returns:
        ConversationManager: 会话管理器实例
    """
    global _conversation_manager_instance
    if _conversation_manager_instance is None:
        _conversation_manager_instance = ConversationManager()
    return _conversation_manager_instance
