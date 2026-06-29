from typing import Optional
from utils.conversation_manager import create_conversation_manager
from helps.context_compression import create_context_compression
from configs.config import settings
from utils.logger import logger


class ContextManager:
    """
    上下文管理器
    
    负责构建和管理对话上下文
    """
    
    def __init__(self):
        """
        初始化上下文管理器
        """
        # 直接使用全局的会话管理器实例，而不是重新创建
        self.conversation_manager = create_conversation_manager()
        # 初始化上下文压缩器
        self.context_compressor = create_context_compression()
        logger.info("上下文管理器初始化成功")
    
    def build_context(self, session_id: Optional[str] = None, history_messages: Optional[list] = None, query: Optional[str] = None) -> str:
        """
        构建对话上下文
        
        Args:
            session_id: 会话ID
            history_messages: 历史对话列表（可选）
            query: 查询文本（用于上下文压缩）
            
        Returns:
            str: 对话上下文字符串
        """
        context_parts = []
        
        # 处理传入的历史对话
        if history_messages:
            history_context = self._build_context_from_history(history_messages)
            if history_context:
                context_parts.append("以下是历史对话：")
                context_parts.append(history_context)
                context_parts.append("")
        
        # 处理会话中的消息
        if session_id:
            session_messages = self.conversation_manager.get_session_messages(session_id)
            if session_messages:
                session_context = self._build_context_from_history(session_messages)
                if session_context:
                    context_parts.append("以下是会话历史：")
                    context_parts.append(session_context)
                    context_parts.append("")
        
        full_context = "\n".join(context_parts)
        
        # 如果启用了上下文压缩且提供了查询，则进行压缩
        if query and settings.ENABLE_CONTEXT_COMPRESSION:
            compressed_context = self.context_compressor.compress_context(full_context, query)
            return compressed_context
        
        return full_context
    
    def _build_context_from_history(self, messages: list) -> str:
        """
        从历史对话构建上下文
        
        Args:
            messages: 历史对话列表
            
        Returns:
            str: 对话上下文字符串
        """
        context_parts = []
        for msg in messages:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            context_parts.append(f"{role}：{content}")
        
        return "\n".join(context_parts)
    
    def build_recent_context(self, messages: list, max_messages: int = 3) -> str:
        """
        从最近的消息构建上下文（用于意图识别等场景）
        
        Args:
            messages: 消息列表
            max_messages: 最大消息数量，默认取最近3条
            
        Returns:
            str: 对话上下文字符串
        """
        if not messages:
            return ""
        
        recent_messages = messages[-max_messages:]
        context_parts = []
        for msg in recent_messages:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            context_parts.append(f"{role}：{content}")
        
        return "\n".join(context_parts)
    
    def add_history_to_session(self, session_id: str, history_messages: list):
        """
        添加历史对话到会话
        
        Args:
            session_id: 会话ID
            history_messages: 历史对话列表
        """
        self.conversation_manager.add_history_messages(session_id, history_messages)
    
    def get_session_context(self, session_id: str) -> str:
        """
        获取会话的上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            str: 对话上下文字符串
        """
        return self.build_context(session_id)


# 单例实例
_context_manager_instance = None

def create_context_manager() -> ContextManager:
    """
    创建上下文管理器实例（单例模式）
    
    Returns:
        ContextManager: 上下文管理器实例
    """
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ContextManager()
    return _context_manager_instance
