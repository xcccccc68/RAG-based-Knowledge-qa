# 提示词管理模块

from typing import Dict, List
from utils.logger import logger
from utils.security_guard import create_security_guard

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class PromptManager:
    """
    提示词管理器，用于管理各种场景的提示词模板
    """
    
    def __init__(self):
        """
        初始化提示词管理器
        """
        logger.info("初始化提示词管理器")
        
        # 创建提示词模板
        self.prompts = {
            # 系统提示词
            "system": self._create_system_prompt(),
            # 对话提示词
            "conversation": self._create_conversation_prompt(),
            # 工具使用提示词
            "tool": self._create_tool_prompt(),
            # 上下文提示词
            "context": self._create_context_prompt(),
            # 错误处理提示词
            "error": self._create_error_prompt(),
            # 总结提示词
            "summary": self._create_summary_prompt()
        }
        
        # 创建聊天提示词模板
        self.chat_prompt_template = self._create_chat_prompt_template()
        
        self.sensitive_audit = create_security_guard()
        logger.info(f"提示词管理器初始化成功，加载了{len(self.prompts)}个提示词模板")
    
    def _create_system_prompt(self) -> PromptTemplate:
        """
        创建系统提示词模板
        
        Returns:
            PromptTemplate: 系统提示词模板
        """
        template = """你是一个专业的审计知识库问答助手，你的名字叫审计助手。

你的职责是：
1. 基于用户的问题和提供的知识库信息，给出准确、专业的回答
2. 严格按照知识库中的信息回答问题，不要编造信息
3. 回答要简洁明了，避免冗长的解释
4. 如果知识库中没有相关信息，要明确告知用户
5. 保持专业、友好的语气

你需要：
- 分析用户的问题，理解其意图
- 从知识库中提取相关信息
- 基于提取的信息生成准确的回答
- 提供结构化的回答，使用适当的标题和列表
- 引用相关的知识库文档（如果有）

禁止：
- 编造不存在的信息
- 回答与审计无关的问题
- 使用不专业的语言或语气
- 泄露敏感信息
"""
        return PromptTemplate(template=template, input_variables=[])
    
    def _create_conversation_prompt(self) -> PromptTemplate:
        """
        创建对话提示词模板
        
        Returns:
            PromptTemplate: 对话提示词模板
        """
        template = """
对话历史：
{history}

当前问题：{question}

请基于对话历史和当前问题，给出专业的回答。
"""
        return PromptTemplate(template=template, input_variables=["history", "question"])
    
    def _create_tool_prompt(self) -> PromptTemplate:
        """
        创建工具使用提示词模板
        
        Returns:
            PromptTemplate: 工具使用提示词模板
        """
        template = """
你可以使用以下工具来获取更多信息：

1. 知识库检索工具：用于从审计知识库中检索相关信息
2. 文档分析工具：用于分析上传的审计文档
3. 数据查询工具：用于查询审计相关的数据库信息

请根据用户的问题，选择合适的工具获取必要的信息，然后给出完整的回答。
"""
        return PromptTemplate(template=template, input_variables=[])
    
    def _create_context_prompt(self) -> PromptTemplate:
        """
        创建上下文提示词模板
        
        Returns:
            PromptTemplate: 上下文提示词模板
        """
        template = """
以下是上下文信息：

{context}

请基于上述上下文信息，回答用户的问题。
"""
        return PromptTemplate(template=template, input_variables=["context"])
    
    def _create_error_prompt(self) -> PromptTemplate:
        """
        创建错误处理提示词模板
        
        Returns:
            PromptTemplate: 错误处理提示词模板
        """
        template = """
遇到以下错误：
{error_message}

请：
1. 保持冷静，不要慌张
2. 向用户解释发生了什么错误
3. 提供可能的解决方案
4. 询问用户是否需要进一步的帮助
"""
        return PromptTemplate(template=template, input_variables=["error_message"])
    
    def _create_summary_prompt(self) -> PromptTemplate:
        """
        创建总结提示词模板
        
        Returns:
            PromptTemplate: 总结提示词模板
        """
        template = """
请根据以下对话内容，生成一个简洁的对话主题：

用户问题：{user_input}
AI回答：{ai_output}

要求：
1. 主题要简洁明了，不超过10个字
2. 能准确反映对话的核心内容
3. 使用专业的审计术语
4. 避免使用疑问句
5. 直接返回主题内容，不要添加任何前缀
6. 每次生成的主题要有所不同，即使输入内容相同
7. 可以从不同角度（如问题类型、涉及领域、解决方案等）生成主题
8. 尝试使用不同的专业术语和表达方式
"""
        return PromptTemplate(template=template, input_variables=["user_input", "ai_output"])
    
    def _create_chat_prompt_template(self) -> ChatPromptTemplate:
        """
        创建聊天提示词模板
        
        Returns:
            ChatPromptTemplate: 聊天提示词模板
        """
        system_template = self._create_system_prompt().template
        human_template = """
对话历史：
{history}

知识库信息：
{context}

用户问题：
{question}

请基于上述信息，给出专业的回答。
"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        获取指定类型的提示词
        
        Args:
            prompt_type: 提示词类型
            **kwargs: 提示词参数
            
        Returns:
            str: 填充后的提示词
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        prompt_template = self.prompts[prompt_type]
        return prompt_template.format(**kwargs)
    
    def generate_summary_prompt(self, user_input: str, ai_output: str) -> str:
        """
        生成总结提示词
        
        Args:
            user_input: 用户输入
            ai_output: AI输出
            
        Returns:
            str: 总结提示词
        """
        return self.get_prompt("summary", user_input=user_input, ai_output=ai_output)
    
# 单例实例
_prompt_manager_instance = None

def create_prompt_manager() -> PromptManager:
    """
    创建提示词管理器实例（单例模式）
    
    Returns:
        PromptManager: 提示词管理器实例
    """
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance
