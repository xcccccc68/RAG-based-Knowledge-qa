# 意图识别模块

import json
from configs.config import settings
from openai import OpenAI
from utils.context_manager import create_context_manager
from utils.logger import logger


def recognize_intent_with_llm(user_query: str, session_messages: list = None) -> dict:
    """
    使用LLM识别用户意图
    """
    context_manager = create_context_manager()
    context = context_manager.build_recent_context(session_messages, max_messages=3)
    
    # 使用prompt_manager构建意图识别提示词
    prompt = f"""请分析以下用户输入的意图，判断用户是在询问"审计知识问答"相关内容，还是其他内容。

【对话历史】
{context}

【用户输入】
{user_query}

【意图分类】
1. KBQA - 审计知识问答相关，例如：
   - 查询审计相关的知识
   - 询问审计流程、规范、标准
   - 查询审计案例、经验
   - 询问审计术语、定义
   - 等等与审计知识库相关的问答

2. OTHER - 其他，例如：
   - 你好/再见等问候
   - 今天天气怎么样
   - 讲个笑话
   - 谢谢/不客气
   - 等等与审计知识无关的对话

【输出格式】
请直接返回JSON格式，不要其他解释：
{{
    "intent": "KBQA" 或 "OTHER",
    "confidence": 0.0-1.0之间的置信度,
    "reason": "判断理由"
}}

【你的输出】"""

    try:
        client = OpenAI(api_key=settings.API_KEY, base_url=settings.BASE_URL)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=200,
            extra_body={"enable_thinking": False} 
        )
        
        content = response.choices[0].message.content.strip()
        
        # 尝试提取JSON部分
        if "{" in content and "}" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
            parsed = json.loads(json_str)
            return {
                "intent": parsed.get("intent", "OTHER"),
                "confidence": float(parsed.get("confidence", 0.7)),
                "reason": parsed.get("reason", "")
            }
        else:
            # 无法解析，默认走知识问答
            return {"intent": "KBQA", "confidence": 0.7, "reason": "解析失败，默认走知识问答"}
            
    except Exception as e:
        # 异常时默认走知识问答
        logger.error(f"[IntentRecognition] API异常: {str(e)}")
        return {"intent": "KBQA", "confidence": 0.7, "reason": f"识别异常: {str(e)}"}


def should_create_new_session(session_messages: list, new_query: str) -> bool:
    """
    判断是否需要创建新的会话
    
    Args:
        session_messages: 当前会话的消息历史
        new_query: 新的用户查询
    
    Returns:
        bool: 是否需要创建新会话
    """
    # 使用意图识别判断新查询的意图，传入会话历史
    intent_result = recognize_intent_with_llm(new_query, session_messages)
    
    # 如果意图不是知识问答，需要创建新会话
    if intent_result["intent"] != "KBQA":
        return True
    
    # 如果置信度低于阈值，需要创建新会话
    if intent_result["confidence"] < settings.INTENT_CONFIDENCE_THRESHOLD:
        return True
    
    # 如果会话为空，不需要创建新会话
    if not session_messages:
        return False
    
    return False
