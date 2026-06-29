import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import uuid

from core.querying import create_rag_system
from core.indexing import create_document_processor
from core.models.llm import create_llm_client
from utils.conversation_manager import create_conversation_manager
from utils.context_manager import create_context_manager
from helps.intent_recognition import should_create_new_session
from utils.logger import log_qa_interaction, logger
from utils.prompt_manager import create_prompt_manager
from utils.security_guard import create_security_guard
from configs.config import settings

app = FastAPI(title="审计知识库问答系统")

rag_system = None
conversation_manager = None
context_manager = None
llm_client = None
prompt_manager = None
sensitive_audit = None


def send_java_callback(request_id: str, status: int, message: str):
    """
    发送Java回调接口
    
    Args:
        request_id: 请求ID
        status: 状态码（0:成功, 1:失败）
        message: 消息内容
    """
    java_callback_url = settings.CALLBACK_URL
    try:
        callback_data = {
            "id": request_id,
            "status": status,
            "message": message
        }
        response = requests.post(
            java_callback_url, 
            json=callback_data, 
            timeout=30
        )
        logger.info(f"Java回调成功: {response.status_code}")
        logger.info(f"Java回调响应: {response.text}")
    except Exception as e:
        logger.error(f"Java回调失败: {e}")


class QuestionRequest(BaseModel):
    """
    问答请求模型
    """
    id: str  # 请求id
    user_id: str | None = None  # 用户id
    user_input: str  # 用户输入文本
    chat_id: str | None = None  # 对话id
    chat_topic: str | None = None  # 对话主题
    session_id: str | None = None  # 会话id
    skill: str  # AI技能名称，固定取值: knowledge_qa
    stream: bool = True  # 是否流式输出，默认取值: true
    history_messages: list | None = None  # 历史对话列表


class FileItem(BaseModel):
    """
    文件项模型
    """
    file_id: str  # 文档id，全局唯一
    file_path: str  # 文档地址
    file_label: str  # 文档是否需要使用AI解析，枚举值：NEED_AI, NO_AI


class DocumentProcessRequest(BaseModel):
    """
    文档处理请求模型
    """
    id: str  # 请求id
    data: list[FileItem]  # 文件列表


def generate_chat_topic(user_input: str, ai_output: str) -> str:
    """
    生成对话主题（使用流式调用避免 ModelScope API 的 bug）

    Args:
        user_input: 用户输入
        ai_output: AI 输出

    Returns:
        str: 生成的对话主题
    """
    try:
        summary_prompt = prompt_manager.generate_summary_prompt(
            user_input=user_input,
            ai_output=ai_output
        )
        
        chat_topic = ""
        stream = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=50,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chat_topic += chunk.choices[0].delta.content
        
        return chat_topic.strip() if chat_topic else "审计知识问答"
    except Exception as e:
        logger.error(f"生成对话主题失败: {e}")
        return "审计知识问答"


def process_qa_request(request: QuestionRequest, session_id: str):
    """
    处理问答请求的公共逻辑
    
    Args:
        request: 问答请求
        session_id: 会话ID
        
    Returns:
        dict: 处理结果，包含answer和reference
    """
    try:
        # 获取回答
        answer = rag_system.get_answer(request.user_input, session_id, request.history_messages if request.history_messages else None)
        
        # 获取引用文档
        reference_docs = rag_system.get_reference_documents(request.user_input)
        reference = reference_docs[0] if reference_docs else None
        
        # 记录系统回答
        conversation_manager.add_message(session_id, "assistant", answer)
        
        # 记录问答交互日志
        log_qa_interaction(request.user_id, session_id, request.user_input, answer)
        
        return {
            "answer": answer,
            "reference": reference
        }
    except Exception as e:
        logger.error(f"问答处理失败: {e}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event():
    """
    服务启动事件
    
    初始化RAG系统、会话管理器和上下文管理器
    """
    global rag_system
    global conversation_manager
    global context_manager
    global llm_client
    global prompt_manager
    global sensitive_audit
    rag_system = create_rag_system()
    conversation_manager = create_conversation_manager()
    context_manager = create_context_manager()
    llm_client = create_llm_client()
    prompt_manager = create_prompt_manager()
    sensitive_audit = create_security_guard()
    logger.info("服务启动成功，敏感词审计功能已初始化")


@app.post("/api/v1/log_audit/ai/kbqa/kb")
async def process_documents(request: DocumentProcessRequest):
    """
    知识文档预处理（同步）
    
    Args:
        request: 文档处理请求
        
    Returns:
        dict: 处理结果
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化")
    
    # 分离NEED_AI和NO_AI文件
    need_ai_files = []
    no_ai_files = []
    
    for file_item in request.data:
        if file_item.file_label == "NEED_AI":
            need_ai_files.append(file_item.file_path)
        elif file_item.file_label == "NO_AI":
            no_ai_files.append(file_item.file_path)
    
    try:
        document_processor = create_document_processor()
        
        # 处理NEED_AI文件
        need_ai_results = {"success": 0, "failed": 0, "total_chunks": 0, "errors": []}
        if need_ai_files:
            need_ai_results = document_processor.process_minio_documents(need_ai_files)
        
        # 处理NO_AI文件（删除）
        no_ai_results = {"success": 0, "failed": 0, "errors": []}
        if no_ai_files:
            no_ai_results = document_processor.remove_minio_documents(no_ai_files)
        
        # 计算总失败数
        total_failed = need_ai_results.get("failed", 0) + no_ai_results.get("failed", 0)
        
        if total_failed > 0:
            # 构建详细错误信息
            error_messages = []
            # 添加NEED_AI文件的错误信息
            if "errors" in need_ai_results:
                for error in need_ai_results["errors"]:
                    error_messages.append(f"{os.path.basename(error['file'])}: {error['error_type']} - {error['error_message']}")
            # 添加NO_AI文件的错误信息
            if "errors" in no_ai_results:
                for error in no_ai_results["errors"]:
                    error_messages.append(f"{os.path.basename(error['file'])}: {error['error_type']} - {error['error_message']}")
            
            if error_messages:
                error_msg = "；".join(error_messages)
                message = f"部分文件处理失败，成功{need_ai_results.get('success', 0) + no_ai_results.get('success', 0)}个，失败{total_failed}个。失败原因：{error_msg}"
            else:
                message = f"部分文件处理失败，成功{need_ai_results.get('success', 0) + no_ai_results.get('success', 0)}个，失败{total_failed}个"
            
            send_java_callback(request.id, 1, message)
            
            return {
                "id": request.id,
                "code": 200,
                "message": message
            }
        else:
            # 全部成功
            message = f"处理成功，共处理{need_ai_results.get('success', 0) + no_ai_results.get('success', 0)}个文件"
            
            send_java_callback(request.id, 0, message)
            
            return {
                "id": request.id,
                "code": 200,
                "message": message
            }
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        
        error_message = f"处理失败: {str(e)}"
        send_java_callback(request.id, 1, error_message)
        
        return {
            "id": request.id,
            "code": 500,
            "message": error_message
        }



@app.post("/api/v1/log_audit/ai/kbqa/qa")
async def ask_question_api(request: QuestionRequest, response: Response):
    """
    知识问答（同步/流式）
    
    Args:
        request: 问答请求
        response: 响应对象
        
    Returns:
        StreamingResponse | dict: 流式响应或一次性响应
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 系统未初始化")
    
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="会话管理器未初始化")
    
    if not sensitive_audit:
        raise HTTPException(status_code=503, detail="敏感词审计器未初始化")
    
    # 检查用户输入是否包含敏感词
    logger.info(f"开始检查用户输入: {request.user_input}")
    # 传递上下文信息（如会话ID、用户ID等）
    context = f"session_id={request.session_id}, user_id={request.user_id}"
    has_sensitive, sensitive_words = sensitive_audit.check_sensitive_content(request.user_input, context)
    logger.info(f"敏感词检查结果: has_sensitive={has_sensitive}, sensitive_words={sensitive_words}")
    if has_sensitive:
        session_id = request.session_id or str(uuid.uuid4())
        chat_id = request.chat_id or str(uuid.uuid4())
        chat_topic = request.chat_topic or ""
        return {
            "id": request.id,
            "code": 400,
            "message": f"输入包含敏感词，请修改后重试: {', '.join(sensitive_words)}",
            "chat_id": chat_id,
            "chat_topic": chat_topic,
            "session_id": session_id,
            "data": {
                "final_answer": "",
                "status": "FAILED",
                "reference": None
            }
        }
    
    # 检查用户输入是否包含提示词注入攻击
    has_injection, injection_patterns = sensitive_audit.check_prompt_injection(request.user_input)
    if has_injection:
        session_id = request.session_id or str(uuid.uuid4())
        chat_id = request.chat_id or str(uuid.uuid4())
        chat_topic = request.chat_topic or ""
        return {
            "id": request.id,
            "code": 400,
            "message": "输入包含不安全内容，请修改后重试",
            "chat_id": chat_id,
            "chat_topic": chat_topic,
            "session_id": session_id,
            "data": {
                "final_answer": "",
                "status": "FAILED",
                "reference": None
            }
        }
    
    # 验证skill参数
    if request.skill != "knowledge_qa":
        session_id = request.session_id or str(uuid.uuid4())
        chat_id = request.chat_id or str(uuid.uuid4())
        chat_topic = request.chat_topic or ""
        return {
            "id": request.id,
            "code": 400,
            "message": "无效的技能名称，固定取值为: knowledge_qa",
            "chat_id": chat_id,
            "chat_topic": chat_topic,
            "session_id": session_id,
            "data": {
                "final_answer": "",
                "status": "FAILED",
                "reference": None
            }
        }
    
    # 处理chat_id
    chat_id = request.chat_id or str(uuid.uuid4())
    chat_topic = ""  # 初始化为空
    
    # 处理session_id和历史对话
    # 先判断session_id是否过期
    if request.session_id and not conversation_manager.is_session_active(request.session_id):
        # 会话过期，直接返回419错误
        session_id = str(uuid.uuid4())
        # 会话过期时，删除旧会话信息
        conversation_manager.end_session(request.session_id)
        
        # 当chat_id为空时生成chat_topic
        if not request.chat_id:
            chat_topic = generate_chat_topic(
                user_input=request.user_input,
                ai_output="会话已过期，请重新开始对话"
            )
        else:
            # chat_id不为空时，保持topic为空
            chat_topic = ""
        
        async def stream_generator():
            # 构造419响应
            stream_response = {
                "id": request.id,
                "code": 419,
                "message": "session_id已过期",
                "chat_id": chat_id,
                "chat_topic": chat_topic,
                "session_id": session_id,
                "data": {
                    "final_answer": "",
                    "status": "PROCESSING",
                    "reference": None
                }
            }
            yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"
            # 结束标记
            yield "data: [DONE]\n\n"
        
        # 设置响应头
        response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        
        return StreamingResponse(stream_generator())
    else:
        # 会话未过期或没有session_id，使用现有session_id或创建新会话
        session_id = request.session_id or conversation_manager.create_session()
    
    # 处理历史对话
    session_messages = []
    if request.history_messages:
        # 有历史会话字段，添加到会话并作为上下文
        context_manager.add_history_to_session(session_id, request.history_messages)
        session_messages = request.history_messages
    elif conversation_manager.is_session_active(session_id):
        # 没有历史会话字段但会话存在，使用会话中的历史对话
        session_messages = conversation_manager.get_session_messages(session_id)
    
    # 任何情况下都进行意图识别
    if session_messages:
        if should_create_new_session(session_messages, request.user_input):
            # 意图识别结果需要创建新会话
            session_id = conversation_manager.create_session()
            # 新会话的消息历史为空
            session_messages = []
    
    try:
        # 记录用户输入
        conversation_manager.add_message(session_id, "user", request.user_input)
        
        # 根据stream参数决定使用哪种输出方式
        if request.stream:
            # 流式输出
            async def stream_generator():
                # 初始化chat_topic
                local_chat_topic = chat_topic
                # 先获取引用文档
                reference_docs = rag_system.get_reference_documents(request.user_input)
                reference = reference_docs[0] if reference_docs else None
                
                # 累积完整回答
                full_answer = ""
                chunk_buffer = ""
                chunk_size = settings.STREAM_CHUNK_SIZE
                
                # 流式获取回答
                for chunk in rag_system.get_answer_stream(request.user_input, session_id, request.history_messages if request.history_messages else None):
                    full_answer += chunk
                    chunk_buffer += chunk
                    
                    # 当缓冲区达到chunk_size时输出
                    if len(chunk_buffer) >= chunk_size:
                        stream_response = {
                            "id": request.id,
                            "code": 200,
                            "message": "成功",
                            "chat_id": chat_id,
                            "chat_topic": local_chat_topic,
                            "session_id": session_id,
                            "data": {
                                "final_answer": full_answer,
                                "status": "PROCESSING",
                                "reference": None
                            }
                        }
                        yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"
                        chunk_buffer = ""
                
                # 输出剩余的内容
                if chunk_buffer:
                    stream_response = {
                        "id": request.id,
                        "code": 200,
                        "message": "成功",
                        "chat_id": chat_id,
                        "chat_topic": local_chat_topic,
                        "session_id": session_id,
                        "data": {
                            "final_answer": full_answer,
                            "status": "PROCESSING",
                            "reference": None
                        }
                    }
                    yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"
                
                # 记录系统回答和日志
                conversation_manager.add_message(session_id, "assistant", full_answer)
                log_qa_interaction(request.user_id, session_id, request.user_input, full_answer)
                
                # 当chat_id为空时生成chat_topic
                if not request.chat_id:
                    local_chat_topic = generate_chat_topic(
                        user_input=request.user_input,
                        ai_output=full_answer
                    )
                else:
                    # chat_id不为空时，保持topic为空
                    local_chat_topic = ""
                
                # 最后一个数据包
                final_response = {
                    "id": request.id,
                    "code": 200,
                    "message": "成功",
                    "chat_id": chat_id,
                    "chat_topic": local_chat_topic,
                    "session_id": session_id,
                    "data": {
                        "final_answer": full_answer,
                        "status": "PROCESSING",
                        "reference": reference
                    }
                }
                yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
                
                # 结束标记
                yield "data: [DONE]\n\n"
            
            # 设置响应头
            response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            
            return StreamingResponse(stream_generator())
        else:
            # 一次性输出
            try:
                result = process_qa_request(request, session_id)
                
                # 当chat_id为空时生成chat_topic
                if not request.chat_id:
                    chat_topic = generate_chat_topic(
                        user_input=request.user_input,
                        ai_output=result["answer"]
                    )
                else:
                    # chat_id不为空时，保持topic为空
                    chat_topic = ""
                
                return {
                    "id": request.id,
                    "code": 200,
                    "message": "成功",
                    "chat_id": chat_id,
                    "chat_topic": chat_topic,
                    "session_id": session_id,
                    "data": {
                        "final_answer": result["answer"],
                        "status": "PROCESSING",
                        "reference": result["reference"]
                    }
                }
            except Exception as e:
                logger.error(f"一次性输出处理失败: {e}", exc_info=True)
                return {
                    "id": request.id,
                    "code": 500,
                    "message": f"处理失败: {str(e)}",
                    "chat_id": chat_id,
                    "chat_topic": chat_topic,
                    "session_id": session_id,
                    "data": {
                        "final_answer": "抱歉，模型调用失败，请稍后再试",
                        "status": "FAILED",
                        "reference": None
                    }
                }
    except Exception as e:
        # 处理错误情况
        logger.error(f"问答处理异常: {e}", exc_info=True)
        # chat_id不为空时，保持topic为空
        if request.chat_id:
            chat_topic = ""
        return {
            "id": request.id,
            "code": 500,
            "message": f"处理失败: {str(e)}",
            "chat_id": chat_id,
            "chat_topic": chat_topic,
            "session_id": session_id,
            "data": {
                "final_answer": "抱歉，模型调用失败，请稍后再试",
                "status": "FAILED",
                "reference": None
            }
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT,)
