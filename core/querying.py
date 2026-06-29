from configs.config import settings
from core.models.llm import create_llm_client
from utils.context_manager import create_context_manager
from helps.context_compression import create_context_compression
from utils.logger import logger
from helps.retriever import create_retriever
from helps.ranker import create_ranker
from utils.security_guard import create_security_guard
from helps.question_cache import create_question_cache
from typing import List, Optional, Generator, Dict, Any
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGSystem:
    """
    RAG（检索增强生成）系统核心类
    """

    def __init__(self):
        """
        初始化 RAG 系统

        创建检索器、重排序器、大模型客户端和上下文管理器
        """
        self.retriever = create_retriever(retrieval_mode=settings.RETRIEVAL_MODE)
        self.ranker = create_ranker()
        self.llm_client = create_llm_client()
        self.context_manager = create_context_manager()
        self.context_compressor = create_context_compression()
        self.sensitive_audit = create_security_guard()
        self.question_cache = create_question_cache()
        self.retrieval_cache = {}

        self.rag_chain = self._create_lcel_rag_chain()

        logger.info("RAG 系统初始化成功")

    def _create_lcel_rag_chain(self):
        """
        创建LCEL RAG链

        Returns:
            Runnable: LCEL RAG链
        """
        prompt = ChatPromptTemplate.from_template("""
        你是一个专业的审计知识库助手，回答要准确、简洁、专业。

        对话历史：
        {history}

        知识库信息：
        {context}

        用户问题：
        {question}

        请基于上述信息，给出专业的回答。
        """)

        def retrieve_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs["question"]

            if query in self.retrieval_cache:
                local_context, _, _ = self.retrieval_cache[query]
                return {"context": local_context}

            docs = self.retriever.invoke(query)

            if not docs:
                return {"context": ""}

            if settings.ENABLE_RE_RANKING:
                reranked_docs = self.ranker.rerank_documents(query, docs, top_k=5)
            else:
                reranked_docs = docs[:5]

            local_context = "\n".join([doc.page_content for doc in reranked_docs])

            # 应用上下文压缩
            compressed_context = self.context_compressor.compress_context(local_context, query)

            doc_names_set = set()
            for doc in reranked_docs:
                source = doc.metadata.get('source', '')
                if source:
                    doc_name = os.path.basename(source)
                    doc_names_set.add(doc_name)
            doc_names = list(doc_names_set)
            self.retrieval_cache[query] = (compressed_context, len(doc_names), doc_names)

            return {"context": compressed_context}

        def format_history(inputs: Dict[str, Any]) -> Dict[str, Any]:
            history_messages = inputs.get("history_messages", [])
            history = ""
            if history_messages:
                for msg in history_messages:
                    if msg.get('role') == 'user':
                        history += f"用户: {msg.get('content', '')}\n"
                    elif msg.get('role') == 'assistant':
                        history += f"助手: {msg.get('content', '')}\n"
            return {"history": history}

        def safe_llm_invoke(inputs):
            try:
                # 将langchain的messages转换为OpenAI格式
                # inputs是ChatPromptTemplate的输出，应该是messages列表
                if isinstance(inputs, list):
                    # 转换为OpenAI格式
                    messages = []
                    for msg in inputs:
                        if hasattr(msg, 'type') and hasattr(msg, 'content'):
                            # langchain的Message对象
                            role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
                            messages.append({"role": role, "content": msg.content})
                        elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            # 已经是OpenAI格式
                            messages.append(msg)
                    # 调用OpenAI API
                    response = self.llm_client.client.chat.completions.create(
                        model=self.llm_client.model,
                        messages=messages,
                        max_tokens=self.llm_client.max_tokens,
                        temperature=self.llm_client.temperature
                    )
                    if response and response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content
                return "抱歉，未获取到有效回答，请稍后再试"
            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
                return "抱歉，模型调用失败，请稍后再试"

        rag_chain = (
            RunnablePassthrough()
            | RunnableParallel(
                question=lambda x: x["question"],
                history=format_history,
                context=retrieve_docs
            )
            | prompt
            | safe_llm_invoke
            | StrOutputParser()
        )

        return rag_chain

    def get_answer(self, question: str, session_id: Optional[str] = None, history_messages: Optional[list] = None) -> str:
        """
        获取回答

        Args:
            question: 问题
            session_id: 会话 ID
            history_messages: 历史对话列表（可选）

        Returns:
            str: 回答
        """
        try:
            cached_result = self.question_cache.search_similar_question(question)
            if cached_result:
                logger.info(f"使用缓存答案：'{cached_result['question']}' (相似度：{cached_result['similarity']:.4f})")
                return cached_result['answer']
            
            answer = self.rag_chain.invoke({
                "question": question,
                "history_messages": history_messages
            })

            self.question_cache.add_question(question, answer)

            # 模型输出敏感词审计：分析→误报处理→过滤→返回
            has_sensitive, sensitive_words = self.sensitive_audit.check_sensitive_content(answer, "模型输出")
            if has_sensitive:
                # 误报处理：如果语义分析认为是技术术语误报，不进行过滤
                context_audit_result = self.sensitive_audit._context_based_audit(answer, sensitive_words, "模型输出")
                if not context_audit_result:
                    # 误报处理：将技术术语加入白名单
                    self.sensitive_audit.update_whitelist(sensitive_words)
                    logger.info(f"模型输出误报处理：将检测到的技术词汇加入白名单：{', '.join(sensitive_words)}")
                    return answer  # 返回原始回答，不进行过滤
                
                # 确认敏感，进行过滤
                filtered_answer = self.sensitive_audit.filter_sensitive_content(answer)
                logger.warning(f"模型输出包含敏感词：{', '.join(sensitive_words)}，已使用*替换")
                return filtered_answer

            return answer
        except Exception as e:
            logger.error(f"获取回答失败：{e}")
            return "抱歉，处理失败，请稍后再试"

    def get_answer_stream(self, question: str, session_id: Optional[str] = None, history_messages: Optional[list] = None) -> Generator[str, None, None]:
        """
        流式获取回答

        Args:
            question: 问题
            session_id: 会话ID
            history_messages: 历史对话列表（可选）

        Yields:
            str: 回答的文本片段
        """
        try:
            cached_result = self.question_cache.search_similar_question(question)
            if cached_result:
                logger.info(f"使用缓存答案：'{cached_result['question']}' (相似度：{cached_result['similarity']:.4f})")
                answer = cached_result['answer']
                for i in range(0, len(answer), 50):
                    yield answer[i:i+50]
                return
            
            history = ""
            if history_messages:
                for msg in history_messages:
                    if msg.get('role') == 'user':
                        history += f"用户: {msg.get('content', '')}\n"
                    elif msg.get('role') == 'assistant':
                        history += f"助手: {msg.get('content', '')}\n"

            local_context, _, _ = self._retrieve_from_vector_store(question)

            prompt = f"""
            你是一个专业的审计知识库助手，回答要准确、简洁、专业。

            对话历史：
            {history}

            知识库信息：
            {local_context}

            用户问题：
            {question}

            请基于上述信息，给出专业的回答。
            """

            full_answer = ""
            for chunk in self.llm_client.generate_stream(prompt):
                full_answer += chunk

            self.question_cache.add_question(question, full_answer)

            # 模型输出敏感词审计：分析→误报处理→过滤→返回
            has_sensitive, sensitive_words = self.sensitive_audit.check_sensitive_content(full_answer, "模型输出")
            if has_sensitive:
                # 误报处理：如果语义分析认为是技术术语误报，不进行过滤
                context_audit_result = self.sensitive_audit._context_based_audit(full_answer, sensitive_words, "模型输出")
                if not context_audit_result:
                    # 误报处理：将技术术语加入白名单
                    self.sensitive_audit.update_whitelist(sensitive_words)
                    logger.info(f"模型输出误报处理：将检测到的技术词汇加入白名单: {', '.join(sensitive_words)}")
                    # 返回原始回答，不进行过滤
                    for i in range(0, len(full_answer), 50):
                        yield full_answer[i:i+50]
                    return
                
                # 确认敏感，进行过滤
                filtered_answer = self.sensitive_audit.filter_sensitive_content(full_answer)
                logger.warning(f"模型输出包含敏感词: {', '.join(sensitive_words)}，已使用*替换")
                for i in range(0, len(filtered_answer), 50):
                    yield filtered_answer[i:i+50]
                return

            for i in range(0, len(full_answer), 50):
                yield full_answer[i:i+50]
            return
        except Exception as e:
            logger.error(f"流式获取回答失败: {e}")
            yield "抱歉，处理失败，请稍后再试"

    def _retrieve_from_vector_store(self, query: str) -> tuple[str, int, List[str]]:
        """
        从向量存储检索相关文档

        Args:
            query: 查询文本

        Returns:
            tuple[str, int, List[str]]: (上下文文本, 文档数量, 文档名称列表)
        """
        if query in self.retrieval_cache:
            return self.retrieval_cache[query]

        try:
            docs = self.retriever.invoke(query)

            if not docs:
                result = ("", 0, [])
                self.retrieval_cache[query] = result
                return result

            if settings.ENABLE_RE_RANKING:
                reranked_docs = self.ranker.rerank_documents(query, docs, top_k=5)
            else:
                reranked_docs = docs[:5]

            local_context = "\n".join([doc.page_content for doc in reranked_docs])
            doc_names_set = set()
            for doc in reranked_docs:
                source = doc.metadata.get('source', '')
                if source:
                    doc_name = os.path.basename(source)
                    doc_names_set.add(doc_name)
            doc_names = list(doc_names_set)

            result = (local_context, len(doc_names), doc_names)
            self.retrieval_cache[query] = result
            return result
        except Exception as e:
            logger.error(f"从向量存储检索失败: {e}")
            result = ("", 0, [])
            self.retrieval_cache[query] = result
            return result

    def get_reference_documents(self, question: str) -> List[str]:
        """
        获取引用文档列表

        Args:
            question: 问题

        Returns:
            List[str]: 文档名称列表
        """
        _, _, doc_names = self._retrieve_from_vector_store(question)
        return doc_names


def create_rag_system():
    """
    创建RAG系统实例

    Returns:
        RAGSystem: RAG系统实例
    """
    return RAGSystem()