from openai import OpenAI
from configs.config import settings
from utils.logger import logger


class LLMClient:
    """
    大语言模型客户端

    用于调用远程大模型 API 生成回答
    """

    def __init__(self):
        """
        初始化大模型客户端

        从配置中读取 API 密钥、地址和模型参数
        """
        self.api_key = settings.API_KEY
        self.api_url = settings.BASE_URL
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self.timeout = settings.LLM_TIMEOUT

        # 使用 OpenAI 原生客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=self.timeout * 1000  # OpenAI 客户端期望毫秒
        )
        logger.info(f"初始化大模型客户端：{self.model}")

    def generate(self, prompt):
        """
        生成回答

        Args:
            prompt: 提示词

        Returns:
            str: 生成的回答
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的审计知识库助手，回答要准确、简洁、专业。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return "抱歉，未获取到有效回答，请稍后再试"
        except Exception as e:
            logger.error(f"调用大模型 API 失败：{e}")
            return "抱歉，模型调用失败，请稍后再试"

    def generate_stream(self, prompt):
        """
        流式生成回答

        Args:
            prompt: 提示词

        Yields:
            str: 生成的文本片段
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的审计知识库助手，回答要准确、简洁、专业。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            yield ""
        except Exception as e:
            logger.error(f"流式调用大模型 API 失败：{e}")
            yield "抱歉，模型调用失败，请稍后再试"


_llm_client_instance = None

def create_llm_client():
    """
    创建大模型客户端实例（单例模式）

    Returns:
        LLMClient: 大模型客户端实例
    """
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance