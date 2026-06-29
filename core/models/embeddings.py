import time
from openai import OpenAI
from configs.config import settings
from utils.logger import logger


class APIEmbeddings:
    """
    API词嵌入模型封装类
    
    用于调用远程词嵌入API将文本转换为向量
    """
    
    def __init__(self):
        """
        初始化API词嵌入模型

        从配置中读取API地址、模型名称和超时时间
        """
        self.api_key = settings.API_KEY
        self.api_url = settings.EMBEDDING_API_URL
        self.model = settings.EMBEDDING_MODEL_NAME
        self.timeout = settings.EMBEDDING_TIMEOUT
        self.max_retries = 3
        self.retry_delay = 2
        
        # 使用 OpenAI 原生客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=self.timeout
        )
        logger.info(f"初始化API词嵌入模型: {self.model}, 最大重试次数: {self.max_retries}")

    def embed_documents(self, texts):
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            list: 嵌入向量列表
        """
        if not texts:
            return []

        total_texts = len(texts)
        logger.info(f"开始批量嵌入 {total_texts} 个文档")
        
        # 分批处理，每批最多 10 个文本，避免 API 服务器过载
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_texts + batch_size - 1) // batch_size
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=self.model
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    # 移除批次详细日志，只显示最终结果
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"批次 {batch_num} 嵌入失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"批次 {batch_num} 嵌入失败，已重试 {self.max_retries} 次：{e}")
                        raise Exception(f"批次 {batch_num} 嵌入失败：{e}")
        
        # 只显示最终统计结果
        logger.info(f"嵌入向量生成完成：共处理{total_texts}个文本，生成{len(all_embeddings)}个嵌入向量")
        
        return all_embeddings

    def embed_query(self, text):
        """
        嵌入单个查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            list: 嵌入向量
        """
        if not text or not text.strip():
            text = " "
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"查询嵌入失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"查询嵌入失败，已重试 {self.max_retries} 次：{e}")
                    raise Exception(f"查询嵌入失败：{e}")


# 单例实例
_api_embeddings_instance = None

def create_api_embeddings():
    """
    创建API词嵌入模型实例（单例模式）
    
    Returns:
        APIEmbeddings: API词嵌入模型实例
    """
    global _api_embeddings_instance
    if _api_embeddings_instance is None:
        _api_embeddings_instance = APIEmbeddings()
    return _api_embeddings_instance
