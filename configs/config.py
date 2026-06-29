from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env（从项目根目录）
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """读取环境变量"""
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CALLBACK_URL: str = ""
    
    # LLM 配置
    BASE_URL: str = ""
    API_KEY: str = ""
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 30
    LLM_CONTEXT_LENGTH: int = 32768  # qwen3-32b上下文长度限制，默认32k
    
    # Embedding 配置
    EMBEDDING_API_URL: str = ""
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_TIMEOUT: int = 30
    
    # Rerank 配置
    RERANK_API_URL: str = ""
    RERANK_MODEL_NAME: str = "rerank-english-v2.0"
    RERANK_TIMEOUT: int = 30
    
    # 流式输出配置
    STREAM_CHUNK_SIZE: int = 50
    
    # Milvus 向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_COLLECTION_NAME: str = "documents"
    MILVUS_DIMENSION: int = 1536
    
    # Minio 对象存储配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = ""  # 从 .env 注入，勿在此填写真实凭证
    MINIO_SECRET_KEY: str = ""  # 从 .env 注入，勿在此填写真实凭证
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_SECURE: bool = False
    
    # Redis 缓存配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_DECODE_RESPONSES: bool = True
    REDIS_TIMEOUT: int = 30
    CONVERSATION_TIMEOUT: int = 1800  # 会话超时时间（秒），默认 30 分钟
    
    # 高频问题缓存配置
    QUESTION_CACHE_DB: int = 3
    QUESTION_CACHE_SIMILARITY_THRESHOLD: float = 0.9
    
    # 意图识别配置
    INTENT_CONFIDENCE_THRESHOLD: float = 0.7
    
    # 分块策略配置
    CHUNKING_STRATEGY: str = "generic"  # 默认为通用分块
    
    # 索引方式配置
    RETRIEVAL_MODE: str = "hybrid"  # 检索方式：vector, keyword, hybrid
    
    # 重召回配置
    ENABLE_RE_RANKING: bool = True  # 是否启用重召回
    
    # 上下文压缩配置
    MAX_CONTEXT_TOKENS: int = 8000  # 最大上下文token数
    ENABLE_CONTEXT_COMPRESSION: bool = True  # 是否启用上下文压缩
    
    # Tika Server 配置
    TIKA_SERVER_URL: str = ""  # Tika Server地址，从 .env 注入，勿在此填写内网地址
    
    # PDF处理配置
    PDF_PROCESSOR_MODE: str = "auto"  # PDF处理模式：auto, digital, scanned
    ENABLE_OCR: bool = False  # 是否启用OCR功能
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
