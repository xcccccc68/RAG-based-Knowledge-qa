from langchain_core.documents import Document
from langchain_milvus import Milvus
from core.models.embeddings import create_api_embeddings
from configs.config import settings
from typing import List
import json
from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection
from utils.logger import logger


class MilvusManager:
    """
    Milvus向量数据库管理器，用于存储和检索文档向量
    """
    
    def __init__(self):
        """
        初始化Milvus管理器
        
        创建词嵌入模型和向量存储
        """
        self.embeddings = create_api_embeddings()
        self.vector_store = None
        self._initialize_vector_store()

    def _create_custom_schema(self):
        """
        创建自定义的 Milvus 集合 schema
        
        所有字段都设置为 nullable，允许某些文本块不包含特定字段
        """
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=2048, nullable=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2048, nullable=True),
            FieldSchema(name="producer", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
            FieldSchema(name="creator", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
            FieldSchema(name="creationdate", dtype=DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
            FieldSchema(name="moddate", dtype=DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema(name="total_pages", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="page_label", dtype=DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema(name="element_type", dtype=DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
            FieldSchema(name="page", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema(name="parent_chunk_index", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="parent_index", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="chunk_index", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="table_data", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="is_table", dtype=DataType.BOOL, nullable=True),
            FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="is_image", dtype=DataType.BOOL, nullable=True),
        ]
        
        schema = CollectionSchema(fields=fields, description="审计知识库集合")
        return schema

    def _initialize_vector_store(self):
        """
        初始化向量存储
        
        连接Milvus，检查集合是否存在，不存在则创建
        """
        if self.vector_store is None:
            try:
                # 连接到 Milvus
                connections.connect(
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD
                )
                
                collection_name = settings.MILVUS_COLLECTION_NAME
                
                # 检查集合是否存在，不存在则创建
                if not utility.has_collection(collection_name):
                    schema = self._create_custom_schema()
                    Collection(name=collection_name, schema=schema, using='default')
                
                # 使用 langchain_milvus 连接到已存在的集合
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    connection_args={
                        "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                        "user": settings.MILVUS_USER,
                        "password": settings.MILVUS_PASSWORD
                    },
                    collection_name=collection_name,
                    drop_old=False,
                    auto_id=True
                )
                
                logger.info(f"Milvus 集合 {collection_name} 初始化成功")
                
            except Exception as e:
                raise Exception(f"Milvus 初始化失败: {e}")

    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 是否添加成功
        """
        try:
            if self.vector_store is None:
                self._initialize_vector_store()
            if not self.vector_store:
                return False
            
            # 处理文档元数据，将 table_data 等列表类型转换为 JSON 字符串
            for doc in documents:
                if 'table_data' in doc.metadata and isinstance(doc.metadata['table_data'], list):
                    doc.metadata['table_data'] = json.dumps(doc.metadata['table_data'], ensure_ascii=False)
                if 'image_base64' in doc.metadata and isinstance(doc.metadata['image_base64'], list):
                    doc.metadata['image_base64'] = json.dumps(doc.metadata['image_base64'], ensure_ascii=False)
            
            # 使用add_texts方法添加文档，Milvus会自动调用embedding_function生成向量
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"添加文档到 Milvus 失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, filter: dict = None) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回前k个结果
            filter: 过滤条件
            
        Returns:
            List[Document]: 搜索结果
        """
        try:
            if self.vector_store is None:
                self._initialize_vector_store()
            if not self.vector_store:
                return []
            
            if filter:
                expr = " && ".join([f"{key} == '{value}'" for key, value in filter.items()])
                return self.vector_store.similarity_search(query, k=k, expr=expr)
            else:
                return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def delete_documents(self, ids: List[str] = None, filter: dict = None) -> bool:
        """
        删除文档
        
        Args:
            ids: 文档ID列表
            filter: 过滤条件
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if self.vector_store is None:
                return False
            if ids:
                self.vector_store.delete(ids)
            elif filter:
                expr = " && ".join([f"{key} == '{value}'" for key, value in filter.items()])
                self.vector_store.delete(expr=expr)
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False


    def check_document_exists(self, source: str) -> bool:
        """
        检查指定source的文档是否已存在
        
        Args:
            source: 文档来源（文件名）
            
        Returns:
            bool: 是否已存在
        """
        try:
            if self.vector_store is None:
                self._initialize_vector_store()
            if not self.vector_store:
                return False
            
            # 直接使用Milvus的query接口按metadata过滤，不需要嵌入API
            collection_name = settings.MILVUS_COLLECTION_NAME
            collection = Collection(collection_name)
            collection.load()
            
            search_result = collection.query(
                expr=f"source == '{source}'",
                output_fields=["source"],
                consistency_level="Strong",
                limit=1
            )
            
            return len(search_result) > 0
        except Exception as e:
            logger.error(f"检查文档是否存在失败: {e}")
            return False



# 单例实例
_milvus_manager_instance = None

def create_milvus_manager():
    """
    创建Milvus管理器实例（单例模式）
    
    Returns:
        MilvusManager: Milvus管理器实例
    """
    global _milvus_manager_instance
    if _milvus_manager_instance is None:
        _milvus_manager_instance = MilvusManager()
    return _milvus_manager_instance
