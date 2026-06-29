import os
from utils.milvus_manager import MilvusManager
from utils.minio_manager import create_minio_manager
from configs.config import settings
from helps.chunk_processor import create_content_processor
from helps.pdf_processor_light import PDFProcessorLight
from helps.pdf_processor import PDFProcessor
from helps.docx_processor import DocxProcessor
from typing import List, Dict, Tuple, Callable
import tempfile
from utils.logger import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough


class DocumentProcessor:
    """
    文档处理器，用于从Minio处理文档并存储到向量数据库
    集成langchain的DocumentLoaders和LCEL
    """
    
    def __init__(self):
        """
        初始化文档处理器
        """
        self.vector_store_manager = MilvusManager()
        self.minio_manager = create_minio_manager()
        self.content_processor = create_content_processor()
        
        # 根据OCR配置选择合适的PDF处理器
        enable_ocr = os.getenv("ENABLE_OCR", "false").lower() == "true"
        
        if enable_ocr:
            logger.info("启用OCR功能，使用完整版PDF处理器")
            self.pdf_processor = PDFProcessor(mode=settings.PDF_PROCESSOR_MODE)
        else:
            logger.info("禁用OCR功能，使用轻量版PDF处理器")
            self.pdf_processor = PDFProcessorLight(mode=settings.PDF_PROCESSOR_MODE, enable_ocr=False)
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # 创建LCEL处理链条
        self.process_chain = self._create_process_chain()

    def _create_process_chain(self) -> Callable:
        """
        创建文档处理的LCEL链条
        
        Returns:
            Callable: 文档处理函数
        """
        # 步骤1: 检查文档是否存在
        def check_document_exists(inputs):
            object_name = inputs["object_name"]
            exists = self.vector_store_manager.check_document_exists(object_name)
            return {**inputs, "exists": exists}
        
        # 步骤2: 下载文档
        def download_document(inputs):
            if inputs["exists"]:
                return {**inputs, "file_content": None}
            
            object_name = inputs["object_name"]
            try:
                response = self.minio_manager.client.get_object(
                    Bucket=settings.MINIO_BUCKET_NAME,
                    Key=object_name
                )
                file_content = response['Body'].read()
                return {**inputs, "file_content": file_content}
            except Exception as e:
                return {**inputs, "file_content": None, "error": str(e)}
        
        # 步骤3: 加载文档
        def load_document(inputs):
            if inputs["exists"] or inputs["file_content"] is None:
                return {**inputs, "documents": []}
            
            object_name = inputs["object_name"]
            file_content = inputs["file_content"]
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(object_name)[1]) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # 准备元数据
                metadata = inputs.get("metadata", {})
                doc_metadata = metadata.copy() if metadata else {}
                # 存储完整的文件路径到source字段，用于精确删除
                doc_metadata['source'] = object_name
                doc_metadata['title'] = os.path.basename(object_name).replace('.pdf', '').replace('.docx', '')

                # 加载文档
                file_ext = os.path.splitext(temp_file_path)[1].lower()
                documents = []
                
                if file_ext == '.pdf':
                    # 使用高级PDF处理器（支持扫描件和OCR）
                    try:
                        elements = self.pdf_processor.extract_elements(temp_file_path)
                        # 将元素转换为Document对象
                        for element in elements:
                            if element['type'] == 'text':
                                doc = Document(
                                    page_content=element['content'],
                                    metadata={
                                        **doc_metadata,
                                        'element_type': 'text',
                                        'page': element.get('page', 1),
                                        'position': element.get('position', ''),
                                        'source': element.get('source', 'pdf_processor')
                                    }
                                )
                                documents.append(doc)
                            elif element['type'] == 'table':
                                # 处理表格内容
                                table_content = self._format_table_element(element)
                                doc = Document(
                                    page_content=table_content,
                                    metadata={
                                        **doc_metadata,
                                        'element_type': 'table',
                                        'page': element.get('page', 1),
                                        'position': element.get('position', ''),
                                        'source': element.get('source', 'pdf_processor'),
                                        'is_table': True
                                    }
                                )
                                documents.append(doc)
                            elif element['type'] == 'image':
                                # 处理图片元素
                                image_content = f"图片位于第{element.get('page', 1)}页，位置：{element.get('position', '')}"
                                doc = Document(
                                    page_content=image_content,
                                    metadata={
                                        **doc_metadata,
                                        'element_type': 'image',
                                        'page': element.get('page', 1),
                                        'position': element.get('position', ''),
                                        'source': element.get('source', 'pdf_processor'),
                                        'bbox': element.get('bbox', ''),
                                        'is_image': True
                                    }
                                )
                                documents.append(doc)
                        
                        logger.info(f"PDF处理完成，提取 {len(documents)} 个文档元素")
                        
                    except Exception as e:
                        logger.error(f"高级PDF处理失败: {e}")
                        # 回退到基础PDF处理（使用pdfplumber直接提取文本）
                        try:
                            import pdfplumber
                            with pdfplumber.open(temp_file_path) as pdf:
                                text = ''
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text += page_text + '\n'
                            
                            if text.strip():
                                doc = Document(
                                    page_content=text.strip(),
                                    metadata=doc_metadata
                                )
                                documents = [doc]
                                logger.info("成功使用基础PDF处理回退方案")
                            else:
                                logger.warning("基础PDF处理也未能提取到文本")
                                documents = []
                        except Exception as fallback_e:
                            logger.error(f"基础PDF处理回退也失败: {fallback_e}")
                            documents = []
                        
                elif file_ext == '.docx':
                    # 使用DocxProcessor处理（支持提取文本、表格、图片）
                    docx_processor = DocxProcessor()
                    elements = docx_processor.extract_elements(temp_file_path)
                    # 将元素转换为Document对象
                    for element in elements:
                        if element['type'] == 'text':
                            doc = Document(
                                page_content=element['content'],
                                metadata={
                                    **doc_metadata,
                                    'element_type': 'text',
                                    'position': element.get('position', ''),
                                    'source': 'docx_processor'
                                }
                            )
                            documents.append(doc)
                        elif element['type'] == 'table':
                            # 处理表格内容
                            table_content = self._format_table_element(element)
                            doc = Document(
                                page_content=table_content,
                                metadata={
                                    **doc_metadata,
                                    'element_type': 'table',
                                    'position': element.get('position', ''),
                                    'source': 'docx_processor',
                                    'is_table': True
                                }
                            )
                            documents.append(doc)
                        elif element['type'] == 'image':
                            # 处理图片元素
                            image_content = f"图片：{element.get('title', '')}，位置：{element.get('position', '')}"
                            doc = Document(
                                page_content=image_content,
                                metadata={
                                    **doc_metadata,
                                    'element_type': 'image',
                                    'position': element.get('position', ''),
                                    'source': 'docx_processor',
                                    'is_image': True
                                }
                            )
                            documents.append(doc)
                    
                    logger.info(f"DOCX处理完成，提取 {len(documents)} 个文档元素")
                
                return {**inputs, "documents": documents, "metadata": doc_metadata}
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # 步骤4: 分割文档
        def split_document(inputs):
            if not inputs["documents"]:
                return {**inputs, "doc_objects": []}
            
            # 合并文档内容
            combined_content = "\n".join([doc.page_content for doc in inputs["documents"]])
            
            # 分割文档
            split_docs = self.text_splitter.split_text(combined_content)
            
            # 转换为Document对象
            doc_objects = [
                Document(page_content=chunk, metadata=inputs["metadata"].copy())
                for chunk in split_docs
            ]
            
            return {**inputs, "doc_objects": doc_objects}
        
        # 步骤5: 存储到向量数据库
        def store_documents(inputs):
            if not inputs["doc_objects"]:
                return {**inputs, "success": False, "chunk_count": 0}
            
            try:
                success = self.vector_store_manager.add_documents(inputs["doc_objects"])
                return {**inputs, "success": success, "chunk_count": len(inputs["doc_objects"])}
            except Exception as e:
                return {**inputs, "success": False, "chunk_count": 0, "error": str(e)}
        
        # 构建LCEL链条
        chain = (
            RunnablePassthrough()
            | check_document_exists
            | download_document
            | load_document
            | split_document
            | store_documents
        )
        
        return chain

    def process_minio_document(self, object_name: str, metadata: dict = None, chunking_strategy: str = None) -> Tuple[bool, int, str, str]:
        """
        处理单个Minio文档
        
        Args:
            object_name: Minio中的对象名称
            metadata: 文档元数据
            chunking_strategy: 分块策略，如果为None则使用配置文件中的设置
            
        Returns:
            Tuple[bool, int, str, str]: (是否成功, 处理的文档块数量, 错误类型, 错误信息)
        """
        # 使用配置文件中的分块策略（如果未指定）
        if chunking_strategy is None:
            chunking_strategy = settings.CHUNKING_STRATEGY
        
        try:
            # 使用LCEL链条处理文档
            result = self.process_chain.invoke({
                "object_name": object_name,
                "metadata": metadata
            })
            
            if result.get("exists"):
                logger.info(f"文档已存在，跳过: {object_name}")
                return True, 0, "", ""
            
            if result.get("error"):
                error_message = result["error"]
                error_type = "UNKNOWN_ERROR"
                
                # 识别常见错误类型
                if "NoSuchKey" in error_message:
                    error_type = "FILE_NOT_FOUND"
                    error_message = "文件不存在"
                elif "XMinioInvalidObjectName" in error_message:
                    error_type = "INVALID_OBJECT_NAME"
                    error_message = "对象名称无效"
                elif "DataNotMatchException" in error_message:
                    error_type = "MILVUS_FIELD_ERROR"
                    error_message = "Milvus字段不匹配"
                elif "No module named" in error_message:
                    error_type = "MISSING_DEPENDENCY"
                    error_message = "缺少依赖模块"
                
                logger.error(f"处理Minio文档失败: {object_name} - {error_type}: {error_message}")
                return False, 0, error_type, error_message
            
            if not result.get("documents"):
                logger.info(f"文档为空: {object_name}")
                return False, 0, "EMPTY_DOCUMENT", "文档内容为空"
            
            if not result.get("doc_objects"):
                logger.info(f"分块失败: {object_name}")
                return False, 0, "CHUNKING_ERROR", "文档分块失败"
            
            if not result.get("success"):
                return False, 0, "VECTOR_STORE_ERROR", "向量数据库存储失败"
            
            return True, result.get("chunk_count", 0), "", ""
        except Exception as e:
            error_type = "UNKNOWN_ERROR"
            error_message = str(e)
            
            # 识别常见错误类型
            if "NoSuchKey" in str(e):
                error_type = "FILE_NOT_FOUND"
                error_message = "文件不存在"
            elif "XMinioInvalidObjectName" in str(e):
                error_type = "INVALID_OBJECT_NAME"
                error_message = "对象名称无效"
            elif "DataNotMatchException" in str(e):
                error_type = "MILVUS_FIELD_ERROR"
                error_message = "Milvus字段不匹配"
            elif "No module named" in str(e):
                error_type = "MISSING_DEPENDENCY"
                error_message = "缺少依赖模块"
            
            logger.error(f"处理Minio文档失败: {object_name} - {error_type}: {error_message}")
            return False, 0, error_type, error_message

    def get_file_size_from_minio(self, object_name: str) -> int:
        """
        从MinIO获取文件大小
        
        Args:
            object_name: MinIO对象名称
            
        Returns:
            int: 文件大小（字节），获取失败返回-1
        """
        try:
            response = self.minio_manager.client.head_object(
                Bucket=settings.MINIO_BUCKET_NAME,
                Key=object_name
            )
            return response['ContentLength']
        except Exception as e:
            logger.warning(f"获取文件大小失败 {object_name}: {e}")
            return -1

    def classify_files_by_size(self, object_names: List[str]) -> tuple:
        """
        根据文件大小分类文件
        
        Args:
            object_names: 文件列表
            
        Returns:
            tuple: (小文件列表, 大文件列表)
        """
        small_files = []
        large_files = []
        
        for obj_name in object_names:
            file_size = self.get_file_size_from_minio(obj_name)
            
            if file_size == -1:
                # 获取大小失败，按小文件处理（保守策略）
                small_files.append(obj_name)
                logger.info(f"文件 {os.path.basename(obj_name)} 大小未知，按小文件处理")
            elif file_size > 5 * 1024 * 1024:  # 5MB
                large_files.append(obj_name)
                logger.info(f"文件 {os.path.basename(obj_name)} 大小: {file_size/1024/1024:.2f}MB，按大文件处理")
            else:
                small_files.append(obj_name)
                logger.info(f"文件 {os.path.basename(obj_name)} 大小: {file_size/1024:.2f}KB，按小文件处理")
        
        return small_files, large_files

    def process_minio_documents(self, object_names: List[str], metadata: dict = None, chunking_strategy: str = None) -> Dict:
        """
        批量处理Minio文档（智能并发控制）
        
        Args:
            object_names: Minio对象名称列表
            metadata: 文档元数据
            chunking_strategy: 分块策略，如果为None则使用配置文件中的设置
            
        Returns:
            Dict: 处理结果统计，包含详细错误信息
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 使用配置文件中的分块策略（如果未指定）
        if chunking_strategy is None:
            chunking_strategy = settings.CHUNKING_STRATEGY
        
        results = {"total": len(object_names), "success": 0, "failed": 0, "total_chunks": 0, "errors": []}

        logger.info(f"开始处理 {len(object_names)} 个文档...")
        
        # 根据文件大小分类
        small_files, large_files = self.classify_files_by_size(object_names)
        logger.info(f"文件分类结果: 小文件 {len(small_files)} 个，大文件 {len(large_files)} 个")
        
        # 处理小文件（并发处理）
        if small_files:
            small_results = self._process_small_files_concurrently(small_files, metadata, chunking_strategy)
            self._merge_results(results, small_results)
        
        # 处理大文件（串行处理）
        if large_files:
            large_results = self._process_large_files_sequentially(large_files, metadata, chunking_strategy)
            self._merge_results(results, large_results)

        logger.info(f"处理完成: 成功 {results['success']} 个，失败 {results['failed']} 个，共生成 {results['total_chunks']} 个文本块")
        return results

    def _format_table_element(self, table_element: Dict) -> str:
        """
        格式化表格元素为文本内容
        
        Args:
            table_element: 表格元素
            
        Returns:
            str: 格式化后的表格文本
        """
        try:
            table_data = table_element.get('table_data', [])
            structured_table = table_element.get('structured_table', [])
            title = table_element.get('title', '')
            
            table_text = ""
            
            if title:
                table_text += f"表格标题: {title}\n"
            
            if structured_table:
                # 使用结构化表格数据
                for row_idx, row_data in enumerate(structured_table):
                    if isinstance(row_data, dict):
                        row_str = " | ".join([f"{k}: {v}" for k, v in row_data.items()])
                    else:
                        row_str = str(row_data)
                    table_text += f"行{row_idx+1}: {row_str}\n"
            elif table_data:
                # 使用原始表格数据
                for row_idx, row in enumerate(table_data):
                    if isinstance(row, list):
                        row_str = " | ".join([str(cell) for cell in row])
                    else:
                        row_str = str(row)
                    table_text += f"行{row_idx+1}: {row_str}\n"
            
            if not table_text:
                table_text = "表格内容: 表格数据已提取"
            
            return f"[表格]\n{table_text}"
            
        except Exception as e:
            logger.error(f"格式化表格元素失败: {e}")
            return "[表格] 表格内容提取完成"

    def _process_small_files_concurrently(self, object_names: List[str], metadata: dict, chunking_strategy: str) -> Dict:
        """并发处理小文件"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {"total": len(object_names), "success": 0, "failed": 0, "total_chunks": 0, "errors": []}
        
        if not object_names:
            return results
        
        logger.info(f"开始并发处理 {len(object_names)} 个小文件...")
        
        # 动态调整并发数，最大不超过5
        max_workers = min(5, len(object_names), self._get_optimal_concurrency())
        logger.info(f"小文件并发数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_minio_document, obj_name, metadata, chunking_strategy): obj_name
                for obj_name in object_names
            }
            
            for idx, future in enumerate(as_completed(future_to_doc), 1):
                obj_name = future_to_doc[future]
                self._handle_single_result(future, obj_name, idx, len(object_names), results)
        
        return results

    def _process_large_files_sequentially(self, object_names: List[str], metadata: dict, chunking_strategy: str) -> Dict:
        """串行处理大文件"""
        results = {"total": len(object_names), "success": 0, "failed": 0, "total_chunks": 0, "errors": []}
        
        if not object_names:
            return results
        
        logger.info(f"开始串行处理 {len(object_names)} 个大文件...")
        
        for idx, obj_name in enumerate(object_names, 1):
            logger.info(f"[{idx}/{len(object_names)}] 处理大文件: {os.path.basename(obj_name)}")
            try:
                success, chunk_count, error_type, error_message = self.process_minio_document(obj_name, metadata, chunking_strategy)
                
                if success:
                    results["success"] += 1
                    results["total_chunks"] += chunk_count
                    if chunk_count > 0:
                        logger.info(f"  处理成功，生成 {chunk_count} 个文本块")
                    else:
                        logger.info(f"  文档已存在，跳过处理")
                else:
                    results["failed"] += 1
                    error_info = {
                        "file": obj_name,
                        "error_type": error_type,
                        "error_message": error_message
                    }
                    results["errors"].append(error_info)
                    logger.error(f"  处理失败: {error_type}: {error_message}")
            except Exception as e:
                results["failed"] += 1
                error_info = {
                    "file": obj_name,
                    "error_type": "EXECUTION_ERROR",
                    "error_message": str(e)
                }
                results["errors"].append(error_info)
                logger.error(f"  执行异常: {str(e)}")
        
        return results

    def _get_optimal_concurrency(self) -> int:
        """获取最优并发数（简单的背压机制）"""
        import psutil
        import os
        
        try:
            # 获取当前进程内存使用
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # 获取系统内存使用率
            system_memory = psutil.virtual_memory()
            memory_percent = system_memory.percent
            
            # 根据内存使用情况调整并发数
            if memory_usage > 800 or memory_percent > 80:  # 内存压力大
                logger.warning(f"内存使用较高（进程: {memory_usage:.1f}MB, 系统: {memory_percent:.1f}%），降低并发数")
                return 1
            elif memory_usage > 500 or memory_percent > 60:  # 内存使用中等
                return 2
            else:  # 内存充足
                return 5
                
        except Exception as e:
            logger.warning(f"获取系统信息失败，使用默认并发数: {e}")
            return 3  # 保守的默认值

    def _handle_single_result(self, future, obj_name, idx, total, results):
        """处理单个文件结果"""
        try:
            success, chunk_count, error_type, error_message = future.result()
            if success:
                results["success"] += 1
                results["total_chunks"] += chunk_count
                if chunk_count > 0:
                    logger.info(f"[{idx}/{total}] 处理成功: {os.path.basename(obj_name)}，生成 {chunk_count} 个文本块")
                else:
                    logger.info(f"[{idx}/{total}] 处理成功: {os.path.basename(obj_name)}，文档已存在，跳过处理")
            else:
                results["failed"] += 1
                error_info = {
                    "file": obj_name,
                    "error_type": error_type,
                    "error_message": error_message
                }
                results["errors"].append(error_info)
                logger.error(f"[{idx}/{total}] 处理失败: {os.path.basename(obj_name)} - {error_type}: {error_message}")
        except Exception as e:
            results["failed"] += 1
            error_info = {
                "file": obj_name,
                "error_type": "EXECUTION_ERROR",
                "error_message": str(e)
            }
            results["errors"].append(error_info)
            logger.error(f"[{idx}/{total}] 执行异常: {os.path.basename(obj_name)} - {str(e)}")

    def _merge_results(self, main_results, sub_results):
        """合并处理结果"""
        main_results["success"] += sub_results["success"]
        main_results["failed"] += sub_results["failed"]
        main_results["total_chunks"] += sub_results["total_chunks"]
        main_results["errors"].extend(sub_results["errors"])

    def remove_minio_documents(self, object_names: List[str]) -> Dict:
        """
        批量删除Minio文档（NO_AI文件）
        
        Args:
            object_names: Minio对象名称列表
            
        Returns:
            Dict: 删除结果统计，包含详细错误信息
        """
        results = {"total": len(object_names), "success": 0, "failed": 0, "errors": []}

        for idx, object_name in enumerate(object_names, 1):
            logger.info(f"正在删除第 {idx} 个文档: {os.path.basename(object_name)}")
            try:
                # 检查文档是否存在
                if self.vector_store_manager.check_document_exists(object_name):
                    # 删除文档
                    success = self.vector_store_manager.delete_documents(filter={"source": object_name})
                    if success:
                        results["success"] += 1
                        logger.info(f"  删除成功")
                    else:
                        results["failed"] += 1
                        error_info = {
                            "file": object_name,
                            "error_type": "DELETE_FAILED",
                            "error_message": "删除操作失败"
                        }
                        results["errors"].append(error_info)
                        logger.info(f"  删除失败")
                else:
                    results["success"] += 1
                    logger.info(f"  文档不存在，跳过")
            except Exception as e:
                results["failed"] += 1
                error_type = "UNKNOWN_ERROR"
                error_message = str(e)
                
                # 识别常见错误类型
                if "DataNotMatchException" in str(e):
                    error_type = "MILVUS_ERROR"
                    error_message = "Milvus操作失败"
                
                error_info = {
                    "file": object_name,
                    "error_type": error_type,
                    "error_message": error_message
                }
                results["errors"].append(error_info)
                logger.error(f"  删除失败: {error_type}: {error_message}")

        return results


def create_document_processor():
    """
    创建文档处理器实例
    
    Returns:
        DocumentProcessor: 文档处理器实例
    """
    return DocumentProcessor()
