# 切块模块

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from utils.logger import logger


class ContentProcessor:
    """
    内容处理器，负责处理文档内容并进行分块
    支持通用分块和父子分块两种方式
    """
    
    def __init__(self):
        """
        初始化内容处理器
        """
        # 父块分割器：以\n\n分段，最大长度1024
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        # 子块分割器：以\n分段，最大长度512
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n", " ", ""]
        )
        logger.info("初始化内容处理器")
    
    def generate_context(self, elements: List[Dict], current_idx: int) -> str:
        """
        生成元素的上下文描述
        
        Args:
            elements: 所有元素列表
            current_idx: 当前元素索引
            
        Returns:
            str: 上下文描述
        """
        context = []
        
        # 获取前一个元素作为上下文
        if current_idx > 0:
            prev_element = elements[current_idx - 1]
            if prev_element['type'] == 'text':
                prev_text = prev_element['content'][:50] + ('...' if len(prev_element['content']) > 50 else '')
                context.append(f"前文: {prev_text}")
            elif prev_element['type'] == 'table':
                context.append("前文: 表格")
            elif prev_element['type'] == 'image':
                context.append("前文: 图片")
        
        # 获取后一个元素作为上下文
        if current_idx < len(elements) - 1:
            next_element = elements[current_idx + 1]
            if next_element['type'] == 'text':
                next_text = next_element['content'][:50] + ('...' if len(next_element['content']) > 50 else '')
                context.append(f"后文: {next_text}")
            elif next_element['type'] == 'table':
                context.append("后文: 表格")
            elif next_element['type'] == 'image':
                context.append("后文: 图片")
        
        return " ".join(context) if context else "无上下文"
    
    def table_to_text(self, table: List[List[str]]) -> str:
        """
        将表格转换为文本格式
        
        Args:
            table: 表格数据
            
        Returns:
            str: 表格的文本表示
        """
        if not table:
            return ""
        
        rows = []
        for row in table:
            cleaned_row = [cell.strip() if cell else "" for cell in row]
            rows.append(" | ".join(cleaned_row))
        
        return "\n".join(rows)
    
    def process_table(self, table_data: List[List[str]], title: str = "") -> str:
        """
        处理表格元素
        
        Args:
            table_data: 表格数据
            title: 表格标题
            
        Returns:
            str: 处理后的表格内容
        """
        table_text = self.table_to_text(table_data)
        
        if title:
            return f"标题：{title}\n{table_text}"
        else:
            return table_text
    
    def process_image(self, title: str = "") -> str:
        """
        处理图片元素
        
        Args:
            title: 图片标题
            
        Returns:
            str: 处理后的图片内容
        """
        if title:
            return f"标题：{title}\n图片内容: 图片已保存"
        else:
            return "图片内容: 图片已保存"
    
    def process_elements_generic(self, elements: List[Dict], object_name: str, metadata: dict) -> List[Document]:
        """
        通用分块方式处理元素
        
        Args:
            elements: 元素列表
            object_name: 对象名称
            metadata: 元数据
            
        Returns:
            List[Document]: 处理后的文档列表
        """
        processed_docs = []
        
        for i, element in enumerate(elements):
            element_metadata = metadata.copy() if metadata else {}
            element_metadata.update({
                'source': object_name,
                'element_type': element['type'],
                'position': element.get('position', f'element_{i}'),
                'page': element.get('page', 1)
            })
            
            # 根据元素类型进行处理
            if element['type'] == 'table':
                # 表格作为整体处理
                table_title = element.get('title', '')
                table_content = self.process_table(element['table_data'], table_title)
                context_description = self.generate_context(elements, i)
                
                final_content = f"[表格]\n标题: {table_title}\n上下文: {context_description}\n{table_content}"
                
                element_metadata['table_data'] = element['table_data']
                element_metadata['is_table'] = True
                if table_title:
                    element_metadata['title'] = table_title
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            elif element['type'] == 'image':
                # 图片作为整体处理
                image_title = element.get('title', '')
                image_content = self.process_image(image_title)
                context_description = self.generate_context(elements, i)
                
                final_content = f"[图片]\n标题: {image_title}\n上下文: {context_description}\n{image_content}"
                
                element_metadata['image_base64'] = element.get('image_base64', '')
                element_metadata['is_image'] = True
                if image_title:
                    element_metadata['title'] = image_title
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            elif element['type'] == 'metadata':
                # 元数据作为整体处理
                metadata_content = str(element['content'])
                context_description = self.generate_context(elements, i)
                
                final_content = f"[元数据]\n上下文: {context_description}\n{metadata_content}"
                
                element_metadata['is_metadata'] = True
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            else:  # text
                # 文本使用分割器处理
                if isinstance(element['content'], str):
                    text_docs = self.parent_splitter.create_documents([element['content']])
                    for j, text_doc in enumerate(text_docs):
                        text_metadata = element_metadata.copy()
                        text_metadata['chunk_index'] = j
                        text_doc.metadata.update(text_metadata)
                        processed_docs.append(text_doc)
                else:
                    # 非字符串类型，转换为字符串处理
                    text_content = str(element['content'])
                    text_docs = self.parent_splitter.create_documents([text_content])
                    for j, text_doc in enumerate(text_docs):
                        text_metadata = element_metadata.copy()
                        text_metadata['chunk_index'] = j
                        text_doc.metadata.update(text_metadata)
                        processed_docs.append(text_doc)
        
        return processed_docs
    
    def process_elements_hierarchical(self, elements: List[Dict], object_name: str, metadata: dict) -> List[Document]:
        """
        父子分块方式处理元素
        
        Args:
            elements: 元素列表
            object_name: 对象名称
            metadata: 元数据
            
        Returns:
            List[Document]: 处理后的文档列表（包含父块和子块）
        """
        processed_docs = []
        
        for i, element in enumerate(elements):
            element_metadata = metadata.copy() if metadata else {}
            element_metadata.update({
                'source': object_name,
                'element_type': element['type'],
                'position': element.get('position', f'element_{i}'),
                'page': element.get('page', 1)
            })
            
            # 生成上下文描述
            context_description = self.generate_context(elements, i)
            
            # 根据元素类型进行处理
            if element['type'] == 'table':
                # 表格作为整体处理（只有父块）
                table_title = element.get('title', '')
                table_content = self.process_table(element['table_data'], table_title)
                
                final_content = f"[表格]\n标题：{table_title}\n上下文：{context_description}\n{table_content}"
                
                element_metadata['table_data'] = element['table_data']
                element_metadata['is_table'] = True
                element_metadata['chunk_type'] = 'parent'
                element_metadata['parent_chunk_index'] = 0
                if table_title:
                    element_metadata['title'] = table_title
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            elif element['type'] == 'image':
                # 图片作为整体处理（只有父块）
                image_title = element.get('title', '')
                image_content = self.process_image(image_title)
                
                final_content = f"[图片]\n标题：{image_title}\n上下文：{context_description}\n{image_content}"
                
                element_metadata['image_base64'] = element.get('image_base64', '')
                element_metadata['is_image'] = True
                element_metadata['chunk_type'] = 'parent'
                element_metadata['parent_chunk_index'] = 0
                if image_title:
                    element_metadata['title'] = image_title
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            elif element['type'] == 'metadata':
                # 元数据作为整体处理（只有父块）
                metadata_content = str(element['content'])
                
                final_content = f"[元数据]\n上下文：{context_description}\n{metadata_content}"
                
                element_metadata['is_metadata'] = True
                element_metadata['chunk_type'] = 'parent'
                element_metadata['parent_chunk_index'] = 0
                
                doc = Document(
                    page_content=final_content,
                    metadata=element_metadata
                )
                processed_docs.append(doc)
            
            else:  # text
                # 文本使用父子分块处理
                # 确保content是字符串类型
                content = element['content']
                if not isinstance(content, str):
                    content = str(content)
                
                # 1. 创建父块（以\n\n分段，最大长度1024）
                parent_docs = self.parent_splitter.create_documents([content])
                for p_idx, parent_doc in enumerate(parent_docs):
                    parent_metadata = element_metadata.copy()
                    parent_metadata['chunk_type'] = 'parent'
                    parent_metadata['parent_chunk_index'] = p_idx
                    
                    parent_content = f"[文本]\n上下文: {context_description}\n{parent_doc.page_content}"
                    parent_doc = Document(
                        page_content=parent_content,
                        metadata=parent_metadata
                    )
                    processed_docs.append(parent_doc)
                    
                    # 2. 创建子块（以\n分段，最大长度512）
                    child_docs = self.child_splitter.create_documents([parent_doc.page_content])
                    for j, child_doc in enumerate(child_docs):
                        child_metadata = element_metadata.copy()
                        child_metadata['chunk_type'] = 'child'
                        child_metadata['parent_index'] = len(processed_docs) - 1  # 指向父块的索引
                        child_metadata['chunk_index'] = j
                        child_doc.metadata.update(child_metadata)
                        processed_docs.append(child_doc)
        
        return processed_docs
    
    def process_content(self, elements: List[Dict], object_name: str, metadata: dict, chunking_strategy: str = 'generic') -> List[Document]:
        """
        处理内容并进行分块

        Args:
            elements: 元素列表
            object_name: 对象名称
            metadata: 元数据
            chunking_strategy: 分块策略 ('generic' 或 'hierarchical')

        Returns:
            List[Document]: 处理后的文档列表
        """
        logger.info(f"开始处理内容，使用{chunking_strategy}分块策略，共{len(elements)}个元素")
        
        if chunking_strategy == 'hierarchical':
            result = self.process_elements_hierarchical(elements, object_name, metadata)
        else:  # generic
            result = self.process_elements_generic(elements, object_name, metadata)
        
        logger.info(f"内容处理完成，生成{len(result)}个文档块")
        return result


# 单例实例
_content_processor_instance = None

def create_content_processor():
    """
    创建内容处理器实例（单例模式）
    
    Returns:
        ContentProcessor: 内容处理器实例
    """
    global _content_processor_instance
    if _content_processor_instance is None:
        _content_processor_instance = ContentProcessor()
    return _content_processor_instance
