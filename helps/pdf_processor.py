from typing import List, Dict, Optional
import pdfplumber
import io
import re
import base64
import requests
from configs.config import settings
from utils.logger import logger


class TikaProcessor:
    """
    Tika处理器，负责使用Tika Server解析文档
    """
    
    def __init__(self, server_url: str = None):
        """
        初始化Tika处理器
        
        Args:
            server_url: Tika Server的URL，默认从配置文件读取
        """
        self.server_url = server_url or settings.TIKA_SERVER_URL
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        使用Tika Server从文档中提取纯文本内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[str]: 提取的纯文本内容
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(f"{self.server_url}/tika", data=f, headers={'Accept': 'text/plain'})
            
            if response.status_code == 200:
                return response.text.replace('\n', '')
            else:
                logger.error(f"Tika Server错误: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Tika解析失败: {e}")
            return None
    
    def extract_metadata(self, file_path: str) -> Optional[Dict]:
        """
        使用Tika Server提取文档元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Dict]: 提取的元数据
        """
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(f"{self.server_url}/meta", data=f, headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Tika Server错误: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Tika提取元数据失败: {e}")
            return None


class PDFProcessor:
    """
    PDF处理器，支持自动识别PDF类型并选择合适的处理流程
    """
    
    def __init__(self, mode: str = None):
        """
        初始化PDF处理器（完整OCR版本）
        
        Args:
            mode: 处理模式，可选 'auto', 'digital', 'scanned'，默认从配置文件读取
        """
        # 从配置文件读取默认值
        self.mode = (mode or settings.PDF_PROCESSOR_MODE).lower()
        # 水印功能始终启用
        self.enable_watermark_filter = True
        
        # 初始化Tika处理器（始终启用）
        self.tika_processor = TikaProcessor()
        
        # 直接初始化OCR引擎（完整版本）
        import cv2
        import numpy as np
        from paddleocr import PaddleOCR
        
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            logger.info("OCR引擎初始化成功")
        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            raise RuntimeError("OCR功能初始化失败，请检查依赖是否安装")
    
    def extract_elements(self, pdf_path: str) -> List[Dict]:
        """
        从PDF中提取元素，根据PDF类型选择不同的处理流程
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            List[Dict]: 包含所有元素的列表
        """
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        if self.mode == 'auto':
            # 自动识别PDF类型
            is_scanned = self._is_scanned_pdf(pdf_path)
            if is_scanned:
                logger.info("检测到扫描件PDF，使用扫描件处理流程")
                return self._process_scanned_pdf(pdf_path)
            else:
                logger.info("检测到数字PDF，使用数字PDF处理流程")
                return self._process_digital_pdf(pdf_path)
        elif self.mode == 'digital':
            logger.info("使用数字PDF处理流程")
            return self._process_digital_pdf(pdf_path)
        elif self.mode == 'scanned':
            logger.info("使用扫描件处理流程")
            return self._process_scanned_pdf(pdf_path)
        else:
            logger.warning(f"未知模式: {self.mode}，使用自动模式")
            self.mode = 'auto'
            return self.extract_elements(pdf_path)
    
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        判断PDF是否为扫描件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否为扫描件
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 检查前3页
                for page_num, page in enumerate(pdf.pages[:3], 1):
                    text = page.extract_text()
                    if text and len(text.strip()) > 100:
                        # 提取到足够的文本，不是扫描件
                        logger.info(f"第{page_num}页提取到足够文本，判断为数字PDF")
                        return False
                # 前3页都没有提取到足够的文本，可能是扫描件
                logger.info("前3页均未提取到足够文本，判断为扫描件PDF")
                return True
        except Exception as e:
            logger.error(f"判断PDF类型失败: {e}")
            return False
    
    def _process_digital_pdf(self, pdf_path: str) -> List[Dict]:
        """
        处理数字PDF
        """
        elements = []
        
        try:
            logger.info("开始处理数字PDF")
            
            # 1. 使用Tika分析文本与元数据
            logger.info("使用Tika分析文本与元数据")
            tika_text = self.tika_processor.extract_text(pdf_path)
            metadata = self.tika_processor.extract_metadata(pdf_path)
            if metadata:
                elements.append({
                    'type': 'metadata',
                    'content': metadata,
                    'source': 'tika'
                })
                logger.info("成功提取元数据")
            
            # 2. 使用pdfplumber提取内容
            logger.info("使用pdfplumber提取内容")
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF文件共{len(pdf.pages)}页")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"处理第{page_num}页")
                    
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        # 水印过滤
                        text = self._remove_watermark(text)
                        if text:
                            elements.append({
                                'type': 'text',
                                'content': text,
                                'page': page_num,
                                'position': 'full_page',
                                'source': 'pdfplumber'
                            })
                            logger.info(f"成功提取第{page_num}页文本")
                    
                    # 提取图像
                    for img_idx, img_obj in enumerate(page.images):
                        try:
                            x0, top, x1, bottom = img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom']
                            bbox = (x0, top, x1, bottom)
                            
                            # 裁剪图片
                            im = page.within_bbox(bbox).to_image()
                            img_bytes = io.BytesIO()
                            im.save(img_bytes, format='PNG')
                            img_bytes.seek(0)
                            
                            # 编码为base64
                            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                            
                            # 提取图片标题
                            image_title = self._extract_image_title(page, bbox)
                            
                            elements.append({
                                'type': 'image',
                                'image_bytes': img_bytes,
                                'image_base64': img_base64,
                                'title': image_title,
                                'page': page_num,
                                'position': f'image_{img_idx}',
                                'bbox': bbox
                            })
                            logger.info(f"成功提取第{page_num}页第{img_idx+1}张图片")
                        except Exception as e:
                            logger.error(f"提取图片失败: {e}")
                            continue
                    
                    # 提取表格
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        # 提取表格标题
                        table_title = self._extract_table_title(page, table.bbox if hasattr(table, 'bbox') else None)
                        
                        # 结构化表格数据
                        structured_table = self._process_table_data(table)
                        
                        elements.append({
                            'type': 'table',
                            'table_data': table,
                            'structured_table': structured_table,
                            'title': table_title,
                            'page': page_num,
                            'position': f'table_{table_idx}',
                            'bbox': table.bbox if hasattr(table, 'bbox') else None
                        })
                        logger.info(f"成功提取第{page_num}页第{table_idx+1}个表格")
            
            # 如果没有提取到内容，使用Tika的结果
            if not any(e['type'] in ['text', 'table', 'image'] for e in elements) and tika_text:
                logger.info("未提取到内容，使用Tika结果作为备用")
                text = self._remove_watermark(tika_text)
                if text:
                    elements.append({
                        'type': 'text',
                        'content': text,
                        'page': 1,
                        'position': 'full_page',
                        'source': 'tika'
                    })
            
            logger.info(f"数字PDF处理完成，共提取{len(elements)}个元素")
            return elements
        except Exception as e:
            logger.error(f"数字PDF处理失败: {e}")
            
            # 尝试使用Tika作为备用
            try:
                logger.info("尝试使用Tika作为备用")
                text = self.tika_processor.extract_text(pdf_path)
                if text:
                    text = self._remove_watermark(text)
                    if text:
                        logger.info("Tika备用处理成功")
                        return [{
                            'type': 'text',
                            'content': text,
                            'page': 1,
                            'position': 'full_page',
                            'source': 'tika'
                        }]
            except Exception as backup_e:
                logger.error(f"Tika备用处理失败: {backup_e}")
                pass
            
            return []
    
    def _process_scanned_pdf(self, pdf_path: str) -> List[Dict]:
        """
        处理扫描件PDF（使用OCR）
        """
        elements = []
        
        try:
            logger.info("开始处理扫描件PDF")
            
            # 直接使用已初始化的OCR引擎
            if self.ocr is None:
                logger.error("OCR引擎未初始化，无法处理扫描件")
                return self._process_digital_pdf(pdf_path)  # 回退到数字PDF处理
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF文件共{len(pdf.pages)}页")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"处理第{page_num}页")
                    
                    # 提取页面为图片
                    img = page.to_image()
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # 图像预处理
                    processed_image = self._preprocess_image(img_bytes)
                    
                    # OCR识别（直接使用已初始化的OCR引擎）
                    logger.info(f"对第{page_num}页进行OCR识别")
                    ocr_result = self.ocr.ocr(processed_image, cls=True)
                    
                    if ocr_result and ocr_result[0]:
                        # 提取识别的文本
                        text_lines = []
                        table_candidates = []
                        
                        for line in ocr_result[0]:
                            if line and len(line) > 1:
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                # 过滤低置信度结果
                                if confidence > 0.5:
                                    text_lines.append(text)
                                    
                                    # 检测表格内容
                                    if self._is_table_content(text):
                                        table_candidates.append(text)
                        
                        # 结构化表格
                        if table_candidates:
                            structured_table = self._structure_table_from_ocr(table_candidates)
                            elements.append({
                                'type': 'table',
                                'table_data': table_candidates,
                                'structured_table': structured_table,
                                'page': page_num,
                                'position': 'ocr_table',
                                'source': 'ocr'
                            })
                            logger.info(f"成功提取第{page_num}页表格")
                        
                        # 输出清洗文本
                        text_content = '\n'.join(text_lines)
                        text_content = self._remove_watermark(text_content)
                        
                        if text_content:
                            elements.append({
                                'type': 'text',
                                'content': text_content,
                                'page': page_num,
                                'position': 'full_page',
                                'source': 'ocr'
                            })
                            logger.info(f"成功提取第{page_num}页OCR文本")
                    else:
                        logger.warning("OCR识别未返回结果")
            
            # 如果没有提取到内容，尝试使用Tika
            if not elements:
                logger.info("未提取到内容，尝试使用Tika")
                try:
                    text = self.tika_processor.extract_text(pdf_path)
                    if text:
                        text = self._remove_watermark(text)
                        if text:
                            elements.append({
                                'type': 'text',
                                'content': text,
                                'page': 1,
                                'position': 'full_page',
                                'source': 'tika'
                            })
                            logger.info("Tika处理成功")
                except Exception as e:
                    logger.error(f"Tika处理失败: {e}")
                    pass
            
            logger.info(f"扫描件PDF处理完成，共提取{len(elements)}个元素")
            return elements
        except Exception as e:
            logger.error(f"扫描件PDF处理失败: {e}")
            
            # 尝试使用Tika作为备用
            try:
                logger.info("尝试使用Tika作为备用")
                text = self.tika_processor.extract_text(pdf_path)
                if text:
                    text = self._remove_watermark(text)
                    if text:
                        logger.info("Tika备用处理成功")
                        return [{
                            'type': 'text',
                            'content': text,
                            'page': 1,
                            'position': 'full_page',
                            'source': 'tika'
                        }]
            except Exception as backup_e:
                logger.error(f"Tika备用处理失败: {backup_e}")
                pass
            
            return []
    
    def _preprocess_image(self, img_bytes: io.BytesIO):
        """
        图像预处理（OCR优化）
        """
        try:
            # 直接使用已导入的cv2和numpy
            import cv2
            import numpy as np
            
            # 读取图像
            img_array = np.frombuffer(img_bytes.getvalue(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning("图像解码失败，返回原始图像")
                return img_array
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 自适应阈值二值化
            thresh = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            # 膨胀操作增强文本
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.dilate(thresh, kernel, iterations=1)
            
            return processed
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            # 返回原始图像
            img_bytes.seek(0)
            import numpy as np
            return np.frombuffer(img_bytes.getvalue(), np.uint8)
    
    def _process_table_data(self, table: List[List[str]]) -> List[Dict]:
        """
        处理表格数据，转换为结构化格式
        """
        structured_table = []
        
        if not table or len(table) < 2:
            return structured_table
        
        # 假设第一行为表头
        headers = table[0]
        
        # 处理数据行
        for row in table[1:]:
            row_data = {}
            for i, value in enumerate(row):
                if i < len(headers):
                    row_data[headers[i] or f'col_{i}'] = value
                else:
                    row_data[f'col_{i}'] = value
            structured_table.append(row_data)
        
        return structured_table
    
    def _is_table_content(self, text: str) -> bool:
        """
        判断文本是否为表格内容
        """
        # 简单判断：包含数字和分隔符的文本可能是表格内容
        if re.search(r'\d+', text) and (',' in text or '\t' in text or ' ' in text):
            return True
        return False
    
    def _structure_table_from_ocr(self, table_candidates: List[str]) -> List[Dict]:
        """
        从OCR结果结构化表格数据
        """
        structured_table = []
        
        if not table_candidates:
            return structured_table
        
        # 简单处理：假设第一行为表头
        headers = table_candidates[0].split()
        
        # 处理数据行
        for row in table_candidates[1:]:
            row_data = {}
            values = row.split()
            for i, value in enumerate(values):
                if i < len(headers):
                    row_data[headers[i]] = value
                else:
                    row_data[f'col_{i}'] = value
            structured_table.append(row_data)
        
        return structured_table
    
    def _remove_watermark(self, text) -> str:
        """
        水印过滤
        """
        if not text:
            return text
        
        # 确保text是字符串类型
        if not isinstance(text, str):
            return str(text)
        
        # 常见水印模式
        watermark_patterns = [
            r'仅供.*参考',
            r'机密.*文件',
            r'内部.*资料',
            r'CONFIDENTIAL',
            r'DRAFT',
            r'草稿',
            r'样本',
            r'SAMPLE',
            r'版本.*\d+\.\d+',
            r'第.*页.*共.*页',
            r'Page\s*\d+\s*of\s*\d+',
            r'©\s*\d{4}',
            r'版权所有',
            r'All rights reserved'
        ]
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是水印
            is_watermark = False
            for pattern in watermark_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_watermark = True
                    break
            
            if not is_watermark:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _extract_table_title(self, page, table_bbox) -> str:
        """
        提取表格标题
        """
        try:
            text = page.extract_text()
            if not text:
                return ""
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if re.search(r'表\s*\d+.*', line) or re.search(r'表格\s*\d*.*', line):
                    return line
            
            return ""
        except Exception as e:
            logger.error(f"提取表格标题失败: {e}")
            return ""
    
    def _extract_image_title(self, page, image_bbox) -> str:
        """
        提取图片标题
        """
        try:
            text = page.extract_text()
            if not text:
                return ""
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if re.search(r'图\s*\d+[-]*.*', line) or re.search(r'图片\s*\d*.*', line):
                    return line
            
            return ""
        except Exception as e:
            logger.error(f"提取图片标题失败: {e}")
            return ""
