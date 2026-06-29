# 轻量版PDF处理器

from typing import List, Dict, Optional
import pdfplumber
import io
import re
import requests
from configs.config import settings
from utils.logger import logger


class TikaProcessor:
    """
    Tika处理器，负责使用Tika Server解析文档
    """
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or settings.TIKA_SERVER_URL
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """使用Tika Server提取文本"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(f"{self.server_url}/tika", data=f, headers={'Accept': 'text/plain'})
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"Tika Server错误: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Tika提取文本失败: {e}")
            return None
    
    def extract_metadata(self, file_path: str) -> Optional[Dict]:
        """使用Tika Server提取文档元数据"""
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


class PDFProcessorLight:
    """
    轻量版PDF处理器，支持可选OCR功能
    """
    
    def __init__(self, mode: str = None, enable_ocr: bool = False):
        """
        初始化PDF处理器
        
        Args:
            mode: 处理模式，可选 'auto', 'digital', 'scanned'
            enable_ocr: 是否启用OCR功能（需要额外依赖）
        """
        self.mode = (mode or settings.PDF_PROCESSOR_MODE).lower()
        self.enable_ocr = enable_ocr
        self.enable_watermark_filter = True
        
        # 初始化Tika处理器
        self.tika_processor = TikaProcessor()
        
        # 延迟初始化OCR引擎
        self.ocr = None
        if self.enable_ocr:
            self._init_ocr()
    
    def _init_ocr(self):
        """初始化OCR引擎（可选）"""
        try:
            # 动态导入，避免依赖问题
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            logger.info("OCR引擎初始化成功")
        except ImportError:
            logger.warning("OCR依赖未安装，扫描件处理功能受限")
            self.ocr = None
        except Exception as e:
            logger.error(f"OCR初始化失败: {e}")
            self.ocr = None
    
    def extract_elements(self, pdf_path: str) -> List[Dict]:
        """
        从PDF中提取元素，支持轻量级处理
        """
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        if self.mode == 'auto':
            # 自动识别PDF类型
            is_scanned = self._is_scanned_pdf(pdf_path)
            if is_scanned and self.enable_ocr and self.ocr:
                logger.info("检测到扫描件PDF，使用OCR处理")
                return self._process_scanned_pdf_light(pdf_path)
            else:
                logger.info("使用数字PDF处理流程")
                return self._process_digital_pdf(pdf_path)
        elif self.mode == 'digital':
            logger.info("使用数字PDF处理流程")
            return self._process_digital_pdf(pdf_path)
        elif self.mode == 'scanned':
            if self.enable_ocr and self.ocr:
                logger.info("使用扫描件处理流程")
                return self._process_scanned_pdf_light(pdf_path)
            else:
                logger.warning("OCR功能未启用，回退到数字PDF处理")
                return self._process_digital_pdf(pdf_path)
        else:
            logger.warning(f"未知模式: {self.mode}，使用数字PDF处理")
            return self._process_digital_pdf(pdf_path)
    
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """判断PDF是否为扫描件"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 检查前3页
                for page_num, page in enumerate(pdf.pages[:3], 1):
                    text = page.extract_text()
                    if text and len(text.strip()) > 100:
                        logger.info(f"第{page_num}页提取到足够文本，判断为数字PDF")
                        return False
                logger.info("前3页均未提取到足够文本，判断为扫描件PDF")
                return True
        except Exception as e:
            logger.error(f"判断PDF类型失败: {e}")
            return False
    
    def _process_digital_pdf(self, pdf_path: str) -> List[Dict]:
        """处理数字PDF（轻量版）"""
        elements = []
        text_count = 0
        table_count = 0
        image_count = 0
        
        try:
            logger.info("开始处理数字PDF")
            
            # 使用pdfplumber提取内容
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        text = self._remove_watermark(text)
                        if text:
                            elements.append({
                                'type': 'text',
                                'content': text,
                                'page': page_num,
                                'position': 'full_page',
                                'source': 'pdfplumber'
                            })
                            text_count += 1
                    
                    # 提取表格（基础功能）
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            elements.append({
                                'type': 'table',
                                'table_data': table,
                                'page': page_num,
                                'position': f'table_{table_idx}',
                                'source': 'pdfplumber'
                            })
                            table_count += 1
                    
                    # 提取图片（轻量版）
                    images = page.images
                    for img_idx, img_obj in enumerate(images):
                        if img_obj:
                            # 获取图片位置信息
                            bbox = (img_obj['x0'], img_obj['top'], img_obj['x1'], img_obj['bottom'])
                            elements.append({
                                'type': 'image',
                                'page': page_num,
                                'position': f'image_{img_idx}',
                                'bbox': bbox,
                                'source': 'pdfplumber'
                            })
                            image_count += 1
            
            # 如果没有提取到内容，尝试使用Tika
            if not elements:
                logger.info("未提取到内容，尝试使用Tika")
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
                        text_count += 1
            
            # 只显示最终统计结果
            logger.info(f"PDF处理完成：共{total_pages}页，提取{len(elements)}个元素（文本：{text_count}个，表格：{table_count}个，图片：{image_count}个）")
            
            return elements
        except Exception as e:
            logger.error(f"数字PDF处理失败: {e}")
            return []
    
    def _process_scanned_pdf_light(self, pdf_path: str) -> List[Dict]:
        """轻量版扫描件处理（依赖OCR）"""
        if not self.ocr:
            logger.warning("OCR功能不可用，回退到数字PDF处理")
            return self._process_digital_pdf(pdf_path)
        
        elements = []
        
        try:
            logger.info("开始轻量版扫描件处理")
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF文件共{len(pdf.pages)}页")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"处理第{page_num}页")
                    
                    # 提取页面为图片
                    img = page.to_image()
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # OCR识别
                    ocr_result = self.ocr.ocr(img_bytes.getvalue(), cls=True)
                    
                    if ocr_result and ocr_result[0]:
                        text_lines = []
                        for line in ocr_result[0]:
                            if line and len(line) > 1:
                                text = line[1][0]
                                confidence = line[1][1]
                                if confidence > 0.5:
                                    text_lines.append(text)
                        
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
            
            logger.info(f"扫描件PDF处理完成，共提取{len(elements)}个元素")
            return elements
        except Exception as e:
            logger.error(f"扫描件处理失败: {e}")
            return self._process_digital_pdf(pdf_path)  # 失败时回退
    
    def _remove_watermark(self, text: str) -> str:
        """移除水印文本"""
        if not text:
            return text
        
        # 常见水印模式
        watermark_patterns = [
            r'机密|秘密|内部资料|严禁外传',
            r'CONFIDENTIAL|SECRET|INTERNAL USE ONLY',
            r'第\s*\d+\s*页.*共\s*\d+\s*页',  # 页码水印
        ]
        
        for pattern in watermark_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
