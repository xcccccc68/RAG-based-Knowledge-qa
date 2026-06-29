"""
安全护栏模块
提供敏感词审计、语义分析和提示词防注入功能
"""

import os
import re
from typing import List, Tuple, Optional, Set, Dict
from utils.redis_manager import redis_manager


class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}
        self.is_end = False


class SemanticAnalyzer:
    """语义分析器，提供基于语义特征的文本分析功能"""
    
    def __init__(self, whitelist: Set[str]):
        self.whitelist = whitelist
        self._logger = None
    
    @property
    def logger(self):
        if self._logger is None:
            from utils.logger import logger
            self._logger = logger
        return self._logger
    
    def analyze_semantic_features(self, sentence: str, sensitive_word: str) -> Dict[str, bool]:
        """分析语句的语义特征"""
        if not sentence or not sensitive_word:
            return self._get_default_features()
        
        return {
            'is_technical': self.analyze_technical_features(sentence),
            'in_technical_context': self.analyze_technical_context(sentence, sensitive_word),
            'has_sensitive_intent': self.analyze_sensitive_intent(sentence),
            'in_sensitive_action_context': self.analyze_sensitive_action_context(sentence, sensitive_word)
        }
    
    def _get_default_features(self):
        """获取默认特征值"""
        return {
            'is_technical': False,
            'in_technical_context': False,
            'has_sensitive_intent': False,
            'in_sensitive_action_context': False
        }
    
    def analyze_technical_features(self, sentence: str) -> bool:
        """分析语句的技术文档特征"""
        if not sentence:
            return False
        
        features = self._analyze_text_features(sentence)
        
        score = 0
        if features['tech_keyword_count'] >= 2:
            score += 1
        if features['avg_word_length'] > 3.0:
            score += 1
        if 0.05 < features['punctuation_density'] < 0.2:
            score += 1
        if features['alnum_density'] > 0.7:
            score += 1
        if features['professional_count'] >= 1:
            score += 1
        
        return score >= 3
    
    def _analyze_text_features(self, text: str) -> Dict[str, float]:
        """分析文本特征，返回特征字典"""
        if not text:
            return {
                'tech_keyword_count': 0,
                'avg_word_length': 0,
                'punctuation_density': 0,
                'alnum_density': 0,
                'professional_count': 0
            }
        
        text_lower = text.lower()
        
        # 技术关键词计数
        tech_keywords = set(self.whitelist)
        tech_keyword_count = sum(1 for keyword in tech_keywords if keyword.lower() in text_lower)
        
        # 单词统计
        words = text_lower.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # 标点密度
        punctuation_count = sum(1 for char in text if char in '.,;:()[]{}<>《》「」')
        punctuation_density = punctuation_count / len(text) if len(text) > 0 else 0
        
        # 字母数字密度
        alnum_count = sum(1 for char in text if char.isalnum())
        alnum_density = alnum_count / len(text) if len(text) > 0 else 0
        
        # 专业模式计数
        professional_patterns = [
            r'[A-Za-z]+-[A-Za-z]+',
            r'[A-Za-z]+_[A-Za-z]+',
            r'[A-Za-z]+\d+',
            r'\d+[A-Za-z]+',
            r'[A-Z]{2,}',
        ]
        professional_count = sum(1 for pattern in professional_patterns if re.search(pattern, text))
        
        return {
            'tech_keyword_count': tech_keyword_count,
            'avg_word_length': avg_word_length,
            'punctuation_density': punctuation_density,
            'alnum_density': alnum_density,
            'professional_count': professional_count
        }
    
    def analyze_technical_context(self, sentence: str, sensitive_word: str) -> bool:
        """分析敏感词是否在技术上下文中"""
        if not sentence or not sensitive_word:
            return False
        
        sentence_lower = sentence.lower()
        word_lower = sensitive_word.lower()
        
        word_pos = sentence_lower.find(word_lower)
        if word_pos == -1:
            return False
        
        window_size = 30
        start = max(0, word_pos - window_size)
        end = min(len(sentence_lower), word_pos + len(word_lower) + window_size)
        context = sentence_lower[start:end]
        
        tech_keywords = set(self.whitelist)
        context_tech_count = sum(1 for keyword in tech_keywords if keyword.lower() in context)
        
        context_words = context.split()
        unique_words = set(context_words)
        word_diversity = len(unique_words) / len(context_words) if context_words else 0
        
        professional_patterns = [
            r'[A-Za-z]+-[A-Za-z]+',
            r'[A-Za-z]+_[A-Za-z]+',
            r'[A-Za-z]+\d+',
            r'\d+[A-Za-z]+',
        ]
        
        professional_count = 0
        for pattern in professional_patterns:
            if re.search(pattern, context):
                professional_count += 1
        
        if context_tech_count >= 2:
            return True
        
        if word_diversity > 0.7 and professional_count >= 1:
            return True
        
        return False
    
    def analyze_sensitive_intent(self, sentence: str) -> bool:
        """分析语句是否包含敏感意图"""
        if not sentence:
            return False
        
        sentence_lower = sentence.lower()
        
        has_strong_emotion = self._has_strong_emotion_features(sentence_lower)
        has_strong_action = self._has_strong_action_features(sentence_lower)
        has_target = self._has_target_features(sentence_lower)
        has_causal = self._has_causal_features(sentence_lower)
        
        feature_count = 0
        if has_strong_emotion:
            feature_count += 1
        if has_strong_action:
            feature_count += 1
        if has_target:
            feature_count += 1
        if has_causal:
            feature_count += 1
        
        return feature_count >= 3
    
    def analyze_sensitive_action_context(self, sentence: str, sensitive_word: str) -> bool:
        """分析敏感词是否在敏感动作上下文中"""
        if not sentence or not sensitive_word:
            return False
        
        sentence_lower = sentence.lower()
        word_lower = sensitive_word.lower()
        
        word_pos = sentence_lower.find(word_lower)
        if word_pos == -1:
            return False
        
        window_size = 20
        start = max(0, word_pos - window_size)
        end = min(len(sentence_lower), word_pos + len(word_lower) + window_size)
        context = sentence_lower[start:end]
        
        has_action_verb = self._has_verb_features(context)
        has_time_feature = self._has_time_features(context)
        has_manner_feature = self._has_manner_features(context)
        has_purpose_feature = self._has_purpose_features(context)
        
        if has_action_verb and (has_time_feature or has_manner_feature or has_purpose_feature):
            return True
        
        return False
    
    def _has_strong_emotion_features(self, sentence_lower: str) -> bool:
        """分析语句是否包含强烈的情感特征"""
        if not sentence_lower:
            return False
        
        strong_emotion_patterns = [
            r"非常\s+\w+",
            r"极其\s+\w+",
            r"特别\s+\w+",
            r"严重\s+\w+",
            r"重大\s+\w+",
        ]
        
        for pattern in strong_emotion_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        strong_emotion_adverbs = {"强烈", "极端", "极其", "特别", "非常", "十分", "极度"}
        
        for adverb in strong_emotion_adverbs:
            if adverb in sentence_lower:
                return True
        
        return False
    
    def _has_strong_action_features(self, sentence_lower: str) -> bool:
        """分析语句是否包含强烈的动作特征"""
        if not sentence_lower:
            return False
        
        strong_action_patterns = [
            r"推翻\s+\w+",
            r"颠覆\s+\w+",
            r"攻击\s+\w+",
            r"破坏\s+\w+",
            r"分裂\s+\w+",
        ]
        
        for pattern in strong_action_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        strong_action_verbs = {"制造", "使用", "携带", "策划", "实施", "进行", "组织", "参与"}
        
        for verb in strong_action_verbs:
            if verb in sentence_lower:
                if re.search(rf"{verb}\s+\w+", sentence_lower):
                    return True
        
        return False
    
    def _has_target_features(self, sentence_lower: str) -> bool:
        """分析语句是否包含明确的目标对象特征"""
        if not sentence_lower:
            return False
        
        target_patterns = [
            r"针对\s+\w+",
            r"对\s+\w+",
            r"向\s+\w+",
            r"对于\s+\w+",
        ]
        
        for pattern in target_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        target_objects = {"政府", "政权", "国家", "党", "组织", "机构", "系统", "社会", "人民"}
        
        for target in target_objects:
            if target in sentence_lower:
                return True
        
        return False
    
    def _has_causal_features(self, sentence_lower: str) -> bool:
        """分析语句是否包含因果关系特征"""
        if not sentence_lower:
            return False
        
        causal_patterns = [
            r"因为\s+\w+",
            r"所以\s+\w+",
            r"因此\s+\w+",
            r"由于\s+\w+",
            r"导致\s+\w+",
            r"造成\s+\w+",
            r"引发\s+\w+",
        ]
        
        for pattern in causal_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        causal_connectors = {"为了", "以便", "使得", "以致", "致使", "故而"}
        
        for connector in causal_connectors:
            if connector in sentence_lower:
                return True
        
        return False
    
    def _has_verb_features(self, context: str) -> bool:
        """分析上下文是否包含动词特征"""
        if not context:
            return False
        
        modal_verbs = {"可以", "能够", "应该", "需要", "必须", "可能", "愿意"}
        
        for modal in modal_verbs:
            if modal in context:
                return True
        
        verb_patterns = [
            r"进行\s+\w+",
            r"执行\s+\w+",
            r"实施\s+\w+",
            r"开展\s+\w+",
        ]
        
        for pattern in verb_patterns:
            if re.search(pattern, context):
                return True
        
        return False
    
    def _has_time_features(self, context: str) -> bool:
        """分析上下文是否包含时间特征"""
        if not context:
            return False
        
        time_indicators = {"时", "时候", "时间", "期间", "同时", "之后", "之前", "然后"}
        
        for indicator in time_indicators:
            if indicator in context:
                return True
        
        time_adverbs = {"现在", "立即", "马上", "很快", "即将", "正在", "已经", "曾经", "将要"}
        
        for adverb in time_adverbs:
            if adverb in context:
                return True
        
        return False
    
    def _has_manner_features(self, context: str) -> bool:
        """分析上下文是否包含方式特征"""
        if not context:
            return False
        
        manner_indicators = {"方式", "方法", "手段", "途径", "通过", "利用", "使用", "借助", "依靠"}
        
        for indicator in manner_indicators:
            if indicator in context:
                return True
        
        manner_patterns = [
            r"通过\s+\w+",
            r"利用\s+\w+",
            r"使用\s+\w+",
        ]
        
        for pattern in manner_patterns:
            if re.search(pattern, context):
                return True
        
        return False
    
    def _has_purpose_features(self, context: str) -> bool:
        """分析上下文是否包含目的特征"""
        if not context:
            return False
        
        purpose_indicators = {"目的", "目标", "为了", "以便", "使得", "为了能够", "以达到", "为实现"}
        
        for indicator in purpose_indicators:
            if indicator in context:
                return True
        
        purpose_patterns = [
            r"为了\s+\w+",
            r"以便\s+\w+",
            r"使得\s+\w+",
        ]
        
        for pattern in purpose_patterns:
            if re.search(pattern, context):
                return True
        
        return False
    
    def extract_sentences(self, text: str) -> List[str]:
        """提取文本中的所有语句"""
        if not text:
            return []
        
        sentence_delimiters = r'[。！？；\n]'
        sentences = re.split(sentence_delimiters, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def is_technical_document(self, text: str) -> bool:
        """判断文本是否为技术文档"""
        if not text:
            return False
        
        sentences = self.extract_sentences(text)
        if not sentences:
            return False
        
        technical_sentence_count = 0
        for sentence in sentences:
            semantic_features = self.analyze_semantic_features(sentence, "")
            if semantic_features['is_technical']:
                technical_sentence_count += 1
        
        technical_ratio = technical_sentence_count / len(sentences) if sentences else 0
        text_lower = text.lower()
        
        tech_keywords = set(self.whitelist)
        tech_keyword_count = sum(1 for keyword in tech_keywords if keyword.lower() in text_lower)
        tech_density = tech_keyword_count / len(text) if len(text) > 0 else 0
        
        words = text_lower.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        punctuation_count = sum(1 for char in text if char in '.,;:()[]{}<>《》「」')
        punctuation_density = punctuation_count / len(text) if len(text) > 0 else 0
        
        score = 0
        
        if technical_ratio > 0.5:
            score += 2
        
        if tech_density > 0.05:
            score += 1
        
        if avg_word_length > 3.0:
            score += 1
        
        if 0.05 < punctuation_density < 0.2:
            score += 1
        
        return score >= 3
    
    def is_clearly_sensitive(self, text: str) -> bool:
        """判断文本是否明显敏感"""
        if not text:
            return False
        
        text_lower = text.lower()
        sentences = self.extract_sentences(text)
        
        for sentence in sentences:
            semantic_features = self.analyze_semantic_features(sentence, "")
            if semantic_features['has_sensitive_intent']:
                return True
        
        highly_sensitive_pairs = [
            ("独立", "国家"), ("分裂", "国家"), ("反动", "政府"),
            ("色情", "图片"), ("色情", "视频"), ("毒品", "贩卖"),
            ("恐怖", "袭击"), ("爆炸", "攻击"), ("赌博", "网站")
        ]
        
        for word1, word2 in highly_sensitive_pairs:
            if word1 in text_lower and word2 in text_lower:
                pos1 = text_lower.find(word1)
                pos2 = text_lower.find(word2)
                
                if pos1 != -1 and pos2 != -1:
                    if abs(pos1 - pos2) < 50:
                        return True
        
        return False


class SecurityGuard:
    """安全护栏类，提供敏感词审计、语义分析和提示词防注入功能"""
    
    def __init__(self):
        self.root = TrieNode()
        
        # 加载黑白名单（Redis优先，本地文件备份）
        self.sensitive_words = self._load_sensitive_words()
        self.whitelist = self._load_whitelist()
        self._build_trie()
        
        self.semantic_analyzer = SemanticAnalyzer(self.whitelist)
        self.llm_client = None
        self.vector_store = None
        
        self.logger.info(f"安全护栏初始化完成，加载了 {len(self.sensitive_words)} 个敏感词，{len(self.whitelist)} 个白名单词")
    
    @property
    def logger(self):
        from utils.logger import logger
        return logger
    
    def _load_sensitive_words(self) -> List[str]:
        """加载敏感词列表（Redis优先，本地文件备份）"""
        # 优先从Redis加载
        redis_words = redis_manager.load_sensitive_words()
        if redis_words:
            self.logger.info(f"从Redis加载了 {len(redis_words)} 个敏感词")
            return redis_words
        
        # Redis不可用时从本地文件加载
        file_words = self._load_sensitive_words_from_file()
        if file_words:
            # 将文件数据同步到Redis
            redis_manager.save_sensitive_words(file_words)
            self.logger.info(f"从文件加载了 {len(file_words)} 个敏感词")
            return file_words
        
        # 都不可用时使用空列表
        self.logger.warning("敏感词库为空，将使用空敏感词列表")
        return []
    
    def _load_sensitive_words_from_file(self) -> List[str]:
        """从文件加载敏感词"""
        sensitive_words = []
        total_words = 0
        
        lexicon_dir = os.path.join(os.path.dirname(__file__), "../docs/Vocabulary")
        if os.path.exists(lexicon_dir):
            word_files = [os.path.join(lexicon_dir, f) for f in os.listdir(lexicon_dir) if f.endswith('.txt')]
            
            for word_file in word_files:
                try:
                    with open(word_file, 'r', encoding='utf-8', errors='ignore') as f:
                        words = [line.strip() for line in f if line.strip()]
                        sensitive_words.extend(words)
                        total_words += len(words)
                except Exception as e:
                    self.logger.error(f"加载敏感词文件 {word_file} 失败: {e}")
            
            if word_files:
                self.logger.info(f"从 {len(word_files)} 个文件加载了 {total_words} 个敏感词")
        else:
            self.logger.warning(f"敏感词目录不存在: {lexicon_dir}")
        
        return sensitive_words
    
    def _load_whitelist(self) -> set:
        """加载白名单词汇（Redis优先，本地文件备份）"""
        # 优先从Redis加载
        redis_whitelist = redis_manager.load_whitelist()
        if redis_whitelist:
            self.logger.info(f"从Redis加载了 {len(redis_whitelist)} 个白名单词")
            return redis_whitelist
        
        # Redis不可用时从本地文件加载
        file_whitelist = self._load_whitelist_from_file()
        if file_whitelist:
            # 将文件数据同步到Redis
            redis_manager.save_whitelist(file_whitelist)
            self.logger.info(f"从文件加载了 {len(file_whitelist)} 个白名单词")
            return file_whitelist
        
        # 都不可用时使用默认白名单
        default_whitelist = self._get_default_whitelist()
        self.logger.info(f"使用默认白名单，共 {len(default_whitelist)} 个词")
        return default_whitelist
    
    def _load_whitelist_from_file(self) -> set:
        """从文件加载白名单"""
        whitelist = set()
        whitelist_file = os.path.join(os.path.dirname(__file__), "../docs/whitelist.txt")
        
        if os.path.exists(whitelist_file):
            try:
                with open(whitelist_file, 'r', encoding='utf-8', errors='ignore') as f:
                    words = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                    whitelist.update(words)
            except Exception as e:
                self.logger.error(f"加载白名单文件失败: {e}")
        
        return whitelist
    
    def _get_default_whitelist(self) -> set:
        """获取默认白名单"""
        return {
            "admin", "administrator", "system", "系统", "管理", "管理系统",
            "监控", "监控系统", "操作", "操作日志", "日志", "信息",
            "数据", "数据库", "服务器", "server", "database", "log",
            "用户", "安全", "认证", "授权", "审计", "访问", "控制"
        }
    
    def update_sensitive_words(self, new_words: List[str]):
        """动态更新敏感词库（Redis + 本地文件双存储）"""
        if not new_words:
            return
        
        # 更新内存缓存
        self.sensitive_words.extend(new_words)
        self.sensitive_words = list(set(self.sensitive_words))
        
        # 双存储更新：Redis优先，本地文件备份
        redis_success = redis_manager.add_sensitive_words(new_words)
        
        if redis_success:
            self.logger.info(f"动态更新敏感词库（Redis+文件），新增 {len(new_words)} 个敏感词，当前总数: {len(self.sensitive_words)}")
        else:
            self.logger.info(f"动态更新敏感词库（本地文件），新增 {len(new_words)} 个敏感词，当前总数: {len(self.sensitive_words)}")
        
        # 重建Trie树
        self.root = TrieNode()
        self._build_trie()
    
    def update_whitelist(self, new_words: List[str]):
        """动态更新白名单（Redis + 本地文件双存储）"""
        if not new_words:
            return
        
        # 更新内存缓存
        self.whitelist.update(new_words)
        
        # 双存储更新：Redis优先，本地文件备份
        redis_success = redis_manager.add_whitelist_words(new_words)
        
        if redis_success:
            self.logger.info(f"动态更新白名单（Redis+文件），新增 {len(new_words)} 个词，当前总数: {len(self.whitelist)}")
        else:
            self.logger.info(f"动态更新白名单（本地文件），新增 {len(new_words)} 个词，当前总数: {len(self.whitelist)}")
        
        # 同时更新到本地文件
        self._save_whitelist_to_file(new_words)
    
    def _save_whitelist_to_file(self, new_words: List[str]):
        """将新增的白名单词汇保存到文件"""
        if not new_words:
            return
        
        whitelist_file = os.path.join(os.path.dirname(__file__), "../docs/whitelist.txt")
        
        try:
            # 读取现有文件内容
            existing_lines = []
            original_words = set()
            if os.path.exists(whitelist_file):
                with open(whitelist_file, 'r', encoding='utf-8', errors='ignore') as f:
                    existing_lines = f.readlines()
                
                # 提取原始文件中的词汇
                for line in existing_lines:
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        original_words.add(stripped_line)
            
            # 重新组织文件内容，保持注释和分类
            new_content = []
            current_section = []
            
            for line in existing_lines:
                stripped_line = line.strip()
                if stripped_line.startswith('#'):
                    # 如果是注释行，先输出当前分类的词汇
                    if current_section:
                        new_content.extend(sorted(current_section))
                        new_content.append('')  # 添加空行分隔
                        current_section = []
                    new_content.append(line.rstrip())
                elif stripped_line and stripped_line in original_words:
                    # 保留原始文件中的词汇
                    current_section.append(stripped_line)
                elif stripped_line:
                    # 跳过不在原始文件中的词汇
                    continue
            
            # 处理最后一个分类
            if current_section:
                new_content.extend(sorted(current_section))
            
            # 添加新词汇到"其他词汇"分类
            new_words_to_add = [word for word in new_words if word not in original_words]
            if new_words_to_add:
                new_content.append('')
                new_content.append('# 其他词汇')
                new_content.extend(sorted(new_words_to_add))
            
            # 写入文件
            with open(whitelist_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_content) + '\n')
            
            from utils.logger import logger
            logger.info(f"白名单已持久化到文件，新增 {len(new_words_to_add)} 个词")
            
        except Exception as e:
            from utils.logger import logger
            logger.error(f"保存白名单到文件失败: {e}")
    
    def _build_trie(self):
        """构建Trie树"""
        for word in self.sensitive_words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
    
    def check_sensitive_content(self, text: str, context: Optional[str] = None) -> Tuple[bool, List[str]]:
        """检查文本是否包含敏感词"""
        if not text:
            return False, []
        
        dfa_found = []
        text_length = len(text)
        
        def is_valid_sensitive_word(word):
            if len(word) <= 1:
                return False
            if word.isdigit():
                return False
            if word.lower() in [w.lower() for w in self.whitelist]:
                return False
            return True
        
        for i in range(text_length):
            node = self.root
            j = i
            current_word = ""
            
            while j < text_length:
                char = text[j]
                if char not in node.children:
                    break
                
                node = node.children[char]
                current_word += char
                
                if node.is_end and is_valid_sensitive_word(current_word):
                    dfa_found.append(current_word)
                
                j += 1
        
        dfa_found = list(set(dfa_found))
        
        if not dfa_found:
            return False, []
        
        context_audit_result = self._context_based_audit(text, dfa_found, context)
        
        if not context_audit_result and dfa_found:
            self.update_whitelist(dfa_found)
            from utils.logger import logger
            logger.info(f"上下文审计确认是误报，已将检测到的词加入白名单: {', '.join(dfa_found)}")
            return False, []
        
        all_found = dfa_found.copy()
        
        has_sensitive = len(all_found) > 0 and context_audit_result
        
        if has_sensitive:
            from utils.logger import logger
            logger.info(f"敏感词检测命中: {', '.join(all_found)}")
            if context:
                logger.info(f"上下文: {context}")
        
        return has_sensitive, all_found
    
    def _context_based_audit(self, text: str, detected_words: List[str], context: Optional[str] = None) -> bool:
        """基于语句语义的上下文审计"""
        if not detected_words:
            return False
        
        sentences = self._extract_sentences(text)
        
        real_sensitive_words = []
        for word in detected_words:
            if word.lower() in [w.lower() for w in self.whitelist]:
                continue
            
            containing_sentences = []
            for sentence in sentences:
                if word.lower() in sentence.lower():
                    containing_sentences.append(sentence)
            
            if not containing_sentences:
                continue
            
            is_sensitive_in_any_sentence = False
            for sentence in containing_sentences:
                if self._is_sentence_sensitive(sentence, word):
                    is_sensitive_in_any_sentence = True
                    break
            
            if is_sensitive_in_any_sentence:
                real_sensitive_words.append(word)
        
        return len(real_sensitive_words) > 0
    
    def _extract_sentences(self, text: str) -> List[str]:
        """提取文本中的所有语句"""
        return self.semantic_analyzer.extract_sentences(text)
    
    def _is_sentence_sensitive(self, sentence: str, sensitive_word: str) -> bool:
        """判断语句中敏感词是否真的敏感"""
        if not sentence or not sensitive_word:
            return False
        
        # 检查敏感词是否在白名单中
        if sensitive_word in self.whitelist:
            return False
        
        # 对于其他敏感词，使用正常的判断逻辑
        semantic_features = self.semantic_analyzer.analyze_semantic_features(sentence, sensitive_word)
        
        if semantic_features['is_technical']:
            return False
        
        if semantic_features['in_technical_context']:
            return False
        
        if semantic_features['has_sensitive_intent']:
            return True
        
        if semantic_features['in_sensitive_action_context']:
            return True
        
        return True
    
    def _is_technical_document(self, text: str) -> bool:
        """判断文本是否为技术文档"""
        return self.semantic_analyzer.is_technical_document(text)
    
    def _is_clearly_sensitive(self, text: str) -> bool:
        """判断文本是否明显敏感"""
        return self.semantic_analyzer.is_clearly_sensitive(text)
    
    def filter_sensitive_content(self, text: str, replace_char: str = '*') -> str:
        """过滤文本中的敏感词"""
        if not text:
            return text
        
        result = list(text)
        text_length = len(text)
        
        i = 0
        while i < text_length:
            node = self.root
            j = i
            end_pos = -1
            
            while j < text_length:
                char = text[j]
                if char not in node.children:
                    break
                
                node = node.children[char]
                
                if node.is_end:
                    end_pos = j
                
                j += 1
            
            if end_pos != -1:
                for k in range(i, end_pos + 1):
                    result[k] = replace_char
                i = end_pos + 1
            else:
                i += 1
        
        return ''.join(result)
    
    def check_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """检查提示词注入攻击"""
        if not text:
            return False, []
        
        injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"prompt injection",
            r"override instructions",
            r"bypass security",
            r"change system prompt",
            r"forget previous",
            r"ignore all previous",
            r"disregard previous",
            r"reset system",
            r"system:.*",
            r"assistant:.*",
            r"user:.*"
        ]
        
        found_patterns = []
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        
        return len(found_patterns) > 0, found_patterns
    
    def sanitize_prompt(self, text: str) -> str:
        """清理提示词，防止注入攻击"""
        if not text:
            return text
        
        injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"prompt injection",
            r"override instructions",
            r"bypass security",
            r"change system prompt",
            r"forget previous",
            r"ignore all previous",
            r"disregard previous",
            r"reset system"
        ]
        
        sanitized_text = text
        for pattern in injection_patterns:
            sanitized_text = re.sub(pattern, "", sanitized_text, flags=re.IGNORECASE)
        
        sanitized_text = re.sub(r"system:.*?\n", "", sanitized_text, flags=re.IGNORECASE | re.DOTALL)
        sanitized_text = re.sub(r"assistant:.*?\n", "", sanitized_text, flags=re.IGNORECASE | re.DOTALL)
        sanitized_text = re.sub(r"user:.*?\n", "", sanitized_text, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized_text.strip()


_security_guard_instance = None

def create_security_guard():
    """创建安全护栏实例（单例模式）"""
    global _security_guard_instance
    if _security_guard_instance is None:
        _security_guard_instance = SecurityGuard()
    return _security_guard_instance
