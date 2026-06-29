import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)


class Logger:
    """
    日志记录器类
    - 使用 TimedRotatingFileHandler 按天轮转日志
    - 自动保留最近N天的日志
    """

    def __init__(self, filename: str, level: str = 'info', when: str = 'D', back_count: int = 10):
        """
        初始化日志记录器

        Args:
            filename: 日志文件路径
            level: 日志级别 (debug/info/warning/error)
            when: 轮转时间单位 (S-秒, M-分, H-小时, D-天, W-周)
            back_count: 保留的备份数量
        """
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self._get_level(level))

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=filename,
            when=when,
            interval=1,
            backupCount=back_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self._get_level(level))
        file_handler.setFormatter(formatter)

        # 只添加文件处理器，移除控制台处理器
        self.logger.addHandler(file_handler)

    def _get_level(self, level: str) -> int:
        """获取日志级别"""
        levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        return levels.get(level.lower(), logging.INFO)


LOG_FILE = log_dir / f"log_audit_qa_{datetime.now().strftime('%Y%m%d')}.log"

logger_instance = Logger(
    filename=str(LOG_FILE),
    level='info',
    when='D',
    back_count=10
)

logger = logger_instance.logger


def log_qa_interaction(user_id: str, session_id: str, user_input: str, ai_output: str):
    """
    记录问答交互日志

    Args:
        user_id: 用户ID
        session_id: 会话ID
        user_input: 用户输入
        ai_output: AI输出
    """
    logger.info(f"USER:{user_id} SESSION:{session_id} INPUT:{user_input}")
    logger.info(f"USER:{user_id} SESSION:{session_id} OUTPUT:{ai_output}")