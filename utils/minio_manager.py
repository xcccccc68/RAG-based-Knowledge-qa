import boto3
from botocore.client import Config
from configs.config import settings
from utils.logger import logger


class MinioManager:
    def __init__(self):
        logger.info(f"初始化Minio管理器，endpoint={settings.MINIO_ENDPOINT}")
        self.client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        logger.info("Minio管理器初始化成功")

    def list_objects(self, prefix: str = ""):
        try:
            logger.info(f"列出Minio对象，prefix={prefix}")
            response = self.client.list_objects_v2(
                Bucket=settings.MINIO_BUCKET_NAME,
                Prefix=prefix
            )
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            logger.info(f"成功列出{len(objects)}个对象")
            return objects
        except Exception as e:
            logger.error(f"列出对象失败: {e}")
            return []


# 单例实例
_minio_manager_instance = None

def create_minio_manager():
    """
    创建Minio管理器实例（单例模式）
    
    Returns:
        MinioManager: Minio管理器实例
    """
    global _minio_manager_instance
    if _minio_manager_instance is None:
        _minio_manager_instance = MinioManager()
    return _minio_manager_instance
