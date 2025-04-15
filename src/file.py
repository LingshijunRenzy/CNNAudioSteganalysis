import os
import io
import tempfile
from src.config import config as cfg

class FileHandler:
    """文件处理工具类，支持从本地或OSS获取文件"""
    
    @staticmethod
    def get_file(path, return_type='path'):
        """
        从本地或OSS获取文件
        
        Args:
            path (str): 文件路径
            return_type (str): 返回类型，'path'返回文件路径，'bytes'返回二进制内容
            
        Returns:
            str or bytes: 根据return_type返回文件路径或二进制内容
        """
        storage_config = cfg.get(key='STORAGE_CONFIG', default={'type': 'local'})
        storage_type = storage_config.get('type', 'local')
        
        # 检查是否有OSS专用路径（由AudioStegDataset提供）
        oss_path = None
        if isinstance(path, dict) and 'oss_path' in path:
            oss_path = path['oss_path']
            path = path['path']
        
        # 根据存储类型处理
        if storage_type == 'local':
            # 本地文件直接返回路径或读取内容
            if return_type == 'path':
                return path
            elif return_type == 'bytes':
                with open(path, 'rb') as f:
                    return f.read()
        
        elif storage_type == 'oss':
            # 从OSS获取文件
            try:
                import oss2
            except ImportError:
                raise ImportError("请安装oss2库: pip install oss2")
            
            oss_config = storage_config.get('oss', {})
            auth = oss2.Auth(oss_config.get('access_key_id'), oss_config.get('access_key_secret'))
            bucket = oss2.Bucket(auth, oss_config.get('endpoint'), oss_config.get('bucket_name'))
            
            # 使用已有的OSS路径或构建OSS对象路径
            object_path = oss_path or path
            
            # 如果没有直接的OSS路径，则处理路径转换
            if not oss_path:
                strip_prefix = oss_config.get('path_strip_prefix', '')
                if strip_prefix and path.startswith(strip_prefix):
                    object_path = path[len(strip_prefix):]
                object_path = object_path.lstrip('/')
            
            if return_type == 'bytes':
                # 直接返回对象内容
                return bucket.get_object(object_path).read()
            elif return_type == 'path':
                # 下载到临时文件并返回路径
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1])
                temp_path = temp_file.name
                temp_file.close()
                
                bucket.get_object_to_file(object_path, temp_path)
                return temp_path
        
        else:
            raise ValueError(f"不支持的存储类型: {storage_type}")