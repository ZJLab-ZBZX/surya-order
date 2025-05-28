import os
import oss2
from oss2.models import SimplifiedObjectInfo
from typing import List, Optional, Union

class OSSDownloader:
    def __init__(self, access_key_id: str, access_key_secret: str, 
                 endpoint: str, bucket_name: str):
        """初始化OSS客户端"""
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)
        
    def _list_files_recursive(self, oss_dir: str) -> List[SimplifiedObjectInfo]:
        """递归列出指定目录及其子目录下的所有文件"""
        # 确保目录以斜杠结尾
        if not oss_dir.endswith('/'):
            oss_dir += '/'
            
        # 列举文件
        files = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=oss_dir):
            if isinstance(obj, oss2.models.SimplifiedObjectInfo):
                # 过滤掉目录对象（以斜杠结尾的）
                if obj.key.endswith('/'):
                    continue
                if 'formular-generated' in obj.key:
                    continue
                if 'table' in obj.key:
                    continue
                if 'page' in obj.key:
                    files.append(obj)
                if obj.key.endswith('layout.json'):
                    files.append(obj)

        return files
    
    def download_file(self, oss_file_path: str, local_file_path: str) -> bool:
        """下载单个文件"""
        try:
            # 检查本地路径是否是目录
            if os.path.isdir(local_file_path):
                # print(f"错误: 本地路径是目录 - {local_file_path}")
                return False
                
            # 创建本地目录
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)
            
            # if not os.path.exists(local_file_path):
            # 下载文件
            result = self.bucket.get_object_to_file(oss_file_path, local_file_path)
            if result.status == 200:
                # print(f"成功下载文件: {oss_file_path} -> {local_file_path}")
                return True
            else:
                print(f"下载失败，状态码: {result.status}")
                return False
        except Exception as e:
            print(f"下载出错: {str(e)}")
            return False
    
    def download_directory(self, oss_dir: str, local_dir: str) -> int:
        """下载指定目录及其子目录下的所有文件"""
        # 获取文件列表（递归）
        files = self._list_files_recursive(oss_dir)
        
        # 计算相对路径并下载
        success_count = 0
        for file in files:
            # 计算相对路径（去掉原始OSS目录前缀）
            relative_path = os.path.relpath(file.key, oss_dir)
            local_file = os.path.join(local_dir, relative_path)

            # 下载文件
            if self.download_file(file.key, local_file):
                success_count += 1
        
        # print(f"共下载 {success_count}/{len(files)} 个文件")
        return success_count


# 配置信息
ACCESS_KEY_ID = 'akA4U9SbWZPRX250'
ACCESS_KEY_SECRET = 'kyDGzF26GqpJ54o2WSwJSG9wfHKT0p'
ENDPOINT = 'http://oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com'
BUCKET_NAME = 'train1'

# 创建下载器实例
downloader = OSSDownloader(ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME)

# 示例1: 下载单个文件
# oss_file = 'mathpix-result/batch_002/265817/265817.lines.json'
# local_file = '/root/surya/data/rawdata/265817/265817.lines.json'
# downloader.download_file(oss_file, local_file)
    
# 示例2: 下载整个目录及其子目录
# oss_directory = 'dataset-train/mathpix-data'
# local_directory = '/root/surya/data/train_dataset/rawdata'
# downloader.download_directory(oss_directory, local_directory)