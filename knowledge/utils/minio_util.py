import os
import logging

from minio.error import S3Error
from minio import Minio
from dotenv import load_dotenv
load_dotenv()


def get_minio_client():
    # 1. 实例化MinIO客户端
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT"),  # 服务端端口
                       access_key=os.getenv("MINIO_ACCESS_KEY"),  # 访问的key
                       secret_key=os.getenv("MINIO_SECRET_KEY"),  # 访问的秘钥
                       secure=False  # 使用HTTP协议
                       )

        # 2. 判断通是否存在
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        bucket_exists = client.bucket_exists(bucket_name)
        if not bucket_exists:
            client.make_bucket(bucket_name)
            logging.info(f"桶:{bucket_name} 不存在")
        else:
            logging.info(f"桶:{bucket_name} 已经存在")

        # 3. 返回MinIo客户端
        return client
    except S3Error as e:
        logging.error("MinIO客户端创建失败")
        return None
