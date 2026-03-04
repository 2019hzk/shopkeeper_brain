import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

from typing import Optional
from pymilvus import MilvusClient

milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> Optional[MilvusClient]:
    global milvus_client

    # 1.判断
    if milvus_client is not None:
        return milvus_client

    # 2. 获取参数
    try:
        milvus_uri = os.getenv('MILVUS_URL', 'http://192.168.200.130:19530')

        # 3. 定义MilVusClient对象
        milvus_client = MilvusClient(
            uri=milvus_uri
        )

        return milvus_client
    except Exception as e:
        logger.error(f"MilVus客户端创建失败:{str(e)}")
        return None
