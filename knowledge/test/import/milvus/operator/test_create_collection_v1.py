import time

from pymilvus import MilvusClient
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model

if __name__ == '__main__':

    # 1. 定义MilVusClient对象
    client = MilvusClient(uri="http://192.168.200.130:19530")

    # 2. 创建集合
    if client.has_collection(collection_name="test_create_collection_v2"):
        client.drop_collection(collection_name="test_create_collection_v2")

    # 全自动
    client.create_collection(
        collection_name="test_create_collection_v2",
        dimension=1024,  # The vectors we will use in this demo has 768 dimensions
        auto_id=True  # 自增值(构建数据的时候在指定该字段)
    )
    docs = [
        "人工智能这一学科于 1956 年被确立为一门学术学科。",
        "艾伦·图灵是首位对人工智能领域进行深入研究的人士。.",
        "图灵出生于伦敦的梅达维尔，他在英格兰南部长大。.",
    ]

    embedding_model = get_beg_m3_embedding_model()
    vector = embedding_model.encode_documents(docs)

    # 3. 构建数据
    data = [
        {"vector": vector['dense'][i].tolist(), "text": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]

    res = client.insert(collection_name="test_create_collection_v2", data=data)
    time.sleep(2)
    query_vectors = embedding_model.encode_queries(["谁是艾伦图灵?"])

    res = client.search(
        collection_name="test_create_collection_v2",  # target collection
        data=[query_vectors['dense'][0].tolist()],  # query vectors
        limit=1,  # number of returned entities
        output_fields=["text"],  # specifies fields to be returned
    )
    print(res)
