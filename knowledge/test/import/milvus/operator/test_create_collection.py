from pymilvus import MilvusClient
from knowledge.utils.bge_me_embedding_util import get_beg_m3_embedding_model

if __name__ == '__main__':

    # 1. 定义MilVusClient对象
    client = MilvusClient(uri="http://192.168.200.130:19530")

    # 2. 创建集合
    if client.has_collection(collection_name="test_create_collection_v1"):
        client.drop_collection(collection_name="test_create_collection_v1")
    client.create_collection(
        collection_name="test_create_collection_v1",
        dimension=1024,  # The vectors we will use in this demo has 768 dimensions
    )
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956."
    ]

    embedding_model = get_beg_m3_embedding_model()
    vector = embedding_model.encode_documents(docs)

    # 3. 构建数据

    data = [
        {"id": 1, "vector": vector['dense'][0].tolist(), "text": docs[0], "subject": "history"}
    ]

    res = client.insert(collection_name="test_create_collection_v1", data=data)

    print(res)
