"""

create_collection(collection_name,dim)---自动版本，简单入门（快速构建数据【AUTOINDEX】 并且进行查询。）
create_collection(collection_name,schema)---半自动（不能支持查询 但是可以将结构创建出来）
create_collection(collection_name,schema,index_params)---自动 高阶版本（结构创建出来、自己指定了索引，所以支持查询）---最终使用的。



"""

import time

from pymilvus import MilvusClient, DataType
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model

if __name__ == '__main__':

    # 1. 定义MilVusClient对象
    client = MilvusClient(uri="http://192.168.200.130:19530")

    # 2. 删除集合
    if client.has_collection(collection_name="test_create_collection_v3"):
        client.drop_collection(collection_name="test_create_collection_v3")

    # 3. 先创建schema
    schema = client.create_schema(enable_dynamic_field=True)

    # 3.1 主键字段
    schema.add_field(
        field_name="my_id",  # 字段名字
        datatype=DataType.INT64,  # 值的类型
        is_primary=True,  # 是否是主键
        auto_id=True,  # 是否是自增
    )
    # 3.2 添加向量字段
    schema.add_field(
        field_name="my_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,

    )

    # 3.3 添加标量字段
    schema.add_field(
        field_name="my_varchar",
        datatype=DataType.VARCHAR,
        max_length=512
    )

    # 2.4 添加索引
    index_params = MilvusClient.prepare_index_params()

    # 向量字段添加索引
    index_params.add_index(
        field_name="my_vector",  # Name of the vector field to be indexed
        index_type="IVF_FLAT",  # Type of the index to create
        index_name="vector_index",  # Name of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "nlist": 64,  # Number of clusters for the index
        }
    )
    # 标量字段添加索引（TODO）
    index_params.add_index(
        field_name="my_varchar",  # Name of the vector field to be indexed
        index_type="INVERTED",  # Type of the index to create
        index_name="inverted_index",  # Name of the index to create
    )

    # 高级自动
    client.create_collection(
        collection_name="test_create_collection_v3",
        schema=schema,
        index_params=index_params  # 搜索的时候用
    )

    # 4. 构建数据文档

    docs = [
        "人工智能这一学科于 1956 年被确立为一门学术学科。",
        "艾伦·图灵是首位对人工智能领域进行深入研究的人士。.",
        "图灵出生于伦敦的梅达维尔，他在英格兰南部长大。.",
    ]

    embedding_model = get_beg_m3_embedding_model()
    vector = embedding_model.encode_documents(docs)

    # 5. 构建数据
    data = [
        {"my_vector": vector['dense'][i].tolist(), "my_varchar": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]

    # 6. 插入数据
    res = client.insert(collection_name="test_create_collection_v3", data=data)

    print(res)

    # time.sleep(2)
    # query_vectors = embedding_model.encode_queries(["谁是艾伦图灵?"])
    #
    # # 7. 查询（向量字段的查询）
    # res = client.search(
    #     collection_name="test_create_collection_v3",  # Collection name
    #     anns_field="my_vector",
    #     data=[query_vectors['dense'][0].tolist()],
    #     output_fields=["my_varchar", "subject"],  # Query vector
    #     limit=3,
    # )
    # print(res)
