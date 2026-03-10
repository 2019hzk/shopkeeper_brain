"""
Milvus 混合检索（Dense + Sparse）演示

演示流程：
    1. 连接 Milvus
    2. 创建集合（dense_vector + sparse_vector + item_name）
    3. 生成 BGE-M3 混合嵌入（稠密 + 稀疏）
    4. 插入数据
    5. 混合检索（WeightedRanker 融合两路结果）


cosine:  ip/||A|| * ||B||
ip:维度*权重+
L2:模长
归一化：各自向量的模长是1 ||A|| =1  ||B||，cosine=ip. 没有做归一化cosine和IP不相等

bgem3嵌入模型，计算出来的分数值都是归一化的。
milvus+bgem3:度量类型：cosine=ip
如果用的不是bgem3嵌入模型。需要自己去做归一化，cosine和ip相等。否则不相等。


"""

from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    WeightedRanker,
    RRFRanker,
)

from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  配置常量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLLECTION_NAME = "hybrid_search_demo"
MILVUS_URI = "http://192.168.200.130:19530"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  嵌入工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_sparse_vectors(csr, count):
    """将 CSR 稀疏矩阵按行解析为 dict 列表。"""
    result = []
    for i in range(count):
        start = csr.indptr[i]
        end = csr.indptr[i + 1]
        token_ids = csr.indices[start:end].tolist()
        weights = csr.data[start:end].tolist()
        result.append(dict(zip(token_ids, weights)))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  初始化：建集合 + 插入数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_collection(client, embedding_model, item_names):
    """创建集合、建索引、插入数据。"""

    # 生成嵌入
    embeddings = embedding_model.encode_documents(item_names)
    dense_vectors = [emb.tolist() for emb in embeddings["dense"]]
    sparse_vectors = parse_sparse_vectors(embeddings["sparse"], len(item_names))

    # 如果集合已存在则删除
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # Schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("item_name", DataType.VARCHAR, max_length=256)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=len(dense_vectors[0]))
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # 索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    # 创建集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # 插入数据
    data = [
        {
            "item_name": item_names[i],
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i],
        }
        for i in range(len(item_names))
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"集合 [{COLLECTION_NAME}] 创建成功，插入 {len(data)} 条数据")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  混合检索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hybrid_search(client, embedding_model, query_text, limit=5):
    """执行混合检索，返回融合排序后的结果列表。

    Args:
        client: MilvusClient 实例。
        embedding_model: BGE-M3 嵌入模型。
        query_text: 查询文本。
        limit: 最终返回条数。

    Returns:
        [{"item_name": "...", "score": 0.87}, ...]
    """
    # 生成查询向量
    query_embeddings = embedding_model.encode_queries([query_text])
    q_dense = query_embeddings["dense"][0].tolist()
    q_sparse = parse_sparse_vectors(query_embeddings["sparse"], 1)[0]

    # 构建两路检索请求
    # 1. 构建稠密向量场的请求
    dense_req = AnnSearchRequest(
        data=[q_dense],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[q_sparse],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}},
        limit=limit,
    )

    # 融合排序
    hits = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[sparse_req, dense_req],
        ranker=WeightedRanker(0.5, 0.5),
        limit=limit,
        output_fields=["item_name"],
    )


    # 解析结果
    return [
        {"item_name": hit["entity"]["item_name"], "score": hit["distance"]}
        for hit in hits[0]
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 模拟商品名称库
    item_names = [
        "苏泊尔RS-12数字万用表",
        "胜利VC890D数字万用表"
    ]

    # 初始化
    embedding_model = get_beg_m3_embedding_model()
    client = MilvusClient(uri=MILVUS_URI)
    # init_collection(client, embedding_model, item_names)

    # # 查询
    query = "万用表"
    print(f"\n查询: '{query}'")
    print("=" * 50)
    #
    results = hybrid_search(client, embedding_model, query)
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r['item_name']}  (score: {r['score']:.4f})")