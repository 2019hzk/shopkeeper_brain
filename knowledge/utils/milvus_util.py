import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

from typing import Optional
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
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



# ------------------------------------------------------------------
# 混合检索
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 创建混合检索请求
# ------------------------------------------------------------------
def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  limit=5):
    """
    创建混合搜索请求

    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    :param dense_params: 稠密向量搜索参数，默认为None
    :param sparse_params: 稀疏向量搜索参数，默认为None
    :param expr: 查询表达式，默认为None
    :param limit: 返回结果数量限制，默认为5
    :return: 包含稠密和稀疏搜索请求的列表
    """
    # 默认参数
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 创建稠密向量搜索请求
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 创建稀疏向量搜索请求
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


# ------------------------------------------------------------------
# 执行混合检索请求
# ------------------------------------------------------------------
def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=False,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行混合搜索
    :param collection_name: 集合名称
    :param search_requests: 搜索请求列表，通常是[dense_req, sparse_req]
    :param ranker_weights: 权重排名器的权重，默认为(0.5, 0.5)
    :param norm_score: 是否对分数进行归一化，默认为True
    :param limit: 返回结果数量限制，默认为5
    :param output_fields: 要返回的字段列表，默认为None
    :param search_params: 搜索参数，默认为None
    :return: 搜索结果
    """
    try:
        # 创建权重融合排序器
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认输出字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行搜索
        res = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        # 动态计算所有查询返回的结果总数
        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"Milvus 混合搜索完成，共处理 {len(res) if res else 0} 个查询，总计找到 {total_hits} 个结果")
        return res
    except Exception as e:
        logger.error(f"执行Milvus混合搜索时发生错误: {e}")
        return None





def fetch_chunks_by_chunk_ids(
    collection_name: str,
    chunk_ids,
    *,
    output_fields=None,
    batch_size: int = 100,
):
    """
    通过 chunk_id（主键）批量查询切片字段
    返回：List[dict]，元素为 Milvus entity（字段字典）。
    """
    client = get_milvus_client()
    if not collection_name:
        return []
    if output_fields is None:
        # 默认返回字段需与 collection schema 保持一致
        output_fields = ["chunk_id", "content", "title", "file_title", "item_name"]


    results = []
    # 分批，避免一次性过大
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i : i + batch_size]
        #  get（主键直取）
        try:
            got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
            if got:
                results.extend(got)
            continue
        except Exception as e:
            logger.error(f"Milvus get() 查询失败: {e}")

    return results
