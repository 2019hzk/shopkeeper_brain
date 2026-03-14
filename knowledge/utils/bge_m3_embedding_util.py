import os, logging
from typing import Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

load_dotenv()

bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None


def get_beg_m3_embedding_model():
    global bge_m3_ef

    # 1.判断
    if bge_m3_ef is not None:
        return bge_m3_ef

    # 2. 获取参数
    model_name = os.getenv('BGE_M3_PATH', 'BAAI/bge-m3')
    device = os.getenv('BGE_DEVICE', 'cpu')
    use_fp16_str = os.getenv('BGE_FP16', 'False')
    use_fp16 = use_fp16_str.lower() in ('true', '1', 'yes')
    try:
        # 3. 定义嵌入模型对象 # 默认维度1024
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16
        )
    except Exception as e:
        return None

    # 4. 返回
    return bge_m3_ef


def generate_hybrid_embeddings(embedding_model: BGEM3EmbeddingFunction, embedding_documents: List[str]):
    """
    为文本生成向量嵌入
    :param embedding_model: 嵌入模型(这里使用BGEM3)
    :param embedding_documents: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    try:
        # 1. 生成嵌入
        embedding_result = embedding_model.encode_documents(embedding_documents)

        processed_sparse_result = []
        # 2. 遍历每一个文档
        for index in range(len(embedding_documents)):
            # 2.1 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            # a) 行索引
            ind_ptr = csr_array.indptr

            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]

            # c) 获取token_id
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()

            # d) 获取权重
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            # 2.2 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))

            processed_sparse_result.append(sparse_vector)

        # 3. 返回
        return {
            "dense": [den.tolist() for den in embedding_result["dense"]],
            "sparse": processed_sparse_result
        }
    except Exception as e:
        return None


if __name__ == '__main__':
    model = get_beg_m3_embedding_model()

    dict = generate_hybrid_embeddings(model, embedding_documents=["我是中国人", "我喜欢自由"])

    print(dict)