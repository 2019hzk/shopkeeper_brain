from typing import Dict, List, Any

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model


class BgeEmbeddingChunksNode(BaseNode):
    """
    BgeEmbeddingChunksNode主要职责：

    1. 获取所有的chunks拼接要向量的内容
    2. 批量嵌入 chunk的（embedding_content:item_name + chunk.get('content')）
    3. 将所有chunk嵌入后的向量值，存储到列表中，在返回给下一个节点用
    """

    name = "beg_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        validated_chunks, config = self._validate_get_inputs(state)

        # 2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, 'embedding_batch_size', 16)

        # 3. 准备分批嵌入(pineline)
        # 待嵌入的所有数据chunks=[1,2,3,4,5,6]
        # 阈值：3
        # 第一批：[1,2,3]
        # 第二批：[4,5,6]

        # 待嵌入的所有数据chunks=[1,2]
        # 阈值：3
        # 第一批：[1,2,3]
        total_length = len(validated_chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i:i + embedding_batch_chunk_size]
            # 拼接要嵌入的内容 向量嵌入的内容 把嵌入的向量注入到chunk中
            batch_chunks = self._process_batch_chunks(batch, embedding_batch_chunk_size, total_length)
            final_chunks.extend(batch_chunks)

        # 4. 更新&返回state
        return state

    def _process_batch_chunks(self, batch: List[Dict[str, Any]], embedding_batch_chunk_size: int, total_length: int):

        # 1. 循环处理所有chunk的要嵌入的内容拼接
        embedding_contents = []
        for _, chunk in enumerate(batch):
            # 1.1 提取content
            content = chunk.get('content')

            # 1.2 提取item_name
            item_name = chunk.get('item_name')

            # 1.3 拼接要嵌入的最终内容
            embedding_content = f"{item_name}\n{content}"

            embedding_contents.append(embedding_content)

        # 2. 批量嵌入
        bge_m3_model = get_beg_m3_embedding_model()
        embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)

        # 3. 循环处理所有chunk的向量以及注入到每一个chunk中
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构csr矩阵&获取稀疏向量
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

            # 3.3 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))

            # 3.4 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        return batch

    def _validate_get_inputs(self, state: ImportGraphState):
        config = get_config()
        # 1.获取chunks
        chunks = state.get('chunks')

        # 2.校验chunks/校验item_name也可以(其实不用)因为有安全边界的设置
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        # 3. 返回chunks
        return chunks, config
