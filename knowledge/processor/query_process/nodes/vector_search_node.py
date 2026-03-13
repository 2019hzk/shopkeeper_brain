import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model,generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 获取嵌入模型&milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if embedding_model is None or milvus_client is None:
            return state

        # 3. 对问题向量化(稀疏向量做了字典的处理) 注意：【generate_hybrid_embeddings】
        embedding_result = generate_hybrid_embeddings(embedding_model,embedding_documents=[validated_query])
        if not embedding_result:
            return state

        # 4. 构建过滤表达式
        item_name_filter_expr = self._item_name_filter(validate_item_names)
        # 5. 创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_expr,
            limit=5
        )

        # 4. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            # ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=["chunk_id", "content", "item_name"]
        )
        if not reps or not reps[0]:
            return state

        # 5. 更新state的embedding_chunks
        state['embedding_chunks'] = reps[0]
        return state

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    def _item_name_filter(self, validate_item_names: List[str]) -> str:
        # filter = 'item_name in '"商品A", "商品B"'
        #  '"商品A", "商品B"'
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        # filter = 'item_name in ["商品A", "商品B", "商品C"]'v   # 标量字段（动态字段）进行过滤
        return f" item_name in [{quoted}]"


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"] #对齐
    }

    vector_search = VectorSearchNode()

    result = vector_search.process(state)
    #
    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))
