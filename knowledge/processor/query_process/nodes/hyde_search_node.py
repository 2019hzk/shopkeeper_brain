import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model


class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[embedding_document])

        if not embedding_result:
            return state

        # 5. 获取item_name的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=embedding_result['dense'][0],
                                                               sparse_vector=embedding_result['sparse'][0],
                                                               expr=item_name_filtered_expr)

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(milvus_client,
                                           collection_name=self.config.chunks_collection,
                                           search_requests=hybrid_search_requests,
                                           norm_score=True,
                                           output_fields=["chunk_id", "content", "item_name"])

        if not reps or not reps[0]:
            return state

        # 8. 更新state
        state['hyde_embedding_chunks'] = reps[0]

        # 9. 返回更新后的state
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

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:

        # 1. 获取LLM客户端
        llm_client = get_llm_client()

        # 2. 判断
        if llm_client is None:
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(item_hint=validate_item_names, rewritten_query=validated_query)
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        try:
            # 4. 获取AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        # filter = 'item_name in '"商品A", "商品B"'
        #  '"商品A", "商品B"'
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        # filter = 'item_name in ["商品A", "商品B", "商品C"]'v   # 标量字段（动态字段）进行过滤
        return f" item_name in [{quoted}]"


if __name__ == '__main__':

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]  # 对齐字段
    }

    vector_search = HyDeSearchNode()

    result = vector_search.process(state)

    for r in result.get('hyde_embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))
