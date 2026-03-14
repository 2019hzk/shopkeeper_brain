"""
node_query_kg — 知识图谱查询节点。

类结构（与导入侧 kg_graph_node.py 的 Writer 模式对称）:
─────────────────────────────────────────────────────────
  _EntityExtractor    LLM 实体抽取 + JSON 解析
  _EntityAligner      Milvus ENTITY_NAME_COLLECTION 实体对齐
  _Neo4jGraphReader   Neo4j 种子节点 / 一跳关系 / chunk 反查
  _ChunkBackfiller    Milvus CHUNKS_COLLECTION chunk 回填
  KGQueryNode         主编排器（组装上述四个组件，执行 pipeline）
─────────────────────────────────────────────────────────
  node_query_kg()     LangGraph 节点入口函数（薄包装）
"""
import logging, re, json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple
from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT

# -------------------------------------------------
# 常量
# -------------------------------------------------
# _ALLOWED_ENTITY_LABELS_CN: str = (
#     "设备(Device)、部件(Part)、操作(Operation)、步骤(Step)、"
#     "警告(Warning)、条件(Condition)、工具(Tool)"
# )

_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN = 0.5


# -------------------------------------------------
# 工具函数 （服务各个组件、不会污染组件）
# -------------------------------------------------

def _clean_parse_llm_content(llm_response_content: str) -> List[str]:
    """
     职责：清洗以及解析LLM输出
    Args:
        llm_response_content:

    Returns:
        List[str]:清洗后的实体名

    """
    # 1. 判断LLM输出内容是否为空
    if not llm_response_content:
        return []

    # 2. 清洗json代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", llm_response_content)
    re_sub = re.sub(r"\s*```$", "", text)

    # 3. 反序列解析
    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except JSONDecodeError as e:
        logging.error(f"JSON 反序列失败，原因: {str(e)}")
        return []

    # 4. 获取提取的实体名
    entities_name = deserialized_result.get('entities', [])

    # 4.1 提取实体名的校验（为空）
    if not entities_name:
        return []
    # 4.2 提取实体名的校验（有效类型）
    if not isinstance(entities_name, list):
        return []

    # 4.3 遍历所有实体名
    seen = set()  # 集合
    entities_name_result = []

    for entity_name in entities_name:
        # 1. 判断是否为空
        if not entities_name:
            continue
        # 2. 判断是否有效类型
        if not isinstance(entity_name, str):
            continue

        # 3. 实体名是否过长
        truncated_entity_name = truncate_entity_name_length(entity_name)

        # 4. 去重保序【顺序：防御性】
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)

    return entities_name_result


def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name


def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"


class _EntityExtractor:
    """
    实体提取器：
    责任： 利用LLM从查询问题中提取实体
    prompt:设计
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self, user_query: str) -> List[str]:
        """
         根据用户问题提取当前问题下的实体名
        Args:
            user_query:  用户问题

        Returns:
            List[str]: 提取后的实体名

        """

        # 1. 获取llm客户端
        llm_client = get_llm_client(response_format=True)

        # 2. 判断
        if llm_client is None:
            return []

        # 3. 获取提示词
        # 3.1 系统提示词
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH)

        # 4. 调用LLM
        try:
            # 4.1 发送请求
            llm_response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f"用户问题:{user_query}")
            ])

            # 4.2 获取模型的结果
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 4.3 清洗&解析
            entities_name = _clean_parse_llm_content(llm_response_content)
            return entities_name
        except Exception as e:
            self._logger.error(f"LLM 调用失败:{str(e)}")
            return []


class _EntityAligner:
    """
     实体对齐器：
     责任： 根据LLM提取到的实体名 查询Milvus，获取真正的实体名（对齐后的实体名、能够查询neo4j(查询节点使用)）
    """

    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        """

        Args:
            entity_names:  LLM提取的实体名
            item_names: 商品名

        Returns:
         Dict[str,Any]:该字典准备封装两个key.
         第一个key:entities_aligned:[] 所有对齐后的实体名
         第二key:entity_elements[]: 所有对齐后的实体信息[source_id ,distance,origin,aligned,content]

        """

        fallback_result = {"entities_aligned": [], "entity_elements": []}
        # 1. 判断实体名是否有
        if not entity_names:
            return fallback_result

        # 2. 获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            self._logger.error("嵌入模型不存在")
            return fallback_result

        # 3. 获取milvus客户端
        milvus_client = get_milvus_client()
        if milvus_client is None:
            self._logger.error("Milvus客户端不存在")
            return fallback_result

        # 4. 向量化实体名
        embedding_result = generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=entity_names)

        # 5. 检验嵌入结果
        if embedding_result is None:
            self._logger.error("嵌入结果无法获取")
            return fallback_result
        # 5.1 获取嵌入后的稠密向量对象（二维数组）
        embedding_result_dense = embedding_result['dense']
        # 5.2 获取嵌入后的稀疏向量对象（二维数组）
        embedding_result_sparse = embedding_result['sparse']

        # 6. 获取item_name的表达式
        item_name_filtered_expr = _item_name_filter_expr(item_names)

        # 7. 遍历所有的实体名字
        aligned_entities_name: List[str] = []  # 存放所有实体的名字
        aligned_entity_elements: List[Dict[str, Any]] = []  # 存放所有实体的详细信息
        seen = set()

        for index, entity_name in enumerate(entity_names):
            # 7.1 对齐一个实体的
            align_one_result: Dict[str, Any] = self._align_one(milvus_client,
                                                               self._collection_name,
                                                               item_name_filtered_expr,
                                                               embedding_result_dense,
                                                               embedding_result_sparse,
                                                               index,
                                                               entity_name)
            # 7.2 构建实体名字
            aligned_entity_name = align_one_result.get('aligned')
            # 防御
            if aligned_entity_name not in seen:
                seen.add(aligned_entity_name)
                aligned_entities_name.append(aligned_entity_name)

            # 7.3 构建对齐后的实体详细信息
            aligned_entity_elements.append(align_one_result)

        self._logger.info(f"对齐后的实体个数 {len(aligned_entities_name)} 实体的名字：{aligned_entities_name}")
        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }

    def _align_one(self, milvus_client: MilvusClient,
                   _collection_name: str,
                   item_name_filtered_expr: str,
                   embedding_result_dense: List,
                   embedding_result_sparse: List,
                   index: int,
                   entity_name: str):

        """
        对齐指定实体名
        Args:
            milvus_client:
            _collection_name:
            item_name_filtered_expr:
            embedding_result_dense:
            embedding_result_sparse:
            index:

        Returns:

        """
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]

        # 1. 判断实体名的稠密和稀释向量
        if not dense_vector or not sparse_vector:
            return {"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}

        # 2. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=dense_vector,
                                                               sparse_vector=sparse_vector,
                                                               expr=item_name_filtered_expr, limit=5)
        # 3. 执行混合搜索请求
        reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                           collection_name=_collection_name,
                                           search_requests=hybrid_search_requests,
                                           ranker_weights=(0.4, 0.6),
                                           norm_score=True,
                                           limit=5,
                                           output_fields=["source_chunk_id", "item_name", "context", "entity_name"],
                                           )

        # 4. 解析结果
        if not reps or not reps[0]:
            return {"original": entity_name, "aligned": "", "context": "", "reason": "search result  is Empty "}

        # 5. 获取结果
        best_entity = self._pick_best_entity_name(reps[0])

        # 6. 返回数据结构
        return {
            "original": entity_name,
            "aligned": best_entity['entity_name'],
            "source_chunk_id": best_entity['source_chunk_id'],
            "item_name": best_entity['item_name'],
            "context": best_entity['context'],
            "reason": "top1"
        }

    def _pick_best_entity_name(self, search_entities_name_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
         职责： 从返回的5个实体名字中留下一个实体名
        Args:
            param:

        Returns:

        """
        # 1. 判断是否检索到了
        if not search_entities_name_result:
            return None

        # 2. 获取第一个
        first_entity = search_entities_name_result[0]
        if not first_entity:
            return None

        # 3. 获取第一个实体名的分数值
        first_entity_name_score = first_entity.get('distance')

        # 4. 判断是否超过阈值
        if not first_entity_name_score:
            return None
        # 5. 返回的实体名是第一个且分数超过阈值的（对齐策略）
        return first_entity if first_entity_name_score > _DEFAULT_ENTITY_NAME_ALIGN else None


class KnowledgeGraphSearchNode(BaseNode):
    """
      知识图谱查询主编排器。

      职责：
      - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
      - 按 pipeline 顺序编排调用

      Pipeline:
      ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
         抽取实体  ──▶   对齐实体   ──▶    Neo4j查询   ──▶ 回填chunk
      └──────────┘   └──────────┘   └────────────┘   └──────────┘
      """

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_inputs(state)

        # 2. 执行流水线
        result = self._run_pipeline(validated_query, validated_item_names)

        return result

    def _validate_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 从重写的问题中踢掉商品名(降噪以及无异议的查询)选择
        user_query = None
        for item_name in item_names:
            user_query = rewritten_query.replace(item_name, '') #
        # 4. 返回
        return user_query, item_names

    def _run_pipeline(self, validated_query: str, validated_item_names: List[str]):

        # 1. 初始化组件
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name=self.config.entity_name_collection)

        # 2. 利用提取器提取实体(核心的实体名字留下，就可以通过该实体节点找和该节点有关系的节点)
        entities_name = entity_extractor.extract(user_query=validated_query)
        entities_name_aligned: Dict[str, Any] = entity_aligner.align(entities_name, item_names=validated_item_names)

        return entities_name_aligned

if __name__ == '__main__':
    kg_search_node = KnowledgeGraphSearchNode()
    state = {
        # "rewritten_query": "RS-12数字万用表如何测量直流电压",
        # "rewritten_query": "RS-12数字万用表如何打开背光灯键",
        # "rewritten_query": "RS-12数字万用表更换电池需要注意什么",
        "rewritten_query": "RS-12 数字万用表更换电池需要注意什么",
        # "rewritten_query": "在RS-12 数字万用表中二极管的操作步骤是什么",
        # "rewritten_query": "RS-12数字万",
        # "item_names": ["RS-12数字万用表"]
        "item_names": ["RS-12 数字万用表"]
    }
    result = kg_search_node.process(state)

    print(result)
