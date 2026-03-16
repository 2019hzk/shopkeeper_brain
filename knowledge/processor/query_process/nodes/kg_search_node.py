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
from knowledge.utils.neo4j_util import get_neo4j_driver

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
# Neo4J的信息
# -------------------------------------------------
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]

# Neo4j的Cypher语句
_CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN  n.item_name as item_name,n.name as name
LIMIT 1
"""

# toLower()小写
_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($entity_name)
      AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""

# Neo4j的常量
_MAX_FUZZY_SEEDS: int = 3


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


def _clean_seed_rows(rows:List[Dict[str,Any]])-> List[EntitySeedNode]:
    """
    职责：清洗查询种子节点的数据
    Args:
        rows:  查询到的结果记录

    Returns:
        干净的结果记录

    """

    if not rows:
        return []

    clean_seeds_result: List[EntitySeedNode] = []
    # 1. 遍历
    for row in rows:
        # 1.1 获取item_name
        item_name = row.get('item_name', '').strip()
        # 1.2 获取entity_name
        entity_name = row.get('name', '').strip()
        # 1.3 判断
        if not item_name or not entity_name:
            continue
        # 1.4 封装一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })
    # 2. 返回
    return clean_seeds_result


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

        fallback_result = {"entities_aligned_name": [], "entities_aligned_elements": []}
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
            align_one_result: List[Dict[str, Any]] = self._align_one(milvus_client,
                                                                     self._collection_name,
                                                                     item_name_filtered_expr,
                                                                     embedding_result_dense,
                                                                     embedding_result_sparse,
                                                                     index,
                                                                     entity_name)

            # 7.2 将商品对齐结果存储到最终结果中
            aligned_entity_elements.extend(align_one_result)

            # 7.3 遍历商品下的对齐结果
            for detail in align_one_result:
                # a) 获取对齐后的实体名
                aligned_name = detail.get("aligned")

                # b) 获取商品名
                item_name = detail.get("item_name")

                # c) 判断对齐名是否有
                if aligned_name:

                    # 去重 同名实体在不同商品下都保留
                    key = (item_name, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities_name.append(aligned_name)

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
                   entity_name: str) -> List[Dict[str, Any]]:

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
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}]

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
        hits = reps[0] if reps else []
        if not hits:
            return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]

        # 4.1  按 item_name 分组，每组取最高分
        best_by_item: Dict[str, Dict] = {}
        for hit in hits:
            # a) 获取实体
            entity = hit.get("entity")

            # b) 从实体中获取
            item_name = entity.get("item_name").strip()

            # c) 只保留每个 item_name 下的第一个（即最高分）
            if item_name not in best_by_item:
                best_by_item[item_name] = hit

        # 4.2 是否有最好的item_name
        if not best_by_item:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]

        # 4.3  item_name 分组输出结果，过滤低于阈值的
        results: List[Dict[str, Any]] = []
        for item_name, best in best_by_item.items():
            # a) 获取最好的那一个分数
            score = best.get("distance")

            # b) 判断分数值
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN):
                continue

            # c) 获取实体信息
            ent = best.get("entity")

            # d) 将不同商品下最好的实体名添加到结果集中
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "item_name": item_name,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_item",
            })

        # 4.4 全部低于阈值时返回未命中
        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results


class _Neo4jGraphReader:
    """
    职责：所有对Neo4j的读操作
    1. 种子节点的查询（1.1 精确查询 1.2 降级走模糊查询兜底 ）
    2. 查询种子节点一跳关系（双向：种子节点指向另外的节点，以及另外的节点指向种子节点） 保留完整的关系
    3. 根据所有的节点（种子节点以及邻居节点）方向查询chunk(item_name，id)
    4. 根据所有的chunk_id 查询milvus得到所有的chunk

    """

    def __init__(self, database: str):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        # 1.获取驱动
        neo4j_driver = get_neo4j_driver()

        # 2. 判断驱动是否存在
        if neo4j_driver is None:
            raise RuntimeError("Neo4J驱动获取失败")

        # 3. 返回session对象
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        职责：根据item_name 以及entity_name 查询种子节点
        策略：精确查询，只返回一条 模糊查询，返回三条
        Args:
            pairs:  _build_item_entity_pairs方法返回的商品名和实体名的pair对

        Returns:
            所有商品名下所有实体名对应的种子节点
        """
        # 1. pair对是否存在
        if not pairs:
            return []

        final_seeds_result: List[EntitySeedNode] = []
        # 2. 遍历所有pair对
        for pair in pairs:
            # 2.1 获取item_name
            item_name = pair.get('item_name', '').strip()
            # 2.2 获取entity_name
            entity_name = pair.get('entity_name', '').strip()
            # 2.3 过滤掉无效
            if not item_name or not entity_name:
                continue
            # 2.4 执行cypher语句（1) 精确查询 2）可能要模糊查询）
            try:

                with self._session() as session:

                    # 1.精确查询
                    exact_rows = session.execute_read(
                        lambda tx: tx.run(
                            _CYPHER_EXACT_SEEDS, item_name=item_name, name=entity_name
                        ).data()
                    )
                    if exact_rows:
                        final_seeds_result.extend(_clean_seed_rows(exact_rows))
                        continue
                    # 2. 模糊查询
                    fuzzy_rows = session.execute_read(
                        lambda tx: tx.run(
                            _CYPHER_FUZZY_SEEDS, item_name=item_name, name=entity_name, limit=_MAX_FUZZY_SEEDS
                        ).data()
                    )
                    final_seeds_result.extend(_clean_seed_rows(fuzzy_rows))

            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")
                return []

        self._logger.info(f"获取种子节点 {len(final_seeds_result)} 个")
        return final_seeds_result


def _build_item_entity_pairs(aligned_entities_info: List[Dict[str, Any]]) -> List[ItemEntityPair]:
    """
    职责：从对齐后的实体详情中获取商品名+实体名的pair对
    去重：本质同一个商品名下的实体名只留一个, 不同商品名下的实体名都留
    Args:
        aligned_entities_info: 对齐后的实体详情列表

    Returns:
        商品+实体名的pairs

    """
    # 1. 判断对齐后的实体详情是否存在
    if not aligned_entities_info:
        return []

    seen = set()
    item_entity_pairs = []

    # 2. 遍历对齐后的实体详情
    for aligned_entity_info in aligned_entities_info:
        # 2.1 获取商品名item_name
        item_name = aligned_entity_info.get('item_name', "").strip()
        # 2.2 获取对齐后的实体名
        aligned_entity_name = aligned_entity_info.get('aligned', "").strip()
        # 2.3 商品名&实体名都存在
        if not (item_name and aligned_entity_name):
            continue
        # 2.4 去重
        key = (item_name, aligned_entity_name)
        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": aligned_entity_name
            })
    # 3. 返回
    return item_entity_pairs


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
        pattern = "|".join(re.escape(name) for name in item_names)
        user_query = re.sub(pattern, "", rewritten_query).strip()
        # 4. 返回
        return user_query, item_names

    def _run_pipeline(self, validated_query: str, validated_item_names: List[str]):

        # 1. 初始化组件
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name=self.config.entity_name_collection)
        neo4g_graph_reader = _Neo4jGraphReader(database=self.config.neo4j_database)

        # 2. 利用提取器提取实体(核心的实体名字留下，就可以通过该实体节点找和该节点有关系的节点)
        entities_name = entity_extractor.extract(user_query=validated_query)
        entities_name_aligned: Dict[str, Any] = entity_aligner.align(entities_name, item_names=validated_item_names)
        # 2.1 获取所有对齐后的实体名(业务逻辑不使用)
        aligned_entities_name = entities_name_aligned.get('entities_aligned_name')
        # 2.2 获取所有对齐后的实体详情（结构信息细粒）
        aligned_entities_info = entities_name_aligned.get('entities_aligned_elements')

        # 3. 构建商品名+实体名的pair对
        item_entity_pairs: List[ItemEntityPair] = _build_item_entity_pairs(aligned_entities_info)

        # 4. 根商品名和实体名的pairs 查询种子节点
        seed_nodes: List[EntitySeedNode] = neo4g_graph_reader.find_seed_nodes(item_entity_pairs)

        # 5.测试种子节点
        return seed_nodes


if __name__ == '__main__':
    kg_search_node = KnowledgeGraphSearchNode()
    state = {
        # "rewritten_query": "RS-12数字万用表如何测量直流电压",
        # "rewritten_query": "RS-12数字万用表如何打开背光灯键",
        # "rewritten_query": "RS-12数字万用表更换电池需要注意什么",
        "rewritten_query": "RS-12数字万用表更换电池需要注意什么",
        # "rewritten_query": "在RS-12 数字万用表中二极管的操作步骤是什么",
        # "rewritten_query": "RS-12数字万",
        # "item_names": ["RS-12数字万用表"]
        "item_names": ["RS-12 数字万用表"]
    }
    result = kg_search_node.process(state)

    print(result)
