import json, time, re, logging
from concurrent.futures import ThreadPoolExecutor,as_completed
import  threading
from json import JSONDecodeError
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import Neo4jError, MilvusError
from knowledge.prompts.upload.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.neo4j_util import get_neo4j_driver
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model


# ------------------------------------------
# 常量
# ------------------------------------------
MAX_ENTITY_NAME_LENGTH = 15

# ------------------------------------------
# 白名单
# ------------------------------------------
# 实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}
# 关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPES = "RELATED_TO"


# ------------------------------------------
# Neo4J的Cypher语句
# ------------------------------------------
# Chunk标签节点创建
CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
"""

# Entity标签节点的创建
CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, item_name: $item_name}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""
# Entity关联Chunk
CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, item_name: $item_name})
    MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

# Entity与Entity的关系
CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, item_name: $item_name}})
    MATCH (t:Entity {{name: $tail, item_name: $item_name}})
    MERGE (h)-[:{rel_type}]->(t)
"""


# 清理Neo4J数据
CYPHER_CLEAR_ITEM = """
    MATCH (n {item_name: $item_name}) DETACH DELETE n
"""


@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


class _Neo4jGraphWriter:
    def __init__(self, database: str = ""):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def clear(self, neo4j_driver, item_name: str) -> None:
        if not neo4j_driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(neo4j_driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
            self._logger.info(f"Neo4j 旧数据已清理: {item_name}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")


    def insert(self, driver, entities, relations, chunk_id, item_name):
        """
        Neo4J的写入

        Args:
            driver: neo4j的驱动
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            item_name: 文档对应LLM提取的商品名

        Returns:

        """
        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")

        # 2.  判断驱动
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
            self._logger.info(f"Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系")
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")

    def _write_graph_tx(self, tx, entities, relations, chunk_id, item_name):

        # 1. 创建 Chunk 节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. 创建实体节点 + 关联到 Chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description")

            # 动态格式化 Cypher，将安全标签注入(TODO )
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)

            tx.run(cypher_query, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            # 关联实体到 Chunk
            tx.run(CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 创建实体间关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, item_name=item_name)



    def _session(self, driver):
        return driver.session(database=self._database)


class _MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus，仅供本模块内部使用。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)


    def  clear(self,milvus_client:MilvusClient,item_name:str):

        # 1. 清理 Milvus
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        collection_name = self.collection_name
        try:
            if milvus_client.has_collection(collection_name):
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'item_name == "{item_name}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: item_name={item_name}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")

    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, content: str, item_name: str) -> None:
        """对外唯一入口：将实体写入 Milvus。"""

        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        # 2. 获取去重后的实体名
        # entities_names = set({e["name"] for e in entities})
        entities_names = list(dict.fromkeys(e["name"] for e in entities if e.get("name")))
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        # 3. 获取嵌入模型
        bge_ef_model = get_beg_m3_embedding_model()

        if bge_ef_model is None:
            raise MilvusError("嵌入模型获取失败")

        # 4. 创建集合（不存在则创建）
        try:
            self._ensure_collection(milvus_client, self.collection_name)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}")

        # 5. 嵌入向量化
        try:
            embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}")

        # 6. 构建记录
        records = self._build_records(entities_names, embedded_result, chunk_id, content, item_name)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        # 7. 写入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.info(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}")

    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx].tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records


class KnowLedgeGraphNode(BaseNode):
    name = "knowledge_graph_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        super().__init__(config)
        self._milvus_writer = _MilvusEntityWriter(self.config.entity_name_collection)
        self._neo4j_writer = _Neo4jGraphWriter(self.config.neo4j_database)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, item_name = self._validate_get_inputs(state)

        # 2. 构建统计初始信息
        stats = ProcessingStats(total_chunks=len(validated_chunks))

        # 3. 获取
        # 3.1 获取milvus客户端
        milvus_client = get_milvus_client()
        neo4j_driver = get_neo4j_driver()

        # 4. 删除已经存在的数据（3.1 删除milvus中存储实体名字的记录（delete:item_name）：幂等性保证 3.2 删除neo4j的整个库下的所有节点以及关系）
        self._clean_exist_double_data(milvus_client, neo4j_driver, item_name)

        # 5. 批量处理（串行版本）
        # self._process_all_chunks_v1(stats, validated_chunks, milvus_client, neo4j_driver)
        # 5. 批量处理（多线程版本）
        self._process_chunks_concurrently(stats, validated_chunks, milvus_client, neo4j_driver)

        # 6. 简单的日志观察
        self.logger.info(stats.summary())

    def _clean_exist_double_data(self, milvus_client: MilvusClient, neo4j_driver,
                                 item_name: str):
        """
        删除milvus以及neo4j的对应文档的记录
        Args:
            milvus_client:
            neo4j_driver:
            item_name:

        Returns:

        """
        # 3.1 导入前清理该 item_name 下的所有旧数据（Milvus）
        self._milvus_writer.clear(milvus_client,item_name)

        # 3.1 导入前清理该 item_name 下的所有旧数据（Neo4J）
        self._neo4j_writer.clear(neo4j_driver,item_name)


    def _process_all_chunks_v1(self, stats: ProcessingStats,
                               validated_chunks: List[Dict[str, Any]],
                               milvus_client: MilvusClient,
                               neo4j_driver):
        """
        循环处理每一个chunk
        Args:
            validated_chunks:
            milvus_client:
            neo4j_driver:

        Returns:

        """

        # 1. 遍历所有的chunk
        for i, chunk in enumerate(validated_chunks):

            if not isinstance(chunk, dict):
                continue

            # 1.1 获取chunk的信息
            chunk_id = chunk.get('chunk_id')
            item_name = chunk.get('item_name')
            content = chunk.get('content')

            # 2. 处理单个chunk
            try:

                entities_count, relations_count = self._process_single_chunk(chunk_id,
                                                                             item_name,
                                                                             content,
                                                                             milvus_client,
                                                                             neo4j_driver)
                stats.processed_chunks += 1
                stats.total_entities += entities_count
                stats.total_relations += relations_count
                self.logger.info(f"成功处理完 {chunk_id} / {len(validated_chunks)}")
            except Exception as e:
                stats.failed_chunks += 1
                stats.errors.append(str(e))
                self.logger.error(f"处理失败 {chunk_id} / {len(validated_chunks)}")

    def _process_single_chunk(self, chunk_id: str,
                              item_name: str,
                              content: str,
                              milvus_client: MilvusClient,
                              neo4j_driver) -> Tuple[int, int]:

        llm_start = time.time()
        thread_name = threading.current_thread().name #  获取线程名
        # 1. 调用模型提取chunk的实体、关系
        llm_response = self._extract_graph_with_retry(content)
        llm_cost = time.time() - llm_start

        # 2. 解析并且清洗数据
        graph_result = self._parse_and_clean(llm_response)

        # 2.1 获取解析后的实体
        final_entities = graph_result.get('entities')
        # 2.2 获取解析后的关系
        final_relations = graph_result.get('relations')

        # 3. 写入
        # 3.1 将清洗后的实体名字（可能是多个）存储到milvus
        milvus_start = time.time()
        self._milvus_writer.insert(milvus_client, final_entities, chunk_id, content, item_name)
        milvus_cost = time.time() - milvus_start

        # 3.2 将清洗后的实体以及关系类型都存储到neo4j
        neo4j_start = time.time()
        self._neo4j_writer.insert(neo4j_driver, final_entities, final_relations, chunk_id, item_name)
        neo4j_cost = time.time() - neo4j_start

        total_cost = time.time() - llm_start
        # 4. 统计单块处理的时间信息
        self.logger.info(
            f"[{thread_name}] chunk={chunk_id} | "
            f"实体={len(final_entities)} 关系={len(final_relations)} | "
            f"LLM={llm_cost:.2f}s Milvus={milvus_cost:.2f}s Neo4j={neo4j_cost:.2f}s | "
            f"总计={total_cost:.2f}s"
        )

        return len(final_entities), len(final_relations)

    def _extract_graph_with_retry(self, content: str) -> str:

        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError(f"LLM客户端初始化失败")

        MAX_COUNT = 3
        last_error = None

        # 2.循环重试3次
        # TODO :将失败的异常原因给到模型
        for attempt in range(1, MAX_COUNT + 1):
            try:
                # 2.1 调用模型
                llm_response = llm_client.invoke([
                    SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
                    HumanMessage(content=f"切片信息\n\n{content}")
                ])
                # 2.2 获取内容
                result = getattr(llm_response, 'content', '').strip()

                # 2.3 有内容
                if result:
                    return result
            except Exception as e:
                last_error = e

                # 2.4 控制重试间隔
                if attempt < MAX_COUNT:
                    # 睡一会：间隔[固定间隔/指数退避]
                    delay = 0.5 * (2 ** (attempt - 1))
                    self.logger.warning(f"开始第{attempt}次重试，间隔：{delay:1.f}s")
                    time.sleep(delay)
        self.logger.error(f"已经进行了{MAX_COUNT}次重试，都失败原因：{str(last_error)}")

        # 3. 最终兜底
        return ""

    def _parse_and_clean(self, llm_response: str) -> Dict[str, Any]:
        """
        1.解析llm返回结果的json代码片段的围栏
        2.反序列化
        3.获取实体信息以及关系信息
        4.分别在清洗实体以及关系
        5. 清洗之后对应的实体和关系返回
        Args:
            llm_response: 模型的输出
        Returns:
             {
                "entities" :[{比较干净的实体名字:标签},{比较干净的实体名字:标签}]
                “relations” :[{比较干净的关系：“head”:"","tail":"","type":""},{比较干净的关系：“head”:"","tail":"","type":""}]
            }
        """

        # 1. 判断
        if not llm_response:
            raise ValueError(f"LLM提取chunk的图谱信息不存在")

        # 2. 清洗json代码块的围栏
        # 2.1 前面的7个非法字符踢掉```json
        # 2.2 后面的3个非法的字符踢掉```
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 3. 反序列化
        try:
            parsed_llm_response: Dict[str, Any] = json.loads(cleaned)
        except  JSONDecodeError as e:
            raise JSONDecodeError(f"反序列化失败 :{str(e)}")

        # 4. 获取信息
        # 4.1 获取实体信息
        entities = parsed_llm_response.get('entities', [])

        # 4.2 获取关系信息
        relations = parsed_llm_response.get('relations', [])

        # 5. 清洗实体
        cleaned_entities = self._clean_entities(entities)

        # 6. 获取清洗后的实体名
        cleaned_unique_entity_names = {entity.get('name') for entity in cleaned_entities}

        # 7. 清洗关系
        cleaned_relations = self._clean_relations(cleaned_unique_entity_names, relations)

        # 8. 构建返回字典
        return {"entities": cleaned_entities, "relations": cleaned_relations}

    def _clean_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. 清洗无效实体（实体名没有）
        2. 阶段过长的实体名（实体名太长）
        3. 实体的标签是否在白名单中
        4，去重（同名同标签的实体只能存在一份）
        5. 返回
        Args:
            entities: LLM中提取的实体信息

        Returns:
            合法干净的实体信息
        """

        unique_seen = set()
        clean_entities_result = []

        # 1. 遍历所有的实体信息
        for entity in entities:

            # 1.1 获取实体名
            entity_name = str(entity.get('name', '')).strip()

            # 1.2 校验名是否存在
            if not entity_name:
                continue

            # 1.3 截取实体名
            if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:15]

            # 1.4 获取实体标签
            entity_label = str(entity.get('label', '')).strip()

            # 1.5 判断标签是否在定义的实体标签是否白名单中
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            # 1.6 定义去重key
            unique_key = (entity_name, entity_label)

            # 1.7 判断是否是同一个实体（实体名+标签）
            if unique_key in unique_seen:
                continue
            unique_seen.add(unique_key)

            # 1.8 构建返回数据结构
            clean_entities = {"name": entity_name, "label": entity_label}

            # 1.9 判断实体的描述
            entity_describe = str(entity.get('description', '')).strip()
            if entity_describe:
                clean_entities['description'] = entity_describe

            # 1.10 将清洗后的实体信息存储到列表
            clean_entities_result.append(clean_entities)

        # 2. 返回最终清洗后的实体列表
        return clean_entities_result

    def _clean_relations(self, cleaned_unique_entity_names: Set[str], relations: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        清洗关系:
        1. 清洗关系的头尾节点是否不存在
        2. 截取头尾实体名过长
        3. 校验头尾实体名是否有效（悬空关系处理）
        4. 校验每一个关系的类型是否关系类型的白名单
        5. 返回

        Args:
            cleaned_unique_entity_names: 所有唯一的实体名集合
            relations:  LLM中提取的关系信息

        Returns:
            List[Dict[str,Any]] 合法干净的关系信息

        """

        clean_relations_result = []

        # 1. 遍历所有的关系
        for relation in relations:

            # 1.1 提取头（head）实体名
            head_entity_name = str(relation.get('head', '')).strip()

            # 1.2 提取尾 (tail) 实体名
            tail_entity_name = str(relation.get('tail', '')).strip()

            # 1.3 判断头尾实体是否有任意一个不存在
            if not head_entity_name or not tail_entity_name:
                continue

            # 1.4 判断头尾实体名是否超过阈值
            if len(head_entity_name) > MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) > MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            # 1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entity_names or tail_entity_name not in cleaned_unique_entity_names:
                continue

            # 1.6 获取关系类型
            relation_type = str(relation.get('type', '')).strip()

            # 1.7 判断关系类型是否在关系类型的白名单中
            if relation_type not in ALLOWED_RELATION_TYPES:
                # TODO 思路：反哺白名单
                relation_type = DEFAULT_RELATION_TYPES

            # 1.8 构建最终关系链的数据结构
            cleaned_relation = {"head": head_entity_name, "tail": tail_entity_name, "type": relation_type}

            # 1.9 将清洗后最终的关系链放到最终的结果中
            clean_relations_result.append(cleaned_relation)

        return clean_relations_result

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_item_name

    def _process_chunks_concurrently(self, stats:ProcessingStats, validated_chunks:List[Dict[str,Any]], milvus_client:MilvusClient, neo4j_driver):
        """
        多线程版本：
        多线程本质压榨CPU 和提高响应时间没有本质的关系
        Args:
            stats:
            validated_chunks:
            milvus_client:
            neo4j_driver:
        Returns:

        """

        with ThreadPoolExecutor(max_workers=4) as pool:
            # 1. 提交所有任务
            future_to_idx = {}
            for i, chunk in enumerate(validated_chunks):
                content = chunk.get("content")
                chunk_id = str(chunk.get("chunk_id"))
                item_name = chunk.get("item_name")

                # 像线程池中提交任务 返回任务对象
                future = pool.submit(
                    self._process_single_chunk,
                     chunk_id,item_name, content, milvus_client,neo4j_driver
                )
                future_to_idx[future] = (i, chunk_id)

            # 2. 收集结果（按完成顺序）（一定要让执行_process_chunks_concurrently方法的线程等所有任务做完）
            for future in as_completed(future_to_idx):
                idx, chunk_id = future_to_idx[future]
                try:

                    entity_count, relation_count = future.result()   # 任务的结果（_process_single_chunk 返回值）
                    stats.processed_chunks += 1
                    stats.total_entities += entity_count
                    stats.total_relations += relation_count
                except Exception as e:
                    stats.failed_chunks += 1
                    msg = f"切片 {chunk_id} 处理失败: {e}"
                    stats.errors.append(msg)
                    self.logger.error(msg)









def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM → 解析 → 清洗全流程。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mock_state = {
        "item_name": "测试万用表",
        "chunks": [
            {
                "content": """# 电池安装
                    警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
                    1. 把表笔与仪表断开。
                    2. 用螺丝刀拧开电池后盖上的螺母。
                    3. 正确安装电池，正负极应一致。
                    4. 盖上电池后盖并拧紧螺丝钉。
                    警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
                    注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "18438591111",
                "item_name": "测试万用表",
            }
        ],
    }

    knowledge_graph_node = KnowLedgeGraphNode()

    knowledge_graph_node.process(mock_state)


if __name__ == "__main__":
    test_kg_extraction()
