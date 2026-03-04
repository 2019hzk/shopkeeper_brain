import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from typing import Sequence, List, Any, Dict, Optional
from dataclasses import dataclass
from pymilvus import DataType, MilvusClient
from pymilvus.orm.schema import CollectionSchema
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.milvus_util import get_milvus_client

"""

门面+建造者设计模式
门面角色：ImportMilvusNode节点的process（1.数据校验 2.insert() 3.更新sate(返回)）

设计3个类都是和Milvus操作相关

类1：MilvusSchemaBuilder:专门负责对Milvus的约束操作
类2：MilvusIndexBuilder:专门负责对Milvus的索引操作
类3：MilvusInserter:专门负责对Milvus做插入操作

Milvus的约束：
# 1.主键字段约束：唯一性最强
# 2.向量字段约束：唯一性还行
# 3.标量字段约束：灵活【title,file_tile,url,author,page_number】


类4：专门负责管理Milvus标量字段 （对大多数标量的共性字段做提取复用）
"""


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# Sequence:表明修饰的对象是一个有序可读的序列
# enable_dynamic_field=True:表示的是schema中没有定义的约束，但是插入数据的时候，有不在schema中没有定义的字段。允许插入进去。 切记：不是我定义的，你插入数据的时候可以不传。
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    # ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder:
    """
     职责：专门负责构建约束
    """

    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:

        logger.info("开始构建约束(schema)...")
        # 1. 构建约束对象(动态映射)
        schema = client.create_schema(enable_dynamic_field=True)

        # 2. 构建主键字段约束
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,  # INT类型
            is_primary=True,
            auto_id=True
        )

        # 3. 构建量字段约束
        # 3.1 稠密向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        # 3.2 稀疏向量字段
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 4. 构建标量字段约束
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {"field_name": scalar_field.field_name, "datatype": scalar_field.datatype}

            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length

            schema.add_field(**kwargs)

        logger.info(f"构建约束(schema)完成...")
        # 5. 返回
        return schema


class _MilvusIndexBuilder:
    """
    职责：负责处理Milvus的索引
    """

    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")
        # 1. 创建索引对象
        index = client.prepare_index_params(collection_name=collection_name)

        # 2. 给向量字段添加索引
        # 2.1 稠密向量字段添加索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 2.2 稀疏向量字段添加索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        logger.info(f"构建集合 {collection_name} 索引完成...")
        # 3. 返回index
        return index


class _MilvusInserter:
    """
    职责：将数据插入到Milvus 以及 回填chunk_id
    """

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[dict[str, Any]]:
        logger.info(f"开始插入{len(chunks)}块到Milvus...")
        # 1. 插入
        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        inserted_count = inserted_result.get('insert_count')

        ids = inserted_result.get('ids')

        # 2. 回填id
        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}记录,并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)

        # 2. 获取milvus客户端
        milvus_client = get_milvus_client()

        # 3. 判断milvus客户端
        if milvus_client is None:
            return state

        # 4. 获取集合名字
        # collection = getattr(config, 'chunks_collection', 'test_chunks_collection')
        collection = "test_chunk"

        # 5.确保集合存在（判断集合是否有、没有 创建新的【schema index】）
        self._ensure_has_collection(milvus_client, collection, dim)

        # 6. 插入
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)

        final_chunks = inserter.insert(chunks=validated_chunks)

        # 7. 更新state
        state['chunks'] = final_chunks

        return state

    def _validate_get_inputs(self, state: ImportGraphState):
        self.log_step("step1", "参数校验")

        config = get_config()

        # 1. 获取chunks
        chunks = state.get('chunks')

        # 2. 校验是否为空
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        # 3. 校验是否有混合向量
        validated_chunks = []
        for chunk in chunks:

            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在")

        # 4. 判断有效集合
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        # 5. 获取向量维度
        dim = len(validated_chunks[0].get('dense_vector'))
        self.logger.info(f"导入Milvus向量数据库的有效块：{len(validated_chunks)},且chunk的向量维度{dim}")

        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client: MilvusClient, collection_name: str, dim: int,
                               delete_flag: bool = True):

        self.log_step("step2", f"准备集合 {collection_name} 创建")
        # 1. 判断是否要删除集合
        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"Milvus中的集合 {collection_name}已被删除")
            milvus_client.drop_collection(collection_name=collection_name)

        # 2. 判断集合是否有
        if milvus_client.has_collection(collection_name=collection_name):
            self.logger(f"{collection_name}集合已经存在")
            return

        # 3. 创建约束
        schema = _MilvusSchemaBuilder.build(milvus_client, dim)

        # 4. 创建索引
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        # 5. 创建集合
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)


from pathlib import Path
import json


def _cli_main() -> None:
    setup_logging()

    temp_dir = Path(
        r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用\hybrid_auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

    state: ImportGraphState = {
        "chunks": content.get("chunks", [])
    }

    import_milvus = ImportMilvusNode()
    result_state = import_milvus.process(state)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_state, fh, ensure_ascii=False, indent=4)

    logger.info(f"备份临时文件{output_path}成功")


if __name__ == "__main__":
    _cli_main()
