from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from knowledge.processor.import_process.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, config = self._validate_inputs(state)

        # 2. 构建LLM的上下文（提取商品名）
        item_name_context = self._prepare_item_name_context(chunks, config)

        # 3.调用LLM模型
        item_name = self._recognition_item_name_by_llm(file_title, item_name_context)

        # 4. 嵌入商品名(嵌入模型：OpenAI:OpenAIEmbeddings dashscope:text_embedding_v1(2/3/4)--->稠密向量：主要作用（语义相似性）、稀疏向量主要作用（关键词相似）【bgem3】)
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到Milvus数据库
        self._save_to_milvus(file_title, item_name, dense_vector, sparse_vector, config)

        # 6. 回填item_name信息【sate/chunk对象】
        self._fill_item_name(item_name, state, chunks)

        return state

    def _fill_item_name(self, item_name: str, state: ImportGraphState, chunks: List[Dict[str, Any]]):

        self.log_step("step6", "回填商品名信息")
        for chunk in chunks:
            chunk['item_name'] = item_name  # 方便下游模型能有参考

        state['item_name'] = item_name  # 程序员使用的时候更加方便

    def _embedding_item_name(self, item_name: str) -> Optional[Tuple[list, dict[Any, Any]]]:

        self.log_step("step4", "embedding模型嵌入商品名")
        try:
            # 1. 获取嵌入模型
            embedding_model = get_beg_m3_embedding_model()

            # 2. 嵌入item_name
            embedding_result = embedding_model.encode_documents([item_name])

            # 3. 获取稠密和稀疏向量
            dense = embedding_result['dense'][0].tolist()
            start_index = embedding_result['sparse'].indptr[0]
            end_index = embedding_result['sparse'].indptr[1]
            weights = embedding_result['sparse'].data[start_index:end_index].tolist()
            tokenIds = embedding_result['sparse'].indices[start_index:end_index].tolist()
            sparse = dict(zip(tokenIds, weights))
            return dense, sparse
        except Exception as e:
            self.log_step(f"嵌入商品名:{item_name}失败,原因是：{str(e)}")
            raise EmbeddingError(f"嵌入商品名:{item_name}失败,原因是：{str(e)}", self.name)

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step1", "检验输入参数")
        config = get_config()

        # 1. 获取state的file_title以及 chunks
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        # 2. 判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)

        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunk为空或者无效", self.name)

        item_name_chunk_k = config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("item_name_chunk_k为空或者无效", self.name)

        self.logger.info(f"检测到文件：{file_title},对应的切片长度:{len(chunks)}")
        # 3. 返回
        return file_title, chunks, config

    def _prepare_item_name_context(self, chunks: Optional[List[Dict[str, Any]]], config):

        self.log_step("step2", "构建商品名提取的上下文")
        result = []
        # 我要从前5块中留下内容的字符数不能超过2000个字符长度
        total = 0
        for index, chunk in enumerate(chunks[:config.item_name_chunk_k]):

            # 1. 判断chunk的类型
            if not isinstance(chunk, dict):
                continue
            ## 构建上下文：【切片-1】标题+ body组成( content:标题+\n\n +body)
            # 2. 提取
            content = chunk.get('content')
            spices = f"【切片】- {index + 1} - {content}"

            # 3. 计算长度
            total += len(spices)

            result.append(spices)

            # 4. 判断是收集到的长度否超过阈值设定
            if total > config.item_name_chunk_size:
                break

        return "\n\n".join(result)[:config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title: str, item_name_context: str) -> str:

        self.log_step("step3", "LLM识别商品名")
        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error(f"LLM初始化失败,商品名安全回退到标题名：{file_title}")
            return file_title

        # 2. 构建LLM提示词(格式化用户提示词模版)
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        # 3. 调用模型(# str [] # PromptValue/XXXMessage的content不能放带变量的字符串)
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])

            # 3.1 获取模型的输出内容
            item_name = getattr(llm_response, 'content', '').strip()

            # 3.2 判断
            if not item_name or item_name.upper() == 'UNKNOWN':
                self.logger.warning(f"LLM无法提取有效的商品名,安全回退到标题名：{file_title}")
                return file_title

            # 3.3 真正返回提取到的商品名
            self.logger.info(f"提取到的商品名:{item_name}")
            return item_name

        # 4. 降级（安全处理）
        except Exception as e:
            self.logger.error(f"LLM调用失败,商品名安全回退到标题名：{file_title}")
            return file_title

    def _save_to_milvus(self, file_title, item_name, dense_vector, sparse_vector, config):

        self.log_step("step5", "保存到向量数据库中")
        # 1. 参数检验
        if not dense_vector or not sparse_vector:
            self.logger.warning(f"[{item_name}] 向量生成不完整，跳过入库！")
            return

        # 2. 操作MilVus
        try:
            # 2.1 获取Milvus的客户端
            milvus_client = get_milvus_client()

            # 判断
            if milvus_client is None:
                return

            # 2.2 获取集合的名字
            collection_name = config.item_name_collection

            # 2.3 幂等性校验（不存在则创建新的）
            if not milvus_client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(milvus_client, collection_name)

            # 2.4 构建字典结构数据
            data = {
                "file_title": file_title,  # 文件名字
                "item_name": item_name,  # 商品名字
                "dense_vector": dense_vector,  # 稠密向量 （list）
                "sparse_vector": sparse_vector  # 稀疏向量  (dict:{tokenId:weight})
            }

            # 2.5 插入数据到Milvus:{"insert_count":10,"ids":[10001,10002,10003]}
            result = milvus_client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")

        except Exception as e:
            self.logger.error(f"Milvus 数据库保存操作彻底失败: {e}")

    def _create_item_name_collection(self, client, collection_name):
        self.logger.info(f"正在创建集合: {collection_name}")

        # 1. 创建约束
        schema = client.create_schema()
        # 1.1 主键字段的约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)

        # 1.2 标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 向量字段的约束
        # 1.3.1 稠密向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 1.3.2 稀疏向量字段约束
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        # 2.1 创建稠密向量字段索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # 2.2 创建稀疏向量字段索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")


# ---------------------------------
# 测试
"""
    json.load()
    json.loads()
    
    json.dump()
    json.dumps()
    
      json.load(文件对象) and  json.dump()
      
      json.loads() and  json.dumps()
      
"""
# ---------------------------------
import  json
if __name__ == '__main__':
    # 1. 读取chunk.json
    chunk_json_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用\hybrid_auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)
    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    item_name_recognition_node = ItemNameRecognitionNode()

    # 4. 调用process
    result=item_name_recognition_node.process(state)

    # 5. 输出
    print(json.dumps(result,ensure_ascii=False, indent=4))

    """
      state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }
    
      state = {
        "file_title": "万用表的使用",
        "chunks": [{"item_name"},{"item_name"}],
        "item_name"
    }
    """



