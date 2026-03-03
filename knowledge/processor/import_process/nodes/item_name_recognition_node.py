from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain.embeddings import init_embeddings  # 1.x(用的时候小心)

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from knowledge.utils.llm_client import get_llm_client
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


        # 5. 存储到Milvus数据库

        # 5. 返回

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

            # 4. 获取模型的输出内容
            item_name = getattr(llm_response, 'content', '').strip()

            # 5. 判断
            if not item_name or item_name.upper() == 'UNKNOWN':
                self.logger.warning(f"LLM无法提取有效的商品名,安全回退到标题名：{file_title}")
                return file_title

            # 6. 真正返回提取到的商品名
            self.logger.info(f"提取到的商品名:{item_name}")
            return item_name

        except Exception as e:
            self.logger.error(f"LLM调用失败,商品名安全回退到标题名：{file_title}")
            return file_title
