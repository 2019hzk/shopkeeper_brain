import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple
from agents.mcp import MCPServerSse  # pip install openai_agents
from langchain_core.messages import SystemMessage, HumanMessage
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    name = "mcp_search_node"
    """
     负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
     mcp形式调用第三方的各种通用的搜索工具。
     百度：【电商】商品比价工具、商品搜索的工具、商品全维度对比工具、商品下单的工具 百度搜索工具 百度地图的工具..
     灵积服务平台:的通用搜索工具【bailian_web_search】
     mcp: 本质：就是各大平台把通用的功能，封装成了工具（函数） 然后通过mcp协议 客户端就可以直接调用它。【mcp客户端】---->【mcp服务端：任意选择某一个】
     
    """

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具  bailian_web_search
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 更新state web_search_docs
        state['web_search_docs'] = mcp_result

        # 4. 返回更新后的state
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

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """
        1. 创建mcp客户端
        2. 客户端执行对mcp服务端调用
        别人的接口：1. 发送请求调用别人
        使用客户端要调用服务端【1. 原生发送请求 2. mcp客户端发送请求 3. Agent【mcp客户端发送请求】】
        Agent:代理【代理对象就是程序员】
        Args:
            validated_rewritten_query:

        Returns:

        """

        # 1. 创建mcp客户端【1.host 2. http 2.1 streamable(主要建议选择使用) http 2.2 sse http 3.stdio sse】
        mcp_client = MCPServerSse(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,  # 服务端的端点
                "headers": {"Authorization": self.config.openai_api_key}  # 认证权限 api_key
            },
            cache_tools_list=True,  # mcp服务端工具列表的工具做缓存
        )
        try:
            # 2. 建立mcp连接
            await mcp_client.connect()
            BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            API_KEY = "sk-26d57c968c364e7bb14f1fc350d4bff0"
            MODEL_NAME = "qwen3-max"

            model_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
            # 3. 执行工具
            agent = Agent(
                name="Assistant",
                mcp_servers=[mcp_client],
                model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=model_client),
            )

            agent_result = await Runner.run(agent,validated_rewritten_query)

            # 4. 解析工具执行完的结果
            print(agent_result)

        finally:
            await  mcp_client.cleanup()  # 关闭连接


if __name__ == '__main__':
    state = {
        # "rewritten_query": "万用表如何测量电阻",
        "rewritten_query": "今天北京天气怎么样，并且告诉我今天具体的日期是什么时候",
        "item_names": ["RS-12 数字万用表"]  # 对齐
    }

    mcp_search = McpSearchNode()

    result = mcp_search.process(state)

    for r in result.get('web_search_docs'):
        print(json.dumps(r, ensure_ascii=False, indent=2))
