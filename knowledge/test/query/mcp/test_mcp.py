
from agents.mcp import MCPServerSse,MCPServerStreamableHttp,MCPServerStreamableHttpParams
import dotenv
dotenv.load_dotenv()
async def test_mcp():
    sse_mcp_client = MCPServerSse(
                name="阿里云百炼_联网搜索",
                params={
                    'url': "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
                    'headers': {
                        'Authorization': ""
                    },
                    'timeout': 300


                },
                cache_tools_list=True,

            )

    streamable_mcp_client = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(
            url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            headers={
                "Authorization": "",
            },
            timeout=10,
            sse_read_timeout=60 * 5,
        ),
        cache_tools_list=True,
        name="WebSearch",
    )



import asyncio
if __name__ == '__main__':
    asyncio.run(test_mcp())


