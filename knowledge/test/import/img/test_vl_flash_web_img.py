from openai import OpenAI
import os

# 1. 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-26d57c968c364e7bb14f1fc350d4bff0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2.创建聊天完成请求（远程：底层都是发送请求 本地：走文件）
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "这道题怎么解答？"
                },
            ],
        },
    ],
)

answer = completion.choices[0].message.content
print(answer)
