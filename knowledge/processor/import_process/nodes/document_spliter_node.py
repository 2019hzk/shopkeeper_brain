import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError


class DocumentSpliterNode(BaseNode):
    def process(self, state: T) -> T:

        # 加载--->打撒（1. 嵌入模型语义更准确 2.注入元数据 3.多路召回 4.性能、成本低----->减少LLM的幻觉，提高检索质量）---->组合:LLM就想成人的脑子

        # 1. 根据标题切割

        # 2. 处理
        # 2.1 section内容过长 继续进行二次切割
        # 2.1 section内容过段 看能不能合并（如果不能合并不合并 反之则合并）

        # 3. 组装

        # 4. 更新state:chunks

        pass