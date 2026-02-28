import json
import re
from pathlib import Path
from typing import Tuple, List
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config


class DocumentSpliterNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 加载--->打撒（1. 嵌入模型语义更准确 2.注入元数据 3.多路召回 4.性能、成本低----->减少LLM的幻觉，提高检索质量）---->组合:LLM就想成人的脑子

        # 1. 获取参数
        md_content, file_title, max_content_length = self._get_inputs(state)

        # 2. 根据标题切割(核心)
        sections, has_title = self._split_by_headings(md_content, file_title)

        # return {"result": sections}

        # 2. 处理
        # 2.1 section内容过长 继续进行二次切割
        # 2.1 section内容过段 看能不能合并（如果不能合并不合并 反之则合并）

        # 3. 组装

        # 4. 更新state:chunks

        pass

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int]:
        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        return md_content, file_title, config.max_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> Tuple[List[dict], bool]:
        """
        根据MD的标题（1-6）级标题进行切分
        Args:
            md_content: md内容
            file_title: 文档名字

        Returns:
        Tuple[List[dict], bool]
            List[dict]:sections
            bool: md文档是否有标题

             {
            "title": "# 第一章",
            "body": "正文内容...",
            "file_title": "万用表",
            "parent_title": "# 第一章"  父标题会更新
            }


        """

        # 1. 定义变量
        in_fence = False
        has_title = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ""
        hierarchy = [""] * 7  # 7个长度 但是第一个（0）不用 =

        # 2. 定义正则表达式(group1:标题的语法符号#【最少1个# 最多6个#】)
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 切分
        content_lines = md_content.split("\n")

        def _flush():
            """
             封装section对象
            Returns:
            """

            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                return sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title
                })

        for content_line in content_lines:
            # 3.1 判断是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None  # None:假的值

            if match:
                has_title = True
                # 当前_flush()行是标题
                _flush()
                level = len(match.group(1))  # 当前标题的级别
                current_level = level  # 当前标题的级别
                current_title = content_line
                hierarchy[level] = current_title  # 当前标题的名字

                for i in range(level + 1, 7):  # 清空
                    hierarchy[i] = ""

                body_lines = []
            else:
                # 除了标题行以外全都收集起来
                body_lines.append(content_line)

        _flush()

        return sections, has_title


if __name__ == '__main__':
    document_node = DocumentSpliterNode()
    # 构造状态字典
    file_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\test\import\test_hierarchy.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,

    }
    print(document_node.process(state).get('result'))
