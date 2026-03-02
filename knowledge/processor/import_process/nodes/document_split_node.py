import json
import re
from email.quoprimime import body_length
from pathlib import Path
from typing import Tuple, List, Dict, Any, final
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSpliterNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 加载--->打撒（1. 嵌入模型语义更准确 2.注入元数据 3.多路召回 4.性能、成本低----->减少LLM的幻觉，提高检索质量）---->组合:LLM就想成人的脑子

        # 1. 获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        # 2. 根据标题切割(核心)
        sections = self._split_by_headings(md_content, file_title)

        # 3. 处理(切分和合并)
        fina_chunks = self.split_and_merge(sections, max_content_length, min_content_length)

        # 3. 组装

        # 4. 更新state:chunks

        pass

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int]:

        self.log_step("step1", "切分文档的参数校验以及获取")

        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
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
        self.log_step("step2", "根据标题进行切割")
        # 1. 定义变量
        in_fence = False
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

        return sections

    def split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int):
        """

        Args:
            sections: 根据一级标题切分后的所有section（章节）块
            max_content_length: 每一个section的content内容【title+body】长度最多不能超过指定 :将标题注入到内容中（标题注入：明确定位这一块的归属）
            min_content_length: 每一个section的content内容 长度如果比min_content_length小，尝试进行合并（合并：同源）

        Returns:
            List[section]

        """
        self.log_step("step2", "切分长内容以及合并段内容")

        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)

        # 3. 返回
        return final_sections

    def split_long_section(self, section: Dict[str, Any], max_content_length: int):
        """
        只要满足条件的才会切（当前section的内容达到了max_content_length）
        Args:
            section:
            max_content_length:

        Returns:

        """

        self.log_step("step3", "进行长内容的切分")

        # 1. 获取section对象属性
        title = section.get('title')  # 不可能空
        body = section.get('body')  # 有可能是空
        file_title = section.get('file_title')  # 不可能是空
        parent_title = section.get('parent_title')  # 不可能是空

        # 2. 对标题做校验(感觉)
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:50]

        # 3. 拼接title前缀
        title_prefix = f"{title}\n\n"

        # 4. 计算总长度(len(title_prefix)+len(body))
        total_length = len(title_prefix) + len(body)

        # 5. 判断小于或者刚好满足阈值（section内容比较短）
        if total_length <= max_content_length:
            return [section]

        # 6.计算body可用的长度
        body_length = max_content_length - len(title_prefix)

        if body_length <= 0:
            return [section]

        # 7. 切分   # 7.1 对谁切【body】 # 7.2 用谁切[1)手写 2)langchain提供的切分器:TextSpliter/TokenSpliter/LenghtSpliter/）:chunkoverlap:块与块之间的重叠/:通用的递归切分器【"\n\n",'\n',' ',''[兜底：一个一个字符]】]
        # 7.1 定义递归的文档切分器对象
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";",
                                                                   " ", ""],
                                                       keep_separator=False
                                                       )
        # 7.2 进行切割
        texts = text_splitter.split_text(body)

        # 7.2 判断【长度：0 ：场景：body没有：长度：1:场景：标题下面的内容很少】
        if len(texts) <= 1:
            return [section]

        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"  # 标记序号
            })
        return sub_section

    def merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int):
        """
        贪心累加算法
        Args:
            current_sections:
            min_content_length:

        Returns:

        """
        current_section = current_sections[0]
        current_section_body = current_section.get('body')
        fina_sections = []  # 最终的箱子

        for next_section in current_sections[1:]:
            # 同源
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section_body) < min_content_length:
                # body的合并(更新当前的section_body)
                current_section['body'] = (
                        current_section.get('body').rstrip() + next_section.get('body').lstrip()
                )

                # TODO(part、标题)
            else:
                # 1. 将原来current_section进行封箱
                fina_sections.append(current_section)
                # 2. 更新next_section
                current_section = next_section

        # 最后一个人（封装起来）
        fina_sections.append(current_section)

        # 对所有section的part做处理




if __name__ == '__main__':
    document_node = DocumentSpliterNode()
    # 构造状态字典
    file_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\test\import\test_spilt.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,

    }
    document_node.process(state)
