import logging
import time
import os, re
import base64
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque
from openai import OpenAI
from knowledge.utils.minio_util import get_minio_client
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.config import get_config


class MarkDownImageNode(BaseNode):
    """
    处理MarkDown图片节点类
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state: 上一个节点处理之后state的最新状态

        Returns:   当前节点处理之后的state最新状态(md_content=process_md)

        """
        # 1. 获取配置对象
        config = get_config()

        # 2. 处理文件路径(2.1 md内容 2.2 md的path 2.3 图片目录)
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            # 图片不用处理了，直接更新state的md_content
            self.logger.warning(f"文件{md_path_obj.name}暂无图片要处理")
            state['md_content'] = md_content
            return state

        # 3. 扫描并处理图片（最复杂）
        target_images_context = self._scan_images_and_context(image_dir, md_content, config)

        # 4. 用VLM给图片生成图片描述(摘要)
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)

        # 5.复合函数
        # 5.1 本地图片上传到minio--->remote_url(图片远程的地址)
        # 5.2 替换md中的图片的本地url 以及vlm生成的摘要
        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem, md_content, images_summaries,
                                                            target_images_context, config)

        # 6. 将更新后的内容备份（调试）
        self._backup_new_md_file(md_path_obj, new_md_content)

        # 7. 更新state
        state['md_content'] = new_md_content

        # 8. 返回更新后的状态
        return state

    def _backup_new_md_file(
            self,
            md_path_obj: Path,
            new_md_content: str
    ) -> str:
        self.log_step("step_5", "备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path)

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:  上一个节点处理之后state的最新状态

        Returns:
            md_content: md的内容
            md_path_obj: md的路径
            image_dir: 图片目录
        """
        self.log_step("step1", "读取md内容以及构建图片目录")
        # 1. 获取md_path
        md_path = state.get('md_path', '')

        # 2. 判断路径是否有内容
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        # 3. 标准化处理
        md_path_obj = Path(md_path)

        # 4. 判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        # 5. 读取md内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()  # 全部读取

        # 6.构建图片目录
        image_dir = md_path_obj.parent / "images"

        # 7. 返回
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content: str, config) -> List[
        Tuple[str, str, Tuple[str, str, str]]]:

        """
        扫描图片并且 找到图片的上下文
        Args:
            image_dir: 图片目录
            md_content: md内容
            config：配置信息
        Returns:
          List[Tuple[str,str,Tuple[str,str,str]]]
          List[("图片名字",“图片的地址”,("离图片最近的上面一个标题","图片的上文","图片的下文"))]
        """

        self.log_step("step2", f"扫描图片文件目录{image_dir}")
        target_images_context = []

        # 1. 遍历图片文件目录
        for img_name in os.listdir(image_dir):
            # 1.1 获取文件的后缀
            file_ext = os.path.splitext(img_name)[1]  # a.txt

            # 1.2 判断后缀是否有效
            if file_ext not in config.image_extensions:
                continue  # 继续处理下一个图片文件

            # 1.3 构建image_path 转成字符串
            img_path = str(image_dir / img_name)

            # 1.4 构建图片（上下文）
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)

            # 1.5 如果该图片没有上下文
            if not img_context:
                self.logger.warning("MD文件中暂未提取到可用的图片")
                continue  # 继续处理下一个图片文件

            # 1.6 提取到当前图片的唯一上下文内容(方便使用获取了第一个充当)
            primary_img_context = img_context[0]

            # 1.7 图片的完整信息构建到列表
            target_images_context.append((img_name, img_path, primary_img_context))

        # 1.8 返回所有图片完整信息
        self.logger.info(f"找到{len(target_images_context)}有效的图片")
        return target_images_context

    def _find_img_context_with_limit(self, md_content: str, img_name: str, max_chars=200) -> List[Tuple[str, str, str]]:
        """
         从MD文档中提取图片上下文信息
         思路：使用正则查找图片在md的位置
        Args:
            md_content:  操作的MD
            img_name:    图片名字
            max_chars:   最大的字符数限制
        Returns:
            List[Tuple[str, str, str]]
            List[("离图片最近的上面一个标题","图片的上文","图片的下文")]
        """
        # 1. 定义找图片的正则规则
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 切分md内容
        md_lines = md_content.split("\n")
        imgs_context = []

        # 3.遍历md
        for line_idx, line in enumerate(md_lines):

            # 3.1 是否是图片
            if not re_pattern.search(line):
                continue  # 继续下一行

            # 3.2 找上文标题内容和标题索引
            head_title = ""  # 初始标题内容
            head_index = -1  # 初始的标题索引
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            pre_content_start_index = head_index + 1
            pre_content = md_lines[pre_content_start_index:line_idx]

            # 3.3 找上文的内容(自下而上,反转)
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction="front")

            # 3.4 找下文标题索引(不要下文标题)
            section_index = len(md_lines)
            for i in range(line_idx + 1, section_index):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break

            post_content_start_index = line_idx + 1
            post_content = md_lines[post_content_start_index:section_index]

            # 3.5 找下文的内容(正常顺序)
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction="end")

            # 3.6 构建该图片的上下文信息
            imgs_context.append((head_title, img_pre_context, img_post_context))

        # 4.返回该md中当前图片的所有上下文信息（大多数情况下列表只要一个三元组对象） 除非该图片在md中有多处引用
        return imgs_context

    def _extract_img_context_with_limit(self, extract_content: list, max_chars: int, direction: str) -> str:

        """
        提取图片到上下标题（最近）之间的内容（段落）
        direction：front:自下而上
        direction：end:自上而下
        如何从给定的内容中找段落？
        策略：md中的段落 \n分割  补充：行与行之间 每一行后面都有两个空格
        Args:
            extract_content:  提取到的内容
            max_chars: 最大字符数
            direction: 放下昂

        Returns:
            str:上下文信息
        """
        current_paragraph = []  # 存储当前遍历到的内容
        final_paragraph = []  # 存储最终遍历到的段落（多个段落）

        # 1. 遍历每一行  收集段落
        for line in extract_content:
            clen_strip = line.strip()
            if not clen_strip:  # 自然而然的段落分割
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clen_strip):  # 遇到其它图片形成的段落
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)

        # 2. 处理最后一个段落
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        # 3.处理上文
        if direction == "front":
            final_paragraph.reverse()  # 找打离图片最近的文档

        # 4. 收集最终返回的段落
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)

            if total + para_len > max_chars and selected:
                break
            selected.append(para)  # 放入
            total += para_len  # 跟新计数器

        # 5. 处理上文
        if direction == "front":
            selected.reverse()  # 返回的顺序和原文档的顺序是一致的

        # 6. 返回上下文
        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title: str,
                             target_images_context: List[
                                 Tuple[str, str, Tuple[str, str, str]]],
                             config):
        """
         所有图片生成图片摘要 VLM (视觉语言模型)
        Args:
            document_title:  （文件）文档名字
            target_images_context: 所有图片信息
            config: 配置信息

        Returns:  Dict{"图片名字1":"摘要","图片名字2":"摘要2"}

        """
        self.log_step("step3", "准备提取图片摘要")

        summaries = {}
        request_timestamps: Deque[float] = deque()
        # 1. 构建OpenAI客户端
        try:
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base
            )
        except Exception as e:
            logging.error(f"VLM客户端创建失败")
            return summaries

        # 2. 发送请求（提取摘要）
        for img_name, img_path, images_context in target_images_context:

            self._enforce_rate_limit(request_timestamps, config.requests_per_minute, 60)

            summary = self._get_img_summary(config, client, document_title, img_path, images_context)
            summaries[img_name] = summary

        # 3. 返回映射表
        logging.info(f"生成{len(summaries)}图片摘要")

        return summaries

    def _enforce_rate_limit(
            self,
            request_timestamps: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ):
        """
        强制执行 API 请求速率限制。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(current_time)

    def _get_img_summary(self, config, client, document_title: str, img_path: str,
                         images_context: Tuple[str, str, str]) -> str:

        # 1. 解包images_context构建上下文
        section_title, pre_context, post_contex = images_context

        # 2. 判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_contex:
            context_parts.append(post_contex)

        # 3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        # 4. 读取图片文件
        local_img_content = ""
        try:
            with open(img_path, "rb") as f:
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
            # 5. 发送请求
        except Exception as e:
            return "暂无图片"

        # 5.调用 VLM
        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                                背景信息：
                                    1. 所属文档标题："{document_title}"
                                    2. 图片上下文：{final_context}
                                    请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                    生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ]
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "图片描述"

    def _upload_img_and_update_new_md(self, document_name, md_content, images_summaries, target_images_context, config

                                      ):

        """
        上传图片到minio以及替换md中的图片url和摘要
        Args:
            md_content:
            images_summaries:
            target_images_context:

        Returns:

        """
        self.log_step("step5","上传图片到minio并且更新md的摘要和图片地址")

        remote_urls = {}
        # 1. 构建MinIO客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning(f"无法将本地的图片上传到minio")  # 可选

        # 2. 遍历图片信息列表
        for img_name, img_path, _ in target_images_context:

            # 2.1 构建对象的名字
            object_name = f"{document_name}/{img_name}"

            try:
                # 2.2 开始上传
                minio_client.fput_object(
                    config.minio_bucket,
                    object_name,
                    img_path,
                )
                # 2.3 手动拼接远程地址
                # http://192.168.200.130:9000/test/temp_3.png
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                self.logger.info(f"{img_name}图片上传到minio成功")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"{img_name}上传到minio失败")
                remote_urls[img_name] = "http://minio_mock/" + document_name + "/" + img_name  # 可选

        self.logger.info(f"成功上传{len(remote_urls)}图片到minio")

        # 3. 替换（摘要和图片地址）到MD内容中
        new_md_content = md_content
        for img_name, images_summary in images_summaries.items():
            # 3.1 提取远程地址
            remote_url = remote_urls.get(img_name)

            if not remote_url:
                continue  # 不替换md中该图片

            # 3.2 替换url和摘要
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(f"![{images_summary}]({remote_url})", new_md_content)

        return new_md_content


if __name__ == '__main__':
    setup_logging()
    img_md_node = MarkDownImageNode()

    state = {
        "md_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }

    img_md_node.process(state)
