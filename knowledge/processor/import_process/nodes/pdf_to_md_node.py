import json
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError


class PdfToMdNode(BaseNode):
    """
      pdf转换md节点
    """
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            self:
            state:

        Returns:
        """
        # 1. 对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2. 利用MinerU工具解析pdf成为md
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        # 3. 获取md的path
        md_path = self._get_md_paths(import_file_path, file_dir_path)

        # 4. 更新state 字典的md_path
        state['md_path'] = md_path

        # 5. 返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        Args:
            state:  该节点接收到的状态

        Returns:

        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf文件路径
        import_file_path = state.get('import_file_path', '')

        # 2. 获取解析后的输出目录
        file_dir = state.get('file_dir', '')

        # 3.校验输出的文件路径(非空判断)
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 4. 用Path标准化
        import_file_path_opj = Path(import_file_path)

        # 5. 校验是一个真实的路径
        if not import_file_path_opj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        # 6.判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_opj.parent

        # 7. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8.返回 输出文件以及输出目录的标准path
        return import_file_path_opj, file_dir_path_obj

    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path):
        """

        Args:
            import_file_path:  解析的文件路径
            file_dir_path:     解析后的文件目录

        Returns:
         mineru -p <input_path> -o <output_path>  --source local
        """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 构建命令行
        cmd = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local"
        ]

        import time
        process_start_time = time.time()
        # 2. 执行命令行(子进程执行命令行) 自动读取到主进程的环境变量
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                errors="replace",  # ？ 或者空心的菱形
                                text=True,  # 输出的内容是字符串 不是字节
                                encoding="utf-8",  # 用指定的中文字符集  进行编解码
                                bufsize=1  # 按行换车区  只要缓冲区一行满了就给我
                                )

        #  3. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")

        #  4.等待子进程做完(主进程等待子进程做完) 如果主进程等到子进程做完得到的状态码是0 代表子进程成功执行完了。 反之，没有执行成功
        processed_code = proc.wait()

        process_end_time = time.time()
        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path.name} 耗时:{process_end_time - process_start_time:.2f}s")
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path.name}失败")

        # 5. 返回状态码
        return processed_code

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path):
        # file_dir_path:D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir
        # name: 名字.xxx  stem:只拿名字  后缀：suffix
        file_name = import_file_path.stem

        md_path = file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md"
        return str(md_path)


###########测试
if __name__ == '__main__':
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node_init_state = {
        "import_file_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir"
    }
    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)

    print(json.dumps(processed_result, indent=4,ensure_ascii=False))
