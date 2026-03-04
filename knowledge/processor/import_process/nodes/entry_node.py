import json
from pathlib import Path
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError


class EntryNode(BaseNode):
    """
     实体节点
     位置：整个导入流程中的位置（第一位）
     作用：对上传的文件类型做判断（.pdf文件 or  .md文件）
    """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测
        Args:
            state: ImportGraphState 该节点处理之前的节点状态

        Returns:ImportGraphState：该节点处理之后的节点状态

        """

        # 1. 获取导入的文件路径以及文件所在的目录
        self.log_step("Step1", "[获取文件路径]")
        file_dir = state.get('file_dir')
        import_file_path = state.get('import_file_path')

        # 2. 简单校验一下 文件路径以及所在目录
        self.log_step("Step2", "[检测文件路径]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件不存在", self.name)

        # import_file_path: D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf
        # 3.使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)

        # 4. 获取上传文件的后缀
        suffix = path.suffix.lower()

        # 5. 判断文件的后缀
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path

        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        # 6. 获取文件的标题名
        file_title = path.stem
        state['file_title'] = file_title

        # 7. 返回state
        return state


############测试###############
if __name__ == '__main__':
    # pdf_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf"
    # 方式一： 直接实例该节点对象 调用process方法
    # setup_logging()
    # # 1. 构建该节点需要的state
    # test_entry_state = {
    #     "file_dir":r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir",
    #     "import_file_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf"
    # }
    #
    # # 2. 实例EntryNode节点
    # entry_node = EntryNode()
    #
    # # 3. 调用process方法
    # processed_state = entry_node.process(test_entry_state)
    #
    # # 序列化打印
    # print(json.dumps(processed_state, ensure_ascii=False, indent=4))

    # 方式一： 直接实例该节点对象 调用process方法
    setup_logging()
    # 1. 构建该节点需要的state
    test_entry_state = {
        "file_dir": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir",
        "import_file_path": r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf"
    }

    # 2. 实例EntryNode节点
    entry_node = EntryNode()

    # 3. 调用process方法
    processed_state = entry_node(test_entry_state)

    # 序列化打印
    print(json.dumps(processed_state, ensure_ascii=False, indent=4))

