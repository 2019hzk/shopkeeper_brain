import json

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.base import setup_logging


def create_import_graph() -> CompiledStateGraph:
    """
    定义导入业务的graph状态拓扑谱（langgraph构建流水线）整个流水线各个节点要读取或者写入的节点。
    Returns:

    """

    # 1. 定义状态图
    graph_pineline = StateGraph(ImportGraphState)  # type:ignore

    # 2. 定义节点（入口、结束节点、自己需要添加的）
    # 2.1 定义入口节点
    graph_pineline.set_entry_point("entry_node")

    # 2.2 添加剩下的节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode()
    }
    for key, value in nodes.items():
        graph_pineline.add_node(key, value)

    # 3. 定义边（顺序边、条件边）
    # TODO:条件边
    # graph_pineline.add_conditional_edges()
    graph_pineline.add_edge("entry_node", "pdf_to_md_node")
    graph_pineline.add_edge("pdf_to_md_node", END)

    # 4. 编译（编排）
    return graph_pineline.compile()


graph_app = create_import_graph()


# 测试使用
def run_import_graph(import_file_path: str, file_dir: str):
    # 1. 构建state
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    init_state = create_default_state(**state)  # 发生了解包

    # 2. 调用stream(用流式获取每一个节点的处理情况：event事件[节点名字 节点处理后的状态])
    final_state = None
    for event in graph_app.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点的:{node_name},state:{state}")
            final_state = state

    return final_state


if __name__ == '__main__':
    setup_logging()

    import_file_path = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\万用表的使用.pdf"
    file_dir = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir"
    # 1. 测试编排流程
    final_state = run_import_graph(import_file_path=import_file_path, file_dir=file_dir)
    print(json.dumps(final_state, indent=2, ensure_ascii=False))

    # 2.打印图结构（ASCII 可视化）# 1. 单独安装：pip install grandalf 2.(单独安装还出错)  【pydantic：定义数据模型 】pip uninstall gradio  3. 单独安装 pip install grandalf 解决冲突
    print("-" * 50)
    print("图结构:")
    graph_app.get_graph().print_ascii()
