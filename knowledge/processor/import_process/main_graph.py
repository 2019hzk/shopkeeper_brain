import json

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus_node  import ImportMilvusNode
from knowledge.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.base import setup_logging


def import_router(state: ImportGraphState):
    if state.get('is_md_read_enabled'):
        return "md_img_node"
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md_node"
    return END  # 安全降级


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
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node":DocumentSplitNode(),
        "item_name_rec_node":ItemNameRecognitionNode(),
        "bge_embedding_node":BgeEmbeddingChunksNode(),
        "import_milvus_node":ImportMilvusNode(),
        "kg_node":KnowLedgeGraphNode()
    }
    for key, value in nodes.items():
        graph_pineline.add_node(key, value)

    # 3. 定义边（顺序边、条件边）
    # source:  路由开始节点
    # path:    路由函数
    # path_map 路由函数的映射

    graph_pineline.add_conditional_edges("entry_node",
                                         import_router,
                                         {
                                             "md_img_node": "md_img_node",
                                             "pdf_to_md_node": "pdf_to_md_node",
                                             END: END
                                         }
                                         )

    # graph_pineline.add_conditional_edges()
    graph_pineline.add_edge("entry_node", "pdf_to_md_node")
    graph_pineline.add_edge("pdf_to_md_node","md_img_node")
    graph_pineline.add_edge("md_img_node","document_split_node")
    graph_pineline.add_edge("document_split_node","item_name_rec_node")
    graph_pineline.add_edge("item_name_rec_node","bge_embedding_node")
    graph_pineline.add_edge("bge_embedding_node","import_milvus_node")
    graph_pineline.add_edge("import_milvus_node","kg_node")
    graph_pineline.add_edge("kg_node",END)

    # 4. 编译（编排）
    return graph_pineline.compile()


kb_import__graph_app = create_import_graph()


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
    for event in kb_import__graph_app.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点的:{node_name}")
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
    kb_import__graph_app.get_graph().print_ascii()
