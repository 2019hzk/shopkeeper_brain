"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from knowledge.processor.query_process.state import QueryGraphState, create_default_state

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        item_name_confirm
              │
              ├── (有答案) ────────────────────────────> answer_output
              │                                              │
              └── (无答案) ──> multi_search ─────┬──────────>│
                                   │             │           │
                         ┌─────────┼─────────────┼───────┐   │
                         │         │             │       │   │
                         v         v             v       v   │
                   embedding  hyde_embedding  query_kg  web  │
                         │         │             │       │   │
                         └─────────┴─────────────┴───────┘   │
                                       │                     │
                                       v                     │
                                     join                    │
                                       │                     │
                                       v                     │
                                      rrf                    │
                                       │                     │
                                       v                     │
                                    rerank                   │
                                       │                     │
                                       v                     │
                               answer_output <───────────────┘
                                       │
                                       v
                                      END
    """

    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState) # type:ignore

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": "",
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )

    # 6. 多路搜索分发（并行执行）
    # TODO

    # 7. 多路搜索汇合
    # TODO

    # 8. 顺序边
    # TODO

    # 9. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()









# -----------------------------------------
# 测试
# -----------------------------------------
def test_run_query(
        query: str,
        session_id: str = "",
        item_names: list = None,
        is_stream: bool = False
) -> dict:
    """便捷函数：运行查询流程。

    Args:
        query: 用户查询文本。
        session_id: 会话 ID。
        item_names: 已知的商品名称列表。
        is_stream: 是否启用流式输出。

    Returns:
        最终状态字典。
    """
    initial_state = create_default_state(
        session_id=session_id or "default",
        original_query=query,
        item_names=item_names or [],
        is_stream=is_stream,
    )

    final_state = None
    for event in query_app.stream(initial_state):
        for key, value in event.items():
            print(f"节点: {key}")
            final_state = value

    return final_state or initial_state


if __name__ == "__main__":
    pass
