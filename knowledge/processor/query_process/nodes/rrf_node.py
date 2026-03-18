from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, setup_logging


class RrfNode(BaseNode):
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 各路搜索的结果（排除网络搜索:reranK节点做）--->本质：因为网络搜索的结果没有chunk_id,embedding_search(有chunk_id) hyde_search(有chunk_id) kg_search(chunk_id)
        # RRF(多路结果的融合：基于文档的排名，把多路都命中的文档，未来计算出来的得分更高，对应顺序靠前)
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get('embedding_chunks') or []
        # 1.2 获取hyde向量检索路的结果
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []
        # 1.3 获取kg图谱检索的结果
        kg_search_chunks = state.get('kg_chunks') or []

        # 2. 为不同路的搜索结果设置不同的权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
            "kg_search_result": (self._normalize_input(kg_search_chunks), 0.7)
        }
        # 3. 构建rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(rrf_inputs, self._rrf_k, self._top_k)

        # 5. 获取rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回state
        return state



    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路检索到的结果
        Args:
            rrf_input:  各路不同数据结构的检索结果

        Returns:
             统一处理后的标准数据结构的检索结果

        """

        diff_path_result = []
        # 1. 遍历各路搜索结果
        if not rrf_input:
            return []

        # 2. 遍历该路的所有结果
        for doc in rrf_input:

            # 2.1 判断是否有效
            if not isinstance(doc, dict):
                continue

            # 2.2  获取entity
            entity = doc.get('entity')
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, _rrf_k, _top_k) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用RRF公式计算每一个文档 的总得分，最后在对总得分进行排序（sorted）
        Args:
            rrf_inputs: 各路的搜索结果+权重
            _rrf_k: 平滑参数通常取60，如果头部排名的文档影响更大，k调小 如果向兼容多个排名的文档，k调整大一些
            _top_k: 合并完之后，返回的文档数
        rrf（score）= 参考课件公式
        Returns:
           合并以及排序后的文档

        """
        chunk_scores = {}  # 存放所有chunk（多路命中的chunk只留一份）的rrf计算后后的分数值
        chunk_data = {}  # 存放所有chunk
        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, float(0)) + weight / (_rrf_k + i)
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序，截取前 top_n 条(自己排序)
        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        return sorted_results[:_top_k] if _top_k else sorted_results




if __name__ == "__main__":


    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟三路检索结果
    # chunk_1 命中 3 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"id": None, "distance": 2.0, "entity": {"chunk_id": "chunk_5", "content": "知识图谱结果#1"}},
            {"id": None, "distance": 1.0, "entity": {"chunk_id": "chunk_1", "content": "知识图谱结果#2"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
