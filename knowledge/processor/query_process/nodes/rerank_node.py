from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, setup_logging, T
from knowledge.utils.bge_rerank_util import get_reranker_model

"""
嵌入--->嵌入检索(混合策略检索：WeightReranker:归一化)（distance:分数【0,1】）
   ----Hyde检索(混合策略检索：WeightReranker:归一化)（distance:分数【0,1】）
   ----kg检索(自定义weight:种子节点权重固定2.0,邻居节点权重固定1.0)（sum(weight)分数【1.0,+∞】）
   ----mcp检索(没打分)****
   ----rrf(【不同路设置不同的权重：选择性】多路检索融合打分排序 打分利用RRF公式，sorted降序排序） 
   ----rerank(重新进行精排打分：不用管，三方做好了)
"""


class RerankNode(BaseNode):
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取query
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        # 3. Rerank精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4. 动态Top_K截取(断崖检测)
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        state['reranked_docs'] = cutoff_docs

        return state

    def _cliff_cutoff(
            self, ranked_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []

        upper_bound = min(self.config.rerank_max_top_k, len(ranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)

        cutoff_pos = upper_bound
        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[i].get("score")
            next_score = ranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            abs_gap = current_score - next_score  # 0.3
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                self.logger.debug(
                    f"断崖检测: 位置 {i + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break

        return ranked_docs[:cutoff_pos]

    def _merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """

        Args:
            state:

        Returns:

        """
        final_docs = []
        # 1. 获取本地RRF的文档
        for rrf_doc in (state.get('rrf_chunks') or []):

            # 1.1 判断当前文档对象类型
            if not isinstance(rrf_doc, dict):
                continue

            # 1.2 获取文档的内容
            content = rrf_doc.get('content', '').strip()

            # 1.3 判断文档内容
            if not content:
                continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id')
            # 1.4 格式化本地RRF的chunk结构
            format_rrf_doc = self._format_rrf_docs(content=content, title=title, chunk_id=chunk_id, source="local")
            final_docs.append(format_rrf_doc)

        # 2. 获取web远程的文档
        for web_doc in (state.get('web_search_docs') or []):

            # 2.1 判断当前文档对象类型
            if not isinstance(web_doc, dict):
                continue

            # 2.2 获取文档内容(snippet)
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()

            # 2.3 判断内容
            if not content:
                continue
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()

            # 2.4 格式化web的文档结构
            format_web_doc = self._format_rrf_docs(content=content, title=title, url=url, source="web")
            final_docs.append(format_web_doc)

        self.logger.info(f"收集到准备进行Rerank精排的文档 {len(final_docs)}")

        return final_docs

    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None, url: str = "", source: str = "") -> Dict[
        str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """

        Args:
            user_query: 用户输出的查询问题
                merged_multi_docs: 不同来源的两路合并之后的文档（all_in_docs）
            Returns:

            """

        # 1. 判断合并后的多源文档是否存在
        if not merged_multi_docs:
            return []

        # 2. 获取reranker模型
        rerank_model = get_reranker_model()
        if rerank_model is None:
            self.logger.error("重排序模型获取失败")
            return []

        # 3. 构建Q->D的pair对 [(Q,D1[content]),(Q,D2:[content]),(Q,Dn)]
        query_doc_content_pairs = [(user_query, doc.get('content')) for doc in merged_multi_docs]
        try:
            # 4. 计算 [-3.558016300201416, 6.255949974060059, -3.1299989223480225, 0.025800857692956924]
            rerank_scores = rerank_model.compute_score(sentence_pairs=query_doc_content_pairs)

            # 5. 映射分数和文档
            score_doc = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

            # 6. 排序 并返回
            sored_score_docs = sorted(score_doc, key=lambda x: x["score"], reverse=True)

            return sored_score_docs
        except Exception as e:
            self.logger.error(f"Rerank重排序失败:{str(e)}")
            return [{**merged_multi_docs, "score": None}]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
