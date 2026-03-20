"""答案输出节点 —— 骨架版本（第一步）"""

from typing import List, Dict, Any, Tuple
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_util import get_llm_client


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 已有答案 → 直接返回
        if state.get("answer"):
            return state

        # 2. 构建提示词 → 调用 LLM 生成答案
        prompt = self._build_prompt(state)
        state["prompt"] = prompt
        state["answer"] = self._invoke_generate(prompt)

        return state

    def _build_prompt(self, state: QueryGraphState) -> str:
        char_budget = self.config.max_context_chars

        # 1. 获取问题和商品名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化图谱关系
        graph_str, char_budget = self._format_kg_triples(
            state.get("kg_triples") or [], char_budget
        )


        # 4. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history="无历史对话",              # 第二部分加入
            item_names=", ".join(item_names),
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"), ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"), ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_kg_triples(kg_triples: List, char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            triple_text = (str(triple) if triple is not None else "").strip()
            if not triple_text:
                continue
            if used_chars + len(triple_text) > char_budget:
                break
            formatted_lines.append(triple_text)
            used_chars += len(triple_text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _invoke_generate(self, prompt: str) -> str:
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"


