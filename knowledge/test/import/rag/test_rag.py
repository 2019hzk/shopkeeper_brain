import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from neo4j import GraphDatabase


# 1. 配置与初始化
os.environ["OPENAI_API_KEY"] = "sk-26d57c968c364e7bb14f1fc350d4bff0"
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 2. Neo4j 连接配置
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "hzk123456"

# 3. 初始化 LLM 和 Embedding
llm = ChatOpenAI(model="qwen-flash", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key="sk-OdqRypFlfJLKYvLmV6GsG9j0u6CRFBYKErn4xV1Wm0R3q0y9",
                              openai_api_base="https://api.openai-proxy.org/v1",
                              model="text-embedding-3-large")

# 4. 测试用例：三段孤立的说明书切片
CHUNKS = [
    "Chunk 1: RS-12数字万用表的电池后盖由两颗十字螺丝固定。",
    "Chunk 2: 拆卸本设备的十字螺丝时，必须使用带有绝缘手柄的金属十字螺丝刀。",
    "Chunk 3: 警告：使用任何金属工具接触仪表内部前，必须先断开测试表笔，否则有触电危险。"
]

# 测试问题
TEST_QUESTION = "我要打开 RS-12 的电池后盖，需要注意什么安全事项？"

# 问答 Prompt 模板
QA_PROMPT = ChatPromptTemplate.from_template("""
基于以下已知信息，请专业、准确地回答用户的问题。如果已知信息中没有相关答案，请明确说明“根据提供的信息无法回答”。

已知信息：
{context}

问题：
{question}
""")


# ==========================================
# 1. 普通 RAG (纯向量检索)
# ==========================================
def run_standard_rag():
    print("\n" + "=" * 50)
    print("普通 RAG (纯向量检索)")
    print("=" * 50)

    # 1.1 将文本块存入向量数据库
    vectorstore = FAISS.from_texts(CHUNKS, embeddings)

    # 1.2 构建检索器 (为了对比 只取 Top 2 最相似的切片)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 1.3 组装 LangChain 问答链
    def format_docs(docs):
        context = "\n".join([doc.page_content for doc in docs])
        print(f" [普通 RAG 召回的上下文]:\n{context}\n")
        return context

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
    )

    # 4. 执行推理
    result = rag_chain.invoke(TEST_QUESTION)
    print(f" [大模型回答]: {result.content}")


# ==========================================
# 2. GraphRAG (图谱多跳检索)
# ==========================================
def run_graph_rag():
    print("\n" + "=" * 50)
    print(" GraphRAG (Neo4j 多跳逻辑检索)")
    print("=" * 50)

    clear_cypher = "MATCH (n) DETACH DELETE n"

    # 2.1 写入图谱数据的语句
    setup_cypher = """
    MERGE (c1:Chunk {id: 'c1', text: 'Chunk 1: RS-12数字万用表的电池后盖由两颗十字螺丝固定。'})
    MERGE (c2:Chunk {id: 'c2', text: 'Chunk 2: 拆卸本设备的十字螺丝时，必须使用带有绝缘手柄的金属十字螺丝刀。'})
    MERGE (c3:Chunk {id: 'c3', text: 'Chunk 3: 警告：使用任何金属工具接触仪表内部前，必须先断开测试表笔，否则有触电危险。'})

    MERGE (cover:Entity {name: '电池后盖'})
    MERGE (screw:Entity {name: '十字螺丝'})
    MERGE (tool:Entity {name: '金属螺丝刀'})
    MERGE (warning:Entity {name: '警告-断开测试表笔'})

    MERGE (cover)-[:SECURED_BY]->(screw)
    MERGE (screw)-[:REQUIRES_TOOL]->(tool)
    MERGE (tool)-[:HAS_WARNING]->(warning)

    MERGE (cover)-[:MENTIONED_IN]->(c1)
    MERGE (screw)-[:MENTIONED_IN]->(c1)
    MERGE (screw)-[:MENTIONED_IN]->(c2)
    MERGE (tool)-[:MENTIONED_IN]->(c2)
    MERGE (tool)-[:MENTIONED_IN]->(c3)
    MERGE (warning)-[:MENTIONED_IN]->(c3)
    """

    # 2.2 图检索逻辑 (从种子节点向外扩展 3 跳)
    retrieval_cypher = """
    MATCH path = (start:Entity {name: '电池后盖'})-[r*1..3]-(connected:Entity)
    WHERE NONE(rel IN r WHERE type(rel) = 'MENTIONED_IN')
    WITH nodes(path) AS entity_nodes
    UNWIND entity_nodes AS e
    MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
    RETURN DISTINCT c.text AS content
    """

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def graph_retriever(question):
        with driver.session() as session:
            # 第一步：先单独清理
            session.run(clear_cypher)
            # 第二步：再单独写入
            session.run(setup_cypher)
            # 第三步：执行查询
            records = session.run(retrieval_cypher)
            context = "\n".join([record["content"] for record in records])
            print(f" [GraphRAG 召回的上下文]:\n{context}\n")
            return context

    graph_chain = (
            {"context": graph_retriever, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
    )

    result = graph_chain.invoke(TEST_QUESTION)
    print(f" [大模型回答]: {result.content}")
    driver.close()


# ==========================================
# 3. 运行对比测试
# ==========================================
if __name__ == "__main__":
    print(f" 测试问题: {TEST_QUESTION}")
    run_standard_rag()
    run_graph_rag()