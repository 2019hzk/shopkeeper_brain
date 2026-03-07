from neo4j import GraphDatabase

# ========== 1. 建立连接 ==========
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "hzk123456")
DATABASE = "neo4j"

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
print("连接成功！")


# ========== 2. 写入 ==========
# with driver.session(database=DATABASE) as session:
#     # 先清空
#     session.run("MATCH (n) DETACH DELETE n").consume()
#     # consume() 就是告诉驱动"我不需要逐行读取结果了，直接跑完这条语句，给我个执行摘要就行"。确保上一步做完了，再做下一步"
#     # 返回 ResultSummary——里面包含执行统计信息，比如删除了多少节点、多少关系、耗时多少等
#
#     # 创建节点
#     session.run(
#         "CREATE (:Customer {name: $name, age: $age, vip: $vip})",
#         name="张三", age=28, vip=True,
#     ).consume()
#
#     session.run(
#         "CREATE (:Customer {name: $name, age: $age, vip: $vip})",
#         name="李四", age=35, vip=True,
#     ).consume()
#
#     # 创建关系
#     session.run("""
#         MATCH (a:Customer {name: $a_name}), (b:Customer {name: $b_name})
#         CREATE (a)-[:FRIEND]->(b)
#     """, a_name="张三", b_name="李四").consume()
#
#     print("写入完成！")


# ========== 3. 查询 ==========
with driver.session(database=DATABASE) as session:
    result = session.run(
        "MATCH (c:Customer) RETURN c.name AS name, c.age AS age, c.vip AS vip"
    )
    for record in result:
        print(record.data())
    # 输出: {'name': '张三', 'age': 28, 'vip': True}
    #       {'name': '李四', 'age': 35, 'vip': True}


# ========== 4. 更新 ==========
# with driver.session(database=DATABASE) as session:
#     result = session.run(
#         "MATCH (c:Customer {name: $name}) SET c.city = $city RETURN c.name, c.city",
#         name="张三", city="北京",
#     )
#     print(result.single().data())  # {'c.name': '张三', 'c.city': '北京'}

#
# # ========== 5. 删除 ==========
with driver.session(database=DATABASE) as session:
    session.run("MATCH (c:Customer {name: $name}) DETACH DELETE c", name="李四").consume()
    print("李四已删除")


# ========== 6. 关闭连接 ==========
driver.close()