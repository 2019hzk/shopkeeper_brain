from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "hzk123456"))


# ---- 定义事务函数（所有操作在同一个事务中） ----
def create_shop_data(tx):
    """创建第（单事务完成）"""

    # 创建分类
    for cat in ['电子产品', '食品饮料', '服装鞋帽']:
        tx.run("CREATE (:Category {name: $name})", name=cat)

    # 创建商品并关联分类
    products = [
        {"name": "华为手机", "price": 5999, "cat": "电子产品"},
        {"name": "进口咖啡", "price": 89, "cat": "食品饮料"},
        {"name": "运动跑鞋", "price": 599, "cat": "服装鞋帽"},
        {"name": "有机牛奶", "price": 168, "cat": "食品饮料"},
    ]
    for p in products:
        tx.run("""
            CREATE (p:Product {name: $name, price: $price})
            WITH p
            MATCH (c:Category {name: $cat})
            CREATE (p)-[:BELONGS_TO]->(c)
        """, name=p["name"], price=p["price"], cat=p["cat"])

    # 创建客户
    customers = [
        {"name": "张三", "age": 28, "vip": True},
        {"name": "李四", "age": 35, "vip": True},
        {"name": "王五", "age": 22, "vip": False},
        {"name": "小红", "age": 26, "vip": False},
    ]
    for c in customers:
        tx.run("CREATE (:Customer {name: $name, age: $age, vip: $vip})",
               name=c["name"], age=c["age"], vip=c["vip"])

    # 创建购买关系
    purchases = [
        ("张三", "华为手机", 5999), ("张三", "进口咖啡", 89),
        ("李四", "华为手机", 5999), ("李四", "有机牛奶", 168),
        ("王五", "运动跑鞋", 599),
        ("小红", "进口咖啡", 89),  ("小红", "运动跑鞋", 599),
    ]
    for cname, pname, amount in purchases:
        tx.run("""
            MATCH (c:Customer {name: $cname}), (p:Product {name: $pname})
            CREATE (c)-[:PURCHASED {amount: $amount}]->(p)
        """, cname=cname, pname=pname, amount=amount)

    # 创建好友关系
    friends = [("张三", "李四"), ("张三", "王五"), ("王五", "小红")]
    for a, b in friends:
        tx.run("""
            MATCH (a:Customer {name: $a}), (b:Customer {name: $b})
            CREATE (a)-[:FRIEND]->(b)
        """, a=a, b=b)


# ---- 定义查询事务函数 ----
def query_customer_purchases(tx, customer_name):
    """查询某客户的购买记录"""
    result = tx.run("""
        MATCH (c:Customer {name: $name})-[r:PURCHASED]->(p:Product)
        RETURN p.name AS product, r.amount AS amount
    """, name=customer_name)
    return [record.data() for record in result]


def query_friend_recommendations(tx, customer_name):
    """好友推荐：朋友买了但我没买的商品"""
    result = tx.run("""
        MATCH (me:Customer {name: $name})-[:FRIEND]-(friend)-[:PURCHASED]->(p:Product)
        WHERE NOT EXISTS { (me)-[:PURCHASED]->(p) }
        RETURN p.name AS product, p.price AS price, friend.name AS friend
    """, name=customer_name)
    return [record.data() for record in result]


# ---- 执行 ----
with driver.session(database="neo4j") as session:
    # 清空旧数据
    # session.run("MATCH (n) DETACH DELETE n").consume()

    # 写入（事务函数，全部原子提交）
    # session.execute_write(create_shop_data)
    # print("图谱创建完成！")

    # # 查询（读事务函数，可自动重试 + 路由到只读节点）
    print("\n===== 张三的购买记录 =====")
    for r in session.execute_read(query_customer_purchases, "张三",):
        print(f"  {r['product']}: ¥{r['amount']}")

    # # print("\n===== 为张三推荐（好友） =====")
    for r in session.execute_read(query_friend_recommendations, "张三"):
        print(f"  {r['product']} ¥{r['price']} — 推荐人: {r['friend']}")


driver.close()