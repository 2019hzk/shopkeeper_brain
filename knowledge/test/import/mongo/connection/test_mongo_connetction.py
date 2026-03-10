from pymongo import MongoClient
from pymongo.collection import Collection

# 1. 定义MongoDB的客户端
mongo_client = MongoClient("mongodb://admin:123456@192.168.200.130:27017")

# 2. 创建库
db = mongo_client["my_db"]

# 3. 创建表
collection = db["students"]

print(db)
print(collection)


def insert_document(connection: Collection):
    result = collection.insert_one({
        "name": "张三",
        "age": 20,
        "major": "计算机科学"
    })

    print(result)


def insert_documents(connection: Collection):
    results = collection.insert_many([
        {"name": "李四", "age": 22, "major": "软件工程"},
        {"name": "王五", "age": 21, "major": "计算机科学"},
    ])

    print(results)


def fetch_collection(connection: Collection):

    # 1. 查询全部
    # for document in collection.find():
    #     print(document['name'])

    # 2. 根据条件查询
    # for doc in collection.find({"major": "计算机科学"}):
    #     print(doc["name"], doc["age"])

    # 3. 查询单条记录
    # student = collection.find_one({"name": "张三"})
    # print(student)

    for doc in collection.find().sort("age", 1).limit(1):
        print(doc["name"], doc["age"])

def  update_document():
    # 更新单条
    result = collection.update_one(
        {"name": "张三"},  # 查询条件
        {"$set": {"age": 21}}  # 更新操作
    )
    print(f"匹配 {result.matched_count} 条，修改 {result.modified_count} 条")

    # 更新多条
    result = collection.update_many(
        {"major": "计算机科学"},
        {"$set": {"status": "在读"}}
    )
    print(f"修改 {result.modified_count} 条")

def  delete_document():
    # 删除单条
    result = collection.delete_one({"name": "王五"})
    print(f"删除 {result.deleted_count} 条")

    # 删除多条
    result = collection.delete_many({"age": {"$lt": 21}})
    print(f"删除 {result.deleted_count} 条")


if __name__ == '__main__':
    # insert_document(collection)

    # insert_documents(collection)

    # fetch_collection(collection)

    # update_document()

    delete_document()