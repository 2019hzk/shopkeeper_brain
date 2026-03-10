"""
学生管理小程序 —— 演示 PyMongo 的完整 CRUD 操作
"""

from pymongo import MongoClient


def get_collection():
    """获取数据库集合。"""
    client = MongoClient("mongodb://admin:123456@192.168.200.130:27017")
    return client["school"]["students"]


def add_student(coll, name: str, age: int, major: str):
    """新增学生。"""
    result = coll.insert_one({"name": name, "age": age, "major": major})
    print(f"[新增] {name}，ID: {result.inserted_id}")


def list_students(coll):
    """列出所有学生。"""
    students = list(coll.find({}, {"_id": 0}))
    if not students:
        print("[查询] 暂无学生记录")
        return
    print(f"[查询] 共 {len(students)} 名学生：")
    for s in students:
        print(f"  - {s['name']}，{s['age']} 岁，{s['major']}")


def update_student_age(coll, name: str, new_age: int):
    """修改学生年龄。"""
    result = coll.update_one({"name": name}, {"$set": {"age": new_age}})
    if result.matched_count:
        print(f"[更新] {name} 的年龄已改为 {new_age}")
    else:
        print(f"[更新] 未找到学生 {name}")


def delete_student(coll, name: str):
    """删除学生。"""
    result = coll.delete_one({"name": name})
    if result.deleted_count:
        print(f"[删除] 已删除 {name}")
    else:
        print(f"[删除] 未找到学生 {name}")


if __name__ == "__main__":
    coll = get_collection()

    # 清空旧数据，从头演示
    # coll.delete_many({})

    # C - 新增
    # add_student(coll, "张三", 20, "计算机科学")
    # add_student(coll, "李四", 22, "软件工程")
    # add_student(coll, "王五", 21, "人工智能")

    # R - 查询
    # list_students(coll)

    # # U - 更新
    # update_student_age(coll, "张三", 21)
    #
    # # D - 删除
    # delete_student(coll, "王五")
    #
    # # 最终结果
    # print("\n--- 最终数据 ---")
    list_students(coll)