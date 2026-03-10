import os

# 项目根目录（.当前目录 ..上一层目录）D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge
KNOWLEDGE_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端页面静态资源目录   D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\front
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")

def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR