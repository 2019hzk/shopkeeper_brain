
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具 (原生 PyMongo 实现)
    并增加了转换为 LangChain 消息对象的功能
    """
    
    def __init__(self):
        """
        初始化 MongoDB 连接
        """
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]
            
            # 创建索引以加速查询
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise




def clear_history( session_id: str) -> int:
    """
    6. 清空指定会话的历史记录
    """
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0

def save_chat_message( session_id: str, role: str, text: str, rewritten_query: str = "",
                       item_names: List[str] = None , message_id:str=None) -> str:
    """
    写入一条会话记录
    :param message_id: 主键
    :param rewritten_query:
    :param session_id: 会话 ID
    :param role: 角色 (user/assistant)
    :param text: 对话内容
    :param item_names: 关联的商品名称 (可选、可多个)
    :param ts: 时间戳 (可选，默认当前时间)
    :return: 插入记录的 ObjectId 字符串
    """

    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_item_names( ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史会话列表的 item_name
    """
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids},
             "$or": [
                 {"item_names": {"$exists": False}},
                 {"item_names": []},
                 {"item_names": None}
             ]
             },
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0

def get_recent_messages( session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
     查询最近 N 条对话记录 (返回原始字典格式)
    逻辑：
    - 必须匹配 session_id

    :param session_id: 会话 ID
    :param limit: 条数限制
    :return: 记录列表 (按时间正序排列，方便直接喂给 LLM)
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}

        # 按时间倒序查最近 limit 条
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        messages = list(cursor)

        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []


_history_mongo_tool = None

def get_history_mongo_tool() -> HistoryMongoTool:
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool

try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")
