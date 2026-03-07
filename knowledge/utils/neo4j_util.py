import os
import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
_neo4j_driver = None


def get_neo4j_driver() -> GraphDatabase:
    """
    获取 Neo4j 驱动实例（单例模式）
    """
    global _neo4j_driver
    try:
        if _neo4j_driver is None:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            logger.info(f"正在初始化 Neo4j 驱动，连接 URI: {uri}")

            _neo4j_driver = GraphDatabase.driver(
                uri=uri,
                auth=(username, password)
            )
            # Neo4j 驱动默认是懒加载，这行代码能确保如果账号密码错误或网络不通，当场就会抛出异常，而不是等到插入数据时才报错。
            _neo4j_driver.verify_connectivity()

            logger.info("Neo4j 驱动初始化成功并已验证连接！")

        return _neo4j_driver

    except Exception as e:
        # exc_info=True 会在日志中打印出完整的 Error Traceback 堆栈，方便排查到底是密码错还是网络不通
        logger.error(f"初始化 Neo4j 驱动失败: {e}", exc_info=True)
        return None