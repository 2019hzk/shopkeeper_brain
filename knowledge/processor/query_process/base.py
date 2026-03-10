"""查询流程节点基类

定义统一的节点接口规范，提供通用功能。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """查询流程节点基类。

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    Attributes:
        name: 节点名称，子类应覆盖。
        config: 配置对象。
        logger: 日志记录器。

    Example:
        >>> class MyNode(BaseNode):
        ...     name = "my_node"
        ...
        ...     def process(self, state):
        ...         # 实现具体逻辑
        ...         return state
        ...
        >>> # 作为 LangGraph 节点使用
        >>> node = MyNode()
        >>> workflow.add_node("my_node", node)
    """

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        """初始化节点。

        Args:
            config: 配置对象，默认使用全局配置。
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口。

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典。

        Returns:
            更新后的状态字典。

        Raises:
            QueryProcessError: 节点执行失败时抛出。
        """
        self.logger.info(f"--- {self.name} 开始 ---")

        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑。

        子类必须实现此方法。

        Args:
            state: 图状态字典。

        Returns:
            更新后的状态字典。
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。

        Args:
            step_name: 步骤名称。
            message: 附加信息。
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)




def setup_logging(level: int = logging.INFO):
    """配置查询流程日志。

    Args:
        level: 日志级别，默认 INFO。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
