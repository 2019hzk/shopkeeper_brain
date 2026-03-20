import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _task_stream[task_id] = q
    return q

def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event (task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在
    if stream_queue:
        # 3. 将事件推送到队列
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    # 1. 获取队列
    stream_queue = get_sse_queue(task_id)

    # 2. 如果没有对应的队列，直接结束
    if stream_queue is None:
        return

    # 3. 获取当前运行的异步事件循环（用于后续将同步阻塞代码放入线程池执行）
    loop = asyncio.get_running_loop()
    try:
        # 4. 发送连接建立信号，告诉前端"管道已通"
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            # 5. 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            try:
                # 6. 使用 run_in_executor 避免阻塞 async 事件循环

                # block=True: 如果队列为空，是否阻塞等待
                # timeout=1.0: 阻塞的超时时间（秒），1.0 秒后若仍无数据则抛出 queue.Empty 异常
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # 7. 若 1 秒内队列为空（超时），则跳过本次循环，重新开始检查断开状态并继续监听
                continue

            # 8. 解析队列中获取到的消息体
            event = msg.get("event")
            data = msg.get("data")


            # 9. 将正常的事件和数据打包成标准 SSE 格式，并通过 yield 推送给前端客户端
            yield _sse_pack(event, data)

    except (ConnectionResetError, BrokenPipeError):
        # 客户端强行刷新页面或关闭标签页，TCP 管道破裂，静默退出
        return
    except asyncio.CancelledError:
        # 协程被取消：重新抛出，让外层框架知道它被成功取消了
        raise
    finally:
        # 清理资源（防止内存泄漏）
        remove_sse_queue(task_id)