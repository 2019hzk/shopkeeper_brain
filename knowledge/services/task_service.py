from knowledge.utils.task_util import (
    add_running_task,
    add_done_task,
    update_task_status,
    get_task_status,
    get_running_task_list,
    get_done_task_list)


class TaskService:
    """
    任务服务类
    """

    # 1. 标记节点正在运行
    def mark_node_running(self, task_id: str, init_node_name: str):
        add_running_task(task_id, init_node_name)

    # 2. 标记节点运行完成
    def mark_node_done(self, task_id: str, init_node_name: str):
        add_done_task(task_id, init_node_name)

    # 3. 更新任务状态
    def update_task_status(self, task_id: str, status: str):
        update_task_status(task_id, status)

    # 4. 查询任务状态
    def get_task_status(self, task_id: str):
        get_task_status(task_id)

    # 5. 查询任务信息(任务的全局信息)
    def get_task_info(self, task_id: str):
        return {
            "status": get_task_status(task_id),
            'done_list': get_done_task_list(task_id),
            'running_list': get_running_task_list(task_id)
        }
