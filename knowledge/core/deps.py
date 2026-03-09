from functools import lru_cache

from knowledge.services.import_file_service import ImportFileService
from knowledge.services.task_service import TaskService

@lru_cache   # 缓存思想
def get_task_service() -> TaskService:
    task_service = TaskService()
    return task_service


@lru_cache  # 缓存思想
def get_import_file_service() -> ImportFileService:
    import_file_service = ImportFileService(get_task_service())
    return import_file_service
