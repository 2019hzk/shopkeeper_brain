from dotenv import load_dotenv

load_dotenv()
import os.path
import uuid
from datetime import datetime
import shutil
from typing import Tuple
from fastapi import UploadFile, HTTPException
from knowledge.core.paths import get_local_base_dir
from knowledge.utils.minio_util import get_minio_client
from knowledge.services.task_service import TaskService
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.main_graph import kb_import__graph_app


class ImportFileService:
    """

    文件导入的业务类

    1. 保存上传文件到本地
    2. 保存上传文件到MinIO
    3. 运行图谱的所有节点
    """

    def __init__(self, task_service: TaskService):
        self._task_service = task_service

    def get_date_dir(self) -> str:
        return os.path.join(get_local_base_dir(), datetime.now().strftime('%Y%m%d'))

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
         将上传的文件保存到本地
        Args:
            file: 上传的文件
            file_dir: 文件的归档目录

        Returns:

        """

        # 1. 确保归档目录存在
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建上传文件的完整的path
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 写入(批量的写入shutil.copyfileobj(file.file,f))
        with open(import_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # 4. 返回导入文件的path
        return import_file_path

    def save_upload_file_to_minio(self, import_file_path: str, file: UploadFile):
        """

        Args:
            import_file_path:
            file:

        Returns:

        """

        # 1. 获取minio客户端
        minio_client = get_minio_client()

        # 2. 判断minio客户端是否存在
        if not minio_client:
            raise HTTPException(status_code=500, detail="MinIO 服务不可用")

        # 3. 构建Minio客户端对象名（归档文件）
        minio_object_name = f"origin_files/{datetime.now().strftime('%Y%d%m')}/{file.filename}"

        # 4. 获取桶名
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        # 5. 开始上传
        try:
            minio_client.fput_object(bucket_name, minio_object_name, import_file_path)
        except Exception as e:
            raise ValueError(f"{file.filename}文件上传失败 原因:{e}")

    def process_upload_file(self, file: UploadFile) ->Tuple[str,str,str]:
        """
        处理上传文件
        Returns:
        1. 标记当前文件上传节点（"upload_file"）为正在运行中
        2. 将上传的文件保存到本地
        3. 将上传的文件保存到minio
        4. 标记当前文件上传节点（"upload_file"）为运行完毕
        5. 需要返回三部分信息（taski_id file_dir import_file_path）
        """

        # 1. 构建时间日期的文件目录出来
        date_dir = self.get_date_dir()

        # 2. 生成一个任务id
        task_id = str(uuid.uuid4())

        # 3. 构建文件的最终归属目录
        file_dir = os.path.join(date_dir, task_id)

        self._task_service.mark_node_running(task_id, "upload_file")
        # 4. 将接收到的文件上传本地
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 5. 将本地磁盘的上传文件同步minio中
        self.save_upload_file_to_minio(import_file_path, file)
        self._task_service.mark_node_done(task_id, "upload_file")

        # 6. 构建返回值
        return task_id, file_dir, import_file_path

    def  run_import_graph(self,task_id:str,file_dir:str,import_file_path:str):
        """
        运行导入graph的流程（跑节点）

        1. 构建初始状态
        graph.stream()之前调用update_task_status更新任务的状态为processing
        2. 运行（graph.stream()）
        2.1 在运行图的某个节点之前调用mark_node_running
        2.2 在运行图的某个节点结束调用mark_node_done
        graph.stream()执行完所有节点 update_task_status更新任务的状态为completed
        Args:
            task_id:
            file_dir:
            import_file_path:

        Returns:

        """
        try:
            # 1. 标记任务开始处理
            self._task_service.update_task_status(task_id, "processing")
            # 2. 构建 LangGraph 初始状态
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 流式执行整个导入流水线
            for event in kb_import__graph_app.stream(global_graph_init_status):
                for key, value in event.items():
                    print(f"[{task_id}] Completed Node: {key}")

            # 4. 标记任务完成
            self._task_service.update_task_status(task_id, "completed")

        except Exception as e:
            self._task_service.update_task_status(task_id, "failed")
            print(f"[{task_id}] Error: {e}")





