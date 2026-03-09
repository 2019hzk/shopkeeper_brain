"""
上传相关 Schema 定义
"""
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    """文件上传响应"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")  # 改为单个字符串