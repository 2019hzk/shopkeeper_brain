from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r'D:\ai_models\modelscope_cache\models\BAAI\bge-m3',
    # 嵌入模型的名字(如果本地已经下载了，直接把嵌入模型的path放在此处 远程下载：【huggingface】)
    device='cuda:0',  # 设备名字:cpu/cuda:0[矩阵的运算]：cpu:慢 gpu:快
    use_fp16=True  # 半精度fp16(空间利用率少一些，计算速度快速度：gpu) 单精度: fp32
)
print(bge_m3_ef)

