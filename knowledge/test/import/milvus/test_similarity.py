"""
演示：L2 归一化后的向量内积 = 余弦相似度，范围 [-1, 1]
"""
import numpy as np
def l2_normalize(vec):
    """L2 归一化：每个分量除以向量的 L2 范数"""
    norm = np.linalg.norm(vec)
    return vec / norm


def cosine_similarity(a, b):
    """余弦相似度公式：cos(θ) = (a·b) / (||a|| * ||b||)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("=" * 60)
print("场景 1：方向相同")
print("=" * 60)

a = np.array([3.0, 4.0])
b = np.array([6.0, 8.0])  # 与 a 同方向，只是长度不同

a_norm = l2_normalize(a)
b_norm = l2_normalize(b)

print(f"原始向量 a = {a},  L2范数 = {np.linalg.norm(a):.4f}")
print(f"原始向量 b = {b},  L2范数 = {np.linalg.norm(b):.4f}")
print(f"归一化后 a = {a_norm}, L2范数 = {np.linalg.norm(a_norm):.4f}")
print(f"归一化后 b = {b_norm}, L2范数 = {np.linalg.norm(b_norm):.4f}")
print(f"归一化后内积 (IP)  = {np.dot(a_norm, b_norm):.4f}")
print(f"余弦相似度 (cosine) = {cosine_similarity(a, b):.4f}")
print()

print("=" * 60)
print("场景 2：方向完全相反")
print("=" * 60)

a = np.array([3.0, 4.0])
b = np.array([-6.0, -8.0])  # 与 a 完全反方向

a_norm = l2_normalize(a)
b_norm = l2_normalize(b)

print(f"原始向量 a = {a}")
print(f"原始向量 b = {b}  (与 a 完全反方向)")
print(f"归一化后 a = {a_norm}")
print(f"归一化后 b = {b_norm}")
print(f"归一化后内积 (IP)  = {np.dot(a_norm, b_norm):.4f}")
print(f"余弦相似度 (cosine) = {cosine_similarity(a, b):.4f}")
print()

print("=" * 60)
print("场景 3: 模拟 BGE-M3 高维向量（1024维）")
print("=" * 60)

np.random.seed(42)
# 模拟两个 1024 维的 embedding
emb_a = np.random.randn(1024).astype(np.float32)
emb_b = np.random.randn(1024).astype(np.float32)

# L2 归一化（BGEM3EmbeddingFunction 默认 normalize_embeddings=True）
emb_a_norm = l2_normalize(emb_a)
emb_b_norm = l2_normalize(emb_b)

ip_result = np.dot(emb_a_norm, emb_b_norm)
cos_result = cosine_similarity(emb_a, emb_b)

print(f"向量维度: {len(emb_a)}")
print(f"归一化前 a 的 L2 范数: {np.linalg.norm(emb_a):.4f}")
print(f"归一化后 a 的 L2 范数: {np.linalg.norm(emb_a_norm):.4f} (单位向量)")
print(f"归一化后内积 (IP)  = {ip_result:.6f}")
print(f"余弦相似度 (cosine) = {cos_result:.6f}")


# L2 归一化消除了向量的"长度"信息，只保留"方向"信息，所以内积就退化成了余弦值，范围就是 [-1, 1]。
