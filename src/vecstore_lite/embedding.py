from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterable

import numpy as np


def hash_embed(text: str, dim: int = 64, *, seed: int = 0) -> np.ndarray:
    """离线文本哈希嵌入（无需 API Key）。

    将文本分词（空格 / 字符 bigram 混合）映射到固定维度稠密向量，
    适合本地 demo、单测与无网环境。**不是**生产级语义模型。
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        tokens = ["<empty>"]
    for token in tokens:
        digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=16).digest()
        # 两个 uint64 -> 多个桶
        a, b = struct.unpack("<QQ", digest)
        for value in (a, b, a ^ b, (a << 1) ^ b):
            idx = value % dim
            sign = 1.0 if (value >> 1) & 1 else -1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _tokenize(text: str) -> list[str]:
    text = text.strip().lower()
    if not text:
        return []
    parts = [p for p in text.replace("\n", " ").split(" ") if p]
    # 中文等无空格文本：补充字符 bigram
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) >= 2 and (len(parts) <= 1 or any(ord(ch) > 127 for ch in compact)):
        parts.extend(compact[i : i + 2] for i in range(len(compact) - 1))
    return parts


def random_projection(vector: Iterable[float] | np.ndarray, out_dim: int, *, seed: int = 42) -> np.ndarray:
    """高斯随机投影降维（JL lemma 近似），用于高维向量压缩存储/检索。"""
    arr = np.asarray(list(vector) if not isinstance(vector, np.ndarray) else vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("vector must be 1-D")
    if out_dim <= 0:
        raise ValueError("out_dim must be positive")
    if out_dim >= arr.size:
        return arr.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    # 固定种子矩阵：按输入维度缓存更佳，这里按调用生成以保证确定性（同 seed+in_dim）
    matrix = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(arr.size, out_dim)).astype(np.float32)
    return arr @ matrix
