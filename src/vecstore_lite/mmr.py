from __future__ import annotations

import numpy as np


def mmr_rerank(
    candidate_ids: list[str],
    candidate_scores: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance 重排，提升结果多样性。

    返回选中的候选下标列表。
    """
    if top_k <= 0 or not candidate_ids:
        return []
    n = len(candidate_ids)
    k = min(top_k, n)
    # 归一化向量便于算相似度
    norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-12
    normalized = candidate_vectors / norms

    selected: list[int] = []
    remaining = set(range(n))
    # 先取相关度最高的
    first = int(np.argmax(candidate_scores))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_idx = None
        best_val = -1e18
        for idx in remaining:
            relevance = float(candidate_scores[idx])
            diversity = max(
                float(normalized[idx] @ normalized[s]) for s in selected
            )
            value = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            if value > best_val:
                best_val = value
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected
