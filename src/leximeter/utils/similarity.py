"""Vector similarity helpers for the embedding prefilter."""

from __future__ import annotations

import math
from typing import List, Sequence


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 for zero vectors."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def top_k_similar(
    query_vecs: Sequence[Sequence[float]],
    candidate_vecs: Sequence[Sequence[float]],
    k: int,
) -> List[List[int]]:
    """For each query, return the indices of its ``k`` most-similar candidates."""
    result: List[List[int]] = []
    for q in query_vecs:
        sims = sorted(
            ((cosine(q, c), i) for i, c in enumerate(candidate_vecs)),
            key=lambda x: -x[0],
        )
        result.append([i for _, i in sims[:k]])
    return result


def cluster_by_similarity(
    vecs: Sequence[Sequence[float]], threshold: float
) -> List[List[int]]:
    """Greedy connected-components clustering above ``threshold`` similarity."""
    n = len(vecs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if cosine(vecs[i], vecs[j]) >= threshold:
                union(i, j)

    clusters: dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())
