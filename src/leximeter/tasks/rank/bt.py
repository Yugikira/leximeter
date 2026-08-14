"""Bradley–Terry model fitting (Zermelo's iterative algorithm)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def fit_bt(
    item_ids: Sequence[str],
    outcomes: Sequence[Tuple[str, str]],
    *,
    pseudo: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """Fit a Bradley–Terry model to pairwise ``(winner, loser)`` outcomes.

    Returns a mapping ``item_id -> log-strength`` (up to an additive constant).
    """
    n = len(item_ids)
    if n == 0:
        return {}
    idx = {item: i for i, item in enumerate(item_ids)}

    wins = [pseudo] * n
    n_ij = [[0.0] * n for _ in range(n)]
    for winner, loser in outcomes:
        wi, li = idx[winner], idx[loser]
        wins[wi] += 1.0
        n_ij[wi][li] += 1.0
        n_ij[li][wi] += 1.0

    total = sum(wins) or 1.0
    p = [w / total for w in wins]

    for _ in range(max_iter):
        p_new: List[float] = []
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j != i and n_ij[i][j] > 0:
                    denom += n_ij[i][j] / (p[i] + p[j])
            p_new.append(wins[i] / denom if denom > 0 else p[i])
        s = sum(p_new) or 1.0
        p_new = [x / s for x in p_new]
        if max(abs(p_new[i] - p[i]) for i in range(n)) < tol:
            p = p_new
            break
        p = p_new

    return {item_ids[i]: math.log(p[i]) for i in range(n)}
