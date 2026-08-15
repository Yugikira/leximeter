"""Bradley–Terry model fitting (Zermelo's iterative algorithm) with standard errors.

Ported from GABRIEL's ``tasks/rank.py`` (``_fit_bt`` / ``_bt_standard_errors``):
``fit_bt`` runs the fixed-point MLE in log space and can optionally return the
match-count and predicted-probability matrices needed to estimate standard
errors; ``bt_standard_errors`` inverts the projected Fisher information to get
a covariance matrix that respects the sum-to-zero identifiability constraint.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fit_bt(
    item_ids: Sequence[str],
    outcomes: Sequence[tuple[str, str]],
    *,
    pseudo: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    return_info: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit a Bradley–Terry model to pairwise ``(winner, loser)`` outcomes.

    Parameters
    ----------
    item_ids:
        Unique item identifiers (their order fixes the matrix indexing).
    outcomes:
        ``(winner, loser)`` tuples.  Ties are represented by including both
        ``(a, b)`` and ``(b, a)``; each entry contributes one increment to
        the win matrix.
    pseudo:
        Pseudo count added to both win and total match counts (smoothing
        prior; acts as the BT "learning rate").
    max_iter, tol:
        Control convergence of the iterative fixed-point updates.
    return_info:
        If ``True`` return ``(scores, n_ij, p_ij)`` where ``n_ij`` holds the
        total match counts and ``p_ij`` the predicted win probabilities,
        both needed for standard-error estimation.

    Returns
    -------
    scores : dict
        Mapping from item identifier to estimated log-skill (zero-mean).
    (scores, n_ij, p_ij) : tuple
        When ``return_info`` is ``True``.
    """
    n = len(item_ids)
    if n == 0:
        if return_info:
            empty = np.zeros((0, 0), dtype=float)
            return {}, empty, empty
        return {}
    idx = {item: i for i, item in enumerate(item_ids)}
    # wins[i, j] counts how many times i beat j.
    wins = np.zeros((n, n), dtype=float)
    for w, l in outcomes:
        if w in idx and l in idx:
            wins[idx[w], idx[l]] += 1.0
    # total matches between each pair, and total wins per item.
    n_ij = wins + wins.T
    w_i = wins.sum(axis=1)
    # add pseudo counts
    n_ij += pseudo
    w_i += pseudo
    # fixed-point iteration (Zermelo): p_new = w_i / sum_j n_ij / (p_i + p_j)
    p = np.ones(n, dtype=float)
    for _ in range(max_iter):
        denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
        p_new = w_i / denom
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    # convert to log space and centre at zero mean
    s = np.log(p)
    s -= s.mean()
    scores = {item: float(val) for item, val in zip(item_ids, s)}
    if not return_info:
        return scores
    exp_s = np.exp(s)
    p_ij = exp_s[:, None] / (exp_s[:, None] + exp_s[None, :])
    return scores, n_ij, p_ij


def bt_standard_errors(
    s: np.ndarray,
    n_ij: np.ndarray,
    p_ij: np.ndarray,
    *,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Estimate standard errors for BT skill parameters.

    The observed Fisher information for the Bradley–Terry model is
    ``I = diag(q 1) - q`` where ``q = n_ij * p_ij * (1 - p_ij)`` encodes the
    uncertainty contributed by each pairwise comparison (Ford, 1957).  The
    estimates satisfy a sum-to-zero constraint, so ``I`` is rank deficient
    with the all-ones vector in its null space.  Instead of selecting an
    arbitrary reference item, we project ``I`` onto the constrained subspace
    and take its Moore–Penrose pseudoinverse.  A small ridge term stabilises
    the inversion for sparse comparison graphs.  The standard error for item
    ``i`` is the square root of the ``i``-th diagonal entry of the resulting
    covariance matrix.

    Parameters
    ----------
    s : np.ndarray
        Estimated log-skills for each item (used only for its length).
    n_ij : np.ndarray
        Total match counts between items (wins + losses), including the
        pseudo counts added by :func:`fit_bt`.
    p_ij : np.ndarray
        Predicted win probabilities between items.
    ridge : float
        Small constant added to the diagonal of the projected Fisher
        information matrix for numerical stability.

    Returns
    -------
    np.ndarray
        Standard errors corresponding to each element of ``s``.
    """
    n = len(s)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)

    q_ij = n_ij * p_ij * (1 - p_ij)
    diag = q_ij.sum(axis=1)
    info = np.diag(diag) - q_ij
    info = np.nan_to_num(info)
    ones = np.ones((n, 1))
    proj = np.eye(n) - ones @ ones.T / n
    info_proj = proj @ info @ proj
    info_proj[np.diag_indices(n)] += ridge
    try:
        cov = np.linalg.pinv(info_proj, rcond=1e-12)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(np.nan_to_num(info_proj), rcond=1e-12)
    cov = proj @ cov @ proj
    cov = 0.5 * (cov + cov.T)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return np.nan_to_num(se)
