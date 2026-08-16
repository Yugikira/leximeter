"""OLS regression helpers (ported from GABRIEL's ``utils/plot_utils.py``).

:func:`fit_ols` fits a linear model with statsmodels when available, applying
robust HC3 covariance (falling back to HC1) by default.  When statsmodels is
missing, :func:`_fit_ols_fallback` provides a lightweight NumPy solver with
classical standard errors so the debiasing pipeline keeps working.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:  # statsmodels is optional; fall back to a lightweight solver if missing
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - fallback exercised when statsmodels absent
    sm = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False


def _safe_fvalue(res: Any) -> float:
    """Return ``res.fvalue`` while gracefully handling statsmodels failures."""
    try:
        return float(res.fvalue)
    except (ValueError, AttributeError):  # pragma: no cover - defensive
        return np.nan


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool = True,
    varnames: list[str] | None = None,
) -> dict[str, Any]:
    """Fit an OLS regression using statsmodels and return a dictionary of results.

    When ``varnames`` is supplied and its length matches the number of columns
    in ``X``, the design matrix is converted to a pandas DataFrame with those
    column names so that parameter estimates in the statsmodels summary carry
    meaningful names instead of ``x1``, ``x2``.  Robust HC3 standard errors
    are applied by default, falling back to HC1 when HC3 fails.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable values.
    X : ndarray of shape (n, k+1)
        Design matrix including an intercept column.  If ``varnames`` is
        provided, it should contain ``k+1`` names corresponding to the
        columns of ``X``.
    robust : bool, default True
        Use HC3 robust covariance estimates.  If False, classical OLS
        standard errors are used.
    varnames : list of str, optional
        Column names for ``X``.  If provided and valid, these names are used
        in the statsmodels regression and stored in the returned dictionary.

    Returns
    -------
    dict
        Contains coefficient arrays, standard errors, t‑values, p‑values, R²,
        adjusted R², residuals, the fitted statsmodels results object, and
        the variable names used.  If ``varnames`` is ``None`` or mismatched,
        parameter names default to ``const``, ``x1``, ``x2``, etc.
    """
    if not _HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for the primary fit_ols implementation"
        )

    # Wrap exogenous matrix in DataFrame when names are provided
    if varnames is not None and len(varnames) == X.shape[1]:
        exog = pd.DataFrame(X, columns=varnames)
    else:
        exog = X
        varnames = None
    n, k_plus1 = X.shape
    k = k_plus1 - 1
    model = sm.OLS(y, exog)
    res = model.fit()
    # Apply robust covariance if requested
    if robust:
        try:
            use = res.get_robustcov_results(cov_type="HC3")
        except Exception:  # noqa: BLE001 - statsmodels cov_type failures are untyped
            use = res.get_robustcov_results(cov_type="HC1")
    else:
        use = res
    # Ensure statsmodels returns Series even when given raw ndarrays
    params = use.params
    bse = use.bse
    tvalues = use.tvalues
    pvalues = use.pvalues
    if isinstance(params, np.ndarray):
        if varnames is None:
            try:
                varnames = list(res.model.exog_names)
            except Exception:  # noqa: BLE001 - pragma: no cover - very unlikely branch
                varnames = [f"x{i}" for i in range(params.shape[0])]
        params = pd.Series(params, index=varnames)
        bse = pd.Series(np.asarray(bse), index=varnames)
        tvalues = pd.Series(np.asarray(tvalues), index=varnames)
        pvalues = pd.Series(np.asarray(pvalues), index=varnames)
    # Extract statistics
    adj_r2 = res.rsquared_adj
    resid = res.resid
    df_resid = n - k_plus1
    rse = np.sqrt((resid @ resid) / df_resid) if df_resid > 0 else np.nan
    F_stat = _safe_fvalue(res) if k > 0 else np.nan
    display_names = varnames or list(params.index)
    return {
        "coef": params,
        "se": bse,
        "t": tvalues,
        "p": pvalues,
        "r2": res.rsquared,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "rse": rse,
        "F": F_stat,
        "resid": resid,
        "varnames": varnames,
        "display_varnames": display_names,
        "param_lookup": {name: name for name in display_names},
        "sm_results": res,
    }


def _fit_ols_fallback(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool = True,
    varnames: list[str] | None = None,
) -> dict[str, Any]:
    """Minimal OLS routine used when :mod:`statsmodels` is unavailable.

    The implementation mirrors the API of :func:`fit_ols` closely enough for
    the debiasing pipeline, returning coefficient estimates, approximate
    standard errors, and residuals.  ``robust`` and ``varnames`` are accepted
    for signature compatibility but otherwise ignored.
    """

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("Design matrix must be two-dimensional")

    beta, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    resid = y_arr - X_arr @ beta
    n, k_plus1 = X_arr.shape
    df_resid = n - k_plus1

    XtX = X_arr.T @ X_arr
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:  # pseudo-inverse when design is singular
        XtX_inv = np.linalg.pinv(XtX)

    if df_resid > 0:
        sigma2 = float(resid @ resid) / df_resid
        cov = sigma2 * XtX_inv
        se = np.sqrt(np.diag(cov))
        rse = float(np.sqrt(sigma2))
    else:
        sigma2 = 0.0
        se = np.full(beta.shape, np.nan)
        rse = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = beta / se
    p_vals = np.full(beta.shape, np.nan)

    mean_y = float(y_arr.mean()) if n else 0.0
    ss_tot = float(np.sum((y_arr - mean_y) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else np.nan

    return {
        "coef": np.asarray(beta),
        "se": np.asarray(se),
        "t": np.asarray(t_vals),
        "p": np.asarray(p_vals),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "n": int(n),
        "k": int(k_plus1 - 1),
        "rse": rse,
        "F": np.nan,
        "resid": np.asarray(resid),
        "varnames": varnames,
        "display_varnames": varnames,
        "param_lookup": {name: name for name in (varnames or [])},
        "sm_results": None,
    }


def _safe_fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool,
    varnames: list[str] | None = None,
) -> dict[str, Any]:
    """Attempt statsmodels-backed fitting, falling back to NumPy OLS when needed."""

    try:
        return fit_ols(y, X, robust=robust, varnames=varnames)
    except ImportError:
        if robust:
            warnings.warn(
                "statsmodels is unavailable; falling back to non-robust OLS for debiasing diagnostics.",
                RuntimeWarning,
                stacklevel=2,
            )
        return _fit_ols_fallback(y, X, robust=False, varnames=varnames)


__all__ = ["_fit_ols_fallback", "_safe_fit_ols", "fit_ols"]
