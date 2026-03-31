"""
Forecast evaluation metrics and probabilistic scoring rules.

This module provides utilities to evaluate predictive trajectories and
probabilistic forecasts, with a focus on horizon-wise assessment across
multiple simulated or ensemble prediction paths.

The main public functions are:
- `type_s_error`: directional (sign) error across forecast horizons
- `type_m_error`: magnitude error on a logarithmic scale
- `scoring_rules`: probabilistic forecast scores including CRPS, DSS,
  and interval score

Additional internal helpers support confidence interval calculation and
data reshaping for evaluation workflows.

Typical usage
-------------
>>> error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
...     true_value=y_true,
...     estimate=preds,
...     return_ci=True,
... )
>>> steps, per_step_errs, means = type_m_error(
...     estimate=preds,
...     true_value=y_true,
... )
>>> scores = scoring_rules(
...     true_value=y_true,
...     estimate=preds,
... )

Notes
-----
- Forecast paths are typically expected to have shape `(N, T)`, where
  `N` is the number of simulated or ensemble trajectories and `T` is the
  forecast horizon.
- Confidence intervals for Type S error are computed with the Wilson
  score interval.
- CRPS is computed using `properscoring` when available, with a fallback
  implementation otherwise.
"""

from __future__ import annotations
from scipy.stats import norm
import numpy as np

from typing import Optional, Tuple, Union

import pandas as pd

__all__ = [
    "type_s_error",
    "type_m_error",
    "scoring_rules",
]

ArrayLike1D = Union[np.ndarray, pd.Series]
ArrayLike2D = Union[np.ndarray, pd.DataFrame]


def _sign_with_tolerance(a: np.ndarray, eps: float) -> np.ndarray:
    """
    Return sign with tolerance:
      +1 if a > eps
       0 if |a| <= eps
      -1 if a < -eps
    """
    out = np.zeros_like(a, dtype=int)
    out[a > eps] = 1
    out[a < -eps] = -1
    return out


def wilson_score_interval(
    error_rate: np.ndarray,
    trials: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Wilson score interval for binomial proportions.

    Parameters
    ----------
    error_rate : np.ndarray
        Estimated proportions, shape (T,)
    trials : np.ndarray
        Number of valid trials for each time step, shape (T,)
    confidence_level : float, default=0.95

    Returns
    -------
    lower_bound, upper_bound : tuple[np.ndarray, np.ndarray]
        Wilson lower and upper bounds, shape (T,)
    """
    error_rate = np.asarray(error_rate, dtype=float)
    trials = np.asarray(trials, dtype=float)

    lower = np.full_like(error_rate, np.nan, dtype=float)
    upper = np.full_like(error_rate, np.nan, dtype=float)

    valid = (trials > 0) & np.isfinite(error_rate)
    if not np.any(valid):
        return lower, upper

    z = norm.ppf(1 - (1 - confidence_level) / 2)
    n = trials[valid]
    p = error_rate[valid]
    z2 = z**2

    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom

    lower[valid] = center - margin
    upper[valid] = center + margin
    return lower, upper


def type_s_error(
    true_value,
    estimate,
    baseline: str = "diff",
    eps: float = 1e-10,
    count_zero_pred_as_error: bool = True,
    return_ci: bool = False,
    alpha: float = 0.05,
    x=None,
):
    """
    Compute horizon-wise Type S (sign) error across prediction paths.

    The Type S error measures the probability that the predicted effect has
    the wrong sign compared to the true effect at each time step. The function
    operates across multiple prediction paths and returns a time series of
    error rates.

    Parameters
    ----------
    true_value : array-like of shape (T,)
        Ground-truth values. Can include a baseline value (e.g., last training
        observation) if evaluating forecast transitions.
    estimate : array-like of shape (N, T)
        Predicted paths aligned with `true_value`, where N is the number of
        simulated or predicted trajectories and T is the number of time steps.
    baseline : {'diff', 'level'}, default='diff'
        Defines how effects are computed:
        - 'diff': uses first differences y[t] - y[t-1]
        - 'level': uses deviation from initial value y[t] - y[0]
    eps : float, default=1e-10
        Tolerance for determining zero-valued effects. Values with absolute
        magnitude <= eps are treated as having zero sign.
    count_zero_pred_as_error : bool, default=True
        Controls how predicted zero effects are treated:
        - True: predicted zero counts as an error if true sign is non-zero
        - False: predicted zero is excluded from both numerator and denominator
    return_ci : bool, default=False
        If True, compute and return Wilson score confidence intervals for the
        error rate at each time step.
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 corresponds
        to a 95% confidence interval).
    x : array-like, optional
        Custom x-axis values corresponding to each horizon (length T-1). If not
        provided, a default integer range [1, ..., T-1] is used.

    Returns
    -------
    error_rate : np.ndarray of shape (T-1,)
        Fraction of prediction paths with incorrect sign at each time step.
    ci_low : np.ndarray or None
        Lower bound of the Wilson confidence interval for each time step.
        Returned only if `return_ci=True`, otherwise None.
    ci_high : np.ndarray or None
        Upper bound of the Wilson confidence interval for each time step.
        Returned only if `return_ci=True`, otherwise None.
    x_out : np.ndarray of shape (T-1,)
        X-axis values corresponding to each horizon.
    n_valid : np.ndarray of shape (T-1,)
        Number of valid prediction paths used at each time step (i.e., excluding
        NaNs or filtered zero predictions).

    Notes
    -----
    - Horizons where the true effect has zero sign are treated as undefined and
      excluded from the computation (resulting in NaN values).
    - When `count_zero_pred_as_error=False`, the effective sample size may vary
      across horizons due to excluded zero-sign predictions.
    - Confidence intervals are computed using the Wilson score interval.

    Examples
    --------
    >>> error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
    ...     true_value=y_gt_ext,
    ...     estimate=preds_ext,
    ...     baseline="diff",
    ...     return_ci=True,
    ... )
    """
    
    y = np.asarray(true_value, dtype=float).ravel()
    P = np.asarray(estimate, dtype=float)

    if P.ndim != 2:
        raise ValueError("estimate must be a 2D array with shape (N, T)")
    if P.shape[1] != y.shape[0]:
        raise ValueError(
            f"Shape mismatch: estimate has T={P.shape[1]} but true_value has T={y.shape[0]}"
        )
    if baseline not in {"diff", "level"}:
        raise ValueError("baseline must be either 'diff' or 'level'")
    if y.shape[0] < 2:
        raise ValueError("Need at least two time points")

    def sign_with_tolerance(a, eps):
        out = np.zeros_like(a, dtype=int)
        out[a > eps] = 1
        out[a < -eps] = -1
        return out

    if baseline == "diff":
        true_effects = y[1:] - y[:-1]
        pred_effects = P[:, 1:] - P[:, :-1]
    else:
        true_effects = y[1:] - y[0]
        pred_effects = P[:, 1:] - P[:, [0]]

    true_signs = sign_with_tolerance(true_effects, eps)
    pred_signs = sign_with_tolerance(pred_effects, eps)

    valid_horizon = true_signs != 0
    mismatches = (pred_signs != true_signs[None, :]).astype(float)

    if not count_zero_pred_as_error:
        mismatches[pred_signs == 0] = np.nan

    mismatches[:, ~valid_horizon] = np.nan

    num_errors = np.nansum(mismatches, axis=0)
    n_valid = np.sum(~np.isnan(mismatches), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_rate = num_errors / n_valid

    if x is not None:
        x_out = np.asarray(x)
        if x_out.shape[0] != error_rate.shape[0]:
            raise ValueError(
                f"x must have length {error_rate.shape[0]}, got {x_out.shape[0]}"
            )
    else:
        x_out = np.arange(1, y.shape[0])

    if not return_ci:
        return error_rate, None, None, x_out, n_valid

    ci_low, ci_high = wilson_score_interval(
        error_rate=error_rate,
        trials=n_valid,
        confidence_level=1 - alpha,
    )
    return error_rate, ci_low, ci_high, x_out, n_valid


# Functions for type m error
# -----------------------------
# Helpers to prepare the data
# -----------------------------
def build_predictions_df(preds):
    """
    Convert prediction paths into a long-format pandas DataFrame.
    
    This utility reshapes a collection of prediction trajectories into a
    DataFrame with one row per predicted value and two columns:
    `step`, indicating the forecast horizon, and `count`, containing the
    predicted value.
    
    Parameters
    ----------
    preds : array-like of shape (n_paths, n_steps) or iterable of array-like
        Collection of prediction paths. Accepted inputs are:
        - a 2D NumPy array with shape `(n_paths, n_steps)`, or
        - a list/tuple of 1D arrays of equal length
    
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - `step`: forecast horizon index, ranging from 1 to `n_steps`
        - `count`: predicted value at that horizon
    
    Raises
    ------
    ValueError
        If `preds` cannot be interpreted as a 2D collection of prediction
        paths.
    
    Examples
    --------
    >>> preds = np.array([[1.0, 2.0, 3.0],
    ...                   [1.5, 2.5, 3.5]])
    >>> build_predictions_df(preds)
       step  count
    0     1    1.0
    1     2    2.0
    2     3    3.0
    3     1    1.5
    4     2    2.5
    5     3    3.5
    """
    if isinstance(preds, (list, tuple)):
        preds = np.stack([np.asarray(p) for p in preds], axis=0)  # (n_paths, n_steps)
    preds = np.asarray(preds)
    if preds.ndim != 2:
        raise ValueError("`preds` must be 2D: (n_paths, n_steps) or list of 1D arrays.")

    n_paths, n_steps = preds.shape
    # Create step index repeated for each path
    steps = np.tile(np.arange(1, n_steps + 1), n_paths)
    counts = preds.reshape(-1)
    return pd.DataFrame({"step": steps, "count": counts})

# Function for type m error
def type_m_error(
    estimate: np.ndarray,
    true_value: np.ndarray,
    threshold: float = 1e-1,
    base: float = 10
):
    """
    Compute horizon-wise Type M (magnitude) error across prediction paths.
    
    The Type M error measures the multiplicative deviation between predicted
    and observed values at each forecast step using a logarithmic scale. It
    quantifies how much predictions over- or under-estimate the true values
    in terms of magnitude.
    
        Type M = | log_base(prediction / observed) |
    
    Parameters
    ----------
    estimate : array-like of shape (N, T)
        Predicted paths aligned with `true_value`, where:
        - N = number of simulated or predicted trajectories
        - T = number of time steps (forecast horizon)
    true_value : array-like of shape (T,)
        Ground-truth values corresponding to each forecast step.
    threshold : float, default=1e-1
        Minimum value used to replace small or zero values in both predictions
        and observations before computing the ratio. This avoids division by
        zero and extreme log-ratio values.
    base : float, default=10
        Base of the logarithm used in the error computation.
    
    Returns
    -------
    steps : np.ndarray of shape (T,)
        Forecast horizons (1, ..., T).
    per_step_errs : list of np.ndarray
        List where each element contains the Type M errors for all prediction
        paths at a given forecast step.
    means : np.ndarray of shape (T,)
        Mean Type M error at each forecast horizon.
    
    Notes
    -----
    - The error is computed as the absolute logarithmic ratio between predicted
      and observed values, so it measures multiplicative deviation.
    - A value of 0 indicates perfect agreement, while larger values indicate
      stronger over- or under-estimation.
    - The threshold parameter prevents instability when values are close to zero.
    - Unlike `type_s_error`, this function does not compute confidence intervals.
    
    Examples
    --------
    >>> steps, per_step_errs, means = type_m_error(
    ...     estimate=preds,
    ...     true_value=y_true,
    ... )
    >>> means
    array([...])
    """

    log_fn = np.log10 if base == 10 else (lambda x: np.log(x) / np.log(base))

    predictions_df = build_predictions_df(estimate)
    true_value_sr = pd.Series(true_value)

    steps = np.sort(predictions_df["step"].unique())
    per_step_errs = []
    means = []

    for step in steps:
        subset = predictions_df.loc[
            predictions_df["step"] == step, "count"
        ].to_numpy(copy=True)

        y_obs = true_value_sr.iloc[step - 1]
        y_obs = max(y_obs, threshold)

        subset = np.maximum(subset, threshold)

        errs = np.abs(log_fn(subset / y_obs))
        per_step_errs.append(errs)
        means.append(errs.mean())

    return steps, per_step_errs, np.asarray(means)
    
# Function for scoring rules calculation:
# --- Fast CRPS from ensemble samples (tries properscoring, else a vectorized fallback) ---
try:
    from properscoring import crps_ensemble as _crps_ensemble

    def _crps(y, samples):
        return float(_crps_ensemble(y, samples))
except Exception:
    def _crps(y, samples):
        # CRPS = E|X - y| - 0.5 E|X - X'|
        s = np.asarray(samples, dtype=float)
        n = s.size
        sx = np.mean(np.abs(s - y))

        # Efficient computation of E|X - X'| using order stats
        s_sorted = np.sort(s)
        i = np.arange(1, n + 1)
        exx = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * s_sorted)

        return sx - 0.5 * exx


def scoring_rules(true_value: np.ndarray,
                  estimate: np.ndarray,
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Calculate probabilistic scoring rules for ensemble forecasts.
    
    This function evaluates predictive samples against observed values at
    each forecast horizon using three standard probabilistic scores:
    
    - CRPS (Continuous Ranked Probability Score), which measures the overall
      agreement between the predictive distribution and the observation
    - DSS (Dawid-Sebastiani Score), which evaluates sharpness and calibration
      using the predictive mean and variance
    - Interval Score (Winkler score), which evaluates the width and coverage
      of a central prediction interval
    
    Parameters
    ----------
    true_value : array-like of shape (T,)
        Observed values for the forecast horizon.
    estimate : array-like of shape (N, T)
        Predictive samples or ensemble members, where:
        - N = number of simulated paths / ensemble members
        - T = forecast horizon
    alpha : float, default=0.05
        Miscoverage level for the central prediction interval.
        For example, `alpha=0.05` corresponds to a 95% prediction interval.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by forecast step with columns:
        - `CRPS`
        - `DSS`
        - `IntervalScore`
    
    Notes
    -----
    - `estimate` must be a 2D array with one row per simulated path and one
      column per forecast step.
    - If `true_value` is longer than the forecast horizon, only the first `T`
      values are used.
    - The DSS uses the sample variance with `ddof=1` and protects against
      zero variance using a small positive epsilon.
    - The Interval Score is computed for the central `(1 - alpha)` prediction
      interval.
    
    Raises
    ------
    ValueError
        If `estimate` is not a 2D array.
    ValueError
        If `alpha` is not strictly between 0 and 1.
    ValueError
        If `true_value` contains fewer than `T` values.
    
    Examples
    --------
    >>> scores = scoring_rules(
    ...     true_value=y_true,
    ...     estimate=preds,
    ...     alpha=0.05,
    ... )
    >>> scores.head()
    """

    # Check entry conditions
    estimate = np.asarray(estimate, dtype=float)
    true_value = np.asarray(true_value, dtype=float).ravel()

    if estimate.ndim != 2:
        raise ValueError("estimate must be a 2D array of shape (N, T).")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    n_paths, horizon = estimate.shape

    if true_value.size < horizon:
        raise ValueError(
            f"true_value must contain at least {horizon} values, got {true_value.size}."
        )

    true_value = true_value[:horizon]

    eps = 1e-12
    scores = []

    # Calculate scores for each forecast step
    for t in range(horizon):
        samples_t = estimate[:, t]
        y_t = true_value[t]

        # CRPS
        crps_value = _crps(y_t, samples_t)

        # Dawid-Sebastiani Score
        mu_t = np.mean(samples_t)
        sigma2_t = np.var(samples_t, ddof=1)
        sigma2_t = max(sigma2_t, eps)
        dss_value = ((y_t - mu_t) ** 2) / sigma2_t + np.log(sigma2_t)

        # Interval Score (Winkler)
        lower_t = np.quantile(samples_t, alpha / 2)
        upper_t = np.quantile(samples_t, 1 - alpha / 2)

        interval_score = upper_t - lower_t
        if y_t < lower_t:
            interval_score += (2 / alpha) * (lower_t - y_t)
        elif y_t > upper_t:
            interval_score += (2 / alpha) * (y_t - upper_t)

        scores.append({
            "step": t + 1,
            "CRPS": crps_value,
            "DSS": dss_value,
            "IntervalScore": interval_score
        })

    return pd.DataFrame(scores).set_index("step")