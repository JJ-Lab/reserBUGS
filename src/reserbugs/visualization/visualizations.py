import plotly.graph_objects as go
import numpy as np
import pandas as pd


from ..evaluation import type_s_error

"""
Plotting utilities for forecast evaluation and visualization.

Includes functions to visualize:
- forecast paths against observed data
- Type S error over forecast horizons
- Type M error distributions
- probabilistic scoring rules
"""

def plot_prediction_vs_ground_truth(
    y_test_predicted,
    y_train,
    y_test=None,
    band=80,
    show_paths=False,
    path_alpha=0.35,
    median_color='red',
    gt_color='black'
) -> go.Figure:
    """
    Plot training data, predictive median, confidence band, and optional ground truth.
    
    Parameters
    ----------
    y_test_predicted : array-like, shape (N_sim, horizon)
        Simulated forecast paths.
    y_train : pd.Series
        Training target series.
    y_test : pd.Series, optional
        Test target series.
    band : int, default=80
        Central prediction interval width.
    show_paths : bool, default=False
        Whether to show individual simulated paths.
    path_alpha : float, default=0.35
        Opacity of simulated paths.
    median_color : str, default='red'
        Color of median forecast line.
    gt_color : str, default='black'
        Color of ground truth line.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure showing the training series, predictive median,
        prediction interval, optional simulated paths, and optional
        ground-truth values.
    
    Notes
    -----
    - The forecast x-axis is constructed by extending the last index value of
      `y_train`, so the index must support addition with integer offsets.
    - The last training observation is prepended to each forecast path to ensure
      continuity between training and forecast segments.

    Examples
    --------
    >>> fig = plot_prediction_vs_ground_truth(preds, y_train, y_test)
    >>> fig.show()
    """

    y_test_predicted = np.asarray(y_test_predicted, dtype=float)

    # Ensure shape = (N_simulations, horizon)
    if y_test_predicted.ndim != 2:
        raise ValueError(f"y_test_predicted must be 2D, got shape {y_test_predicted.shape}")

    horizon = y_test_predicted.shape[1]

    last_train_index = y_train.index[-1]
    last_y_train = float(y_train.iloc[-1])

    # x positions for forecast section: include last training point for continuity
    x_forecast = np.concatenate((
        [last_train_index],
        last_train_index + np.arange(1, horizon + 1)
    ))

    # prepend last observed training value to every simulated path
    preds_extended = np.concatenate(
        [np.full((y_test_predicted.shape[0], 1), last_y_train), y_test_predicted],
        axis=1
    )

    # interval summaries
    lower_percentile = (100 - band) / 2
    upper_percentile = 100 - lower_percentile

    median = np.nanmedian(preds_extended, axis=0)
    q_lo, q_hi = np.nanpercentile(
        preds_extended,
        [lower_percentile, upper_percentile],
        axis=0
    )

    fig = go.Figure()

    # training series
    fig.add_trace(go.Scatter(
        x=y_train.index,
        y=y_train.values,
        mode='lines',
        name='Training Data',
        line=dict(color='blue')
    ))

    # optional individual paths
    if show_paths:
        for i in range(preds_extended.shape[0]):
            fig.add_trace(go.Scatter(
                x=x_forecast,
                y=preds_extended[i, :],
                mode='lines',
                line=dict(color='grey', width=1),
                opacity=path_alpha,
                showlegend=False
            ))

    # confidence band as closed polygon
    band_x = np.concatenate([x_forecast, x_forecast[::-1]])
    band_y = np.concatenate([q_hi, q_lo[::-1]])

    fig.add_trace(go.Scatter(
        x=band_x,
        y=band_y,
        fill='toself',
        fillcolor='rgba(160,160,160,0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name=f'{band}% Confidence Band'
    ))

    # median prediction
    fig.add_trace(go.Scatter(
        x=x_forecast,
        y=median,
        mode='lines',
        line=dict(color=median_color),
        name='Median Prediction'
    ))

    # ground truth, connected to last training point
    if y_test is not None:
        x_gt = np.concatenate(([last_train_index], y_test.index.to_numpy()))
        y_gt = np.concatenate(([last_y_train], y_test.to_numpy()))

        fig.add_trace(go.Scatter(
            x=x_gt,
            y=y_gt,
            mode='lines',
            line=dict(color=gt_color),
            name='Ground Truth'
        ))

    return fig


def plot_type_s_errors(
    true_value,
    estimates,
    baseline: str = "diff",
    eps: float = 1e-10,
    count_zero_pred_as_error: bool = True,
    alpha: float = 0.05,
    show_ci: bool = True,
    title: str = "Type S Error",
    line_color: str = "red",
    ci_color: str = "rgba(255, 0, 0, 0.18)",
    benchmark: float | None = 0.5,
    benchmark_color: str = "gray",
    x=None,
) -> go.Figure:
    """
    Plot horizon-wise Type S (sign) error using Plotly.

    This function computes the Type S error across prediction paths and
    visualizes it as a time series, optionally including Wilson confidence
    intervals and a benchmark reference line.

    Parameters
    ----------
    true_value : array-like of shape (T,)
        Ground-truth values aligned with `estimates`. Can include a baseline
        value (e.g., last training observation) if evaluating forecast
        transitions.
    estimates : array-like of shape (N, T)
        Predicted paths aligned with `true_value`, where N is the number of
        trajectories and T is the number of time steps.
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
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 corresponds
        to a 95% confidence interval).
    show_ci : bool, default=True
        Whether to display Wilson confidence intervals as shaded bands.
    title : str, default='Type S Error'
        Title of the plot.
    line_color : str, default='red'
        Color of the Type S error line.
    ci_color : str, default='rgba(255, 0, 0, 0.18)'
        Fill color of the confidence interval bands.
    benchmark : float or None, default=0.5
        Optional horizontal reference line (e.g., 0.5 for random sign).
        If None, no benchmark line is drawn.
    benchmark_color : str, default='gray'
        Color of the benchmark reference line.
    x : array-like, optional
        Custom x-axis values corresponding to each horizon (length T-1).
        If not provided, a default integer range [1, ..., T-1] is used.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure containing the Type S error curve, confidence intervals,
        and optional benchmark.

    Notes
    -----
    - Confidence intervals are computed using the Wilson score interval.
    - The function handles missing values (NaNs) by splitting the plot into
      contiguous segments to avoid incorrect visual connections.
    - The shaded confidence interval is rendered as independent polygons per
      valid segment to ensure correct visualization when gaps are present.
    - Hover information includes the error rate, confidence interval bounds,
      and the number of valid prediction paths (`n_valid`) at each horizon.

    Examples
    --------
    >>> fig = plot_type_s_errors(
    ...     true_value=y_gt_ext,
    ...     estimates=preds_ext,
    ...     baseline="diff",
    ...     count_zero_pred_as_error=False,
    ...     x=y_test.index.values,
    ... )
    >>> fig.show()
    """
    error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
        true_value=true_value,
        estimate=estimates,
        baseline=baseline,
        eps=eps,
        count_zero_pred_as_error=count_zero_pred_as_error,
        return_ci=show_ci,
        alpha=alpha,
        x=x,
    )

    x_out = np.asarray(x_out)
    error_rate = np.asarray(error_rate, dtype=float)

    if show_ci and ci_low is not None and ci_high is not None:
        ci_low = np.asarray(ci_low, dtype=float)
        ci_high = np.asarray(ci_high, dtype=float)

    n_valid = np.asarray(n_valid)

    fig = go.Figure()

    if show_ci and ci_low is not None and ci_high is not None:
        valid_ci = np.isfinite(x_out) & np.isfinite(ci_low) & np.isfinite(ci_high)
        valid_idx = np.flatnonzero(valid_ci)

        if valid_idx.size > 0:
            split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
            blocks = np.split(valid_idx, split_points)

            first_band = True
            for block in blocks:
                xb = x_out[block]
                lowb = ci_low[block]
                highb = ci_high[block]

                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([xb, xb[::-1]]),
                        y=np.concatenate([highb, lowb[::-1]]),
                        mode="lines",
                        fill="toself",
                        fillcolor=ci_color,
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        hoverinfo="skip",
                        name=f"{int((1 - alpha) * 100)}% Wilson CI",
                        legendgroup="type_s_ci",
                        showlegend=first_band,
                    )
                )
                first_band = False

    valid_line = np.isfinite(x_out) & np.isfinite(error_rate)
    valid_idx = np.flatnonzero(valid_line)

    if valid_idx.size > 0:
        split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
        blocks = np.split(valid_idx, split_points)

        first_line = True
        for block in blocks:
            xb = x_out[block]
            yb = error_rate[block]
            lowb = ci_low[block] if show_ci and ci_low is not None else np.full_like(yb, np.nan)
            highb = ci_high[block] if show_ci and ci_high is not None else np.full_like(yb, np.nan)
            nb = n_valid[block]

            customdata = np.column_stack([lowb, highb, nb])

            fig.add_trace(
                go.Scatter(
                    x=xb,
                    y=yb,
                    mode="lines",
                    line=dict(color=line_color, width=3),
                    name="Type S Error",
                    legendgroup="type_s_error",
                    showlegend=first_line,
                    customdata=customdata,
                    hovertemplate=(
                        "x=%{x}<br>"
                        "Type S error=%{y:.6f}<br>"
                        "CI low=%{customdata[0]:.6f}<br>"
                        "CI high=%{customdata[1]:.6f}<br>"
                        "n_valid=%{customdata[2]}<extra></extra>"
                    ),
                )
            )
            first_line = False

    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=x_out,
                y=np.full(x_out.shape[0], benchmark, dtype=float),
                mode="lines",
                line=dict(color=benchmark_color, dash="dot", width=1.5),
                name=f"Benchmark = {benchmark}",
                hovertemplate="Benchmark=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Time-step",
        yaxis_title="Probability of wrong sign",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(groupclick="togglegroup"),
    )

    return fig


def plot_type_s_errors_from_forecast(
    y_train,
    y_test,
    preds,
    baseline: str = "diff",
    eps: float = 1e-10,
    count_zero_pred_as_error: bool = True,
    alpha: float = 0.05,
    show_ci: bool = True,
    title: str = "Type S Error",
):
    """
    Plot Type S error for forecast paths using train/test split data.

    This is a convenience wrapper around `plot_type_s_errors` that prepares
    extended ground-truth and prediction arrays by including the last training
    value as the initial reference point. This allows evaluation of the first
    forecast step relative to the end of the training period.

    Parameters
    ----------
    y_train : pandas.Series
        Training time series. Only the last value is used as the reference
        baseline for computing forecast transitions.
    y_test : pandas.Series
        Ground-truth values for the forecast horizon.
    preds : array-like of shape (N, T)
        Predicted forecast paths, where N is the number of trajectories and
        T is the forecast horizon length. Must be aligned with `y_test`.
    baseline : {'diff', 'level'}, default='diff'
        Defines how effects are computed:
        - 'diff': uses first differences y[t] - y[t-1]
        - 'level': uses deviation from the initial value
    eps : float, default=1e-10
        Tolerance for determining zero-valued effects.
    count_zero_pred_as_error : bool, default=True
        Controls how predicted zero effects are treated:
        - True: predicted zero counts as an error if true sign is non-zero
        - False: predicted zero is excluded from computation
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    show_ci : bool, default=True
        Whether to display Wilson confidence intervals.
    title : str, default='Type S Error'
        Title of the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure containing the Type S error curve and optional
        confidence intervals.

    Notes
    -----
    - The function prepends the last training value to both `y_test` and
      each prediction path in `preds` to ensure proper alignment for
      computing first-step effects.
    - The x-axis is derived from `y_test.index`, so `y_test` must be a
      pandas Series with a meaningful index (e.g., time steps).
    - This function is intended for forecasting workflows where predictions
      start immediately after the training period.

    Examples
    --------
    >>> fig = plot_type_s_errors_from_forecast(
    ...     y_train=y_train,
    ...     y_test=y_test,
    ...     preds=preds,
    ...     baseline="diff",
    ...     count_zero_pred_as_error=False,
    ... )
    >>> fig.show()
    """
    last_y = float(y_train.iloc[-1])
    y_gt_ext = np.concatenate(([last_y], y_test.values))
    preds_ext = np.concatenate((np.full((preds.shape[0], 1), last_y), preds), axis=1)

    return plot_type_s_errors(
        true_value=y_gt_ext,
        estimates=preds_ext,
        baseline=baseline,
        eps=eps,
        count_zero_pred_as_error=count_zero_pred_as_error,
        alpha=alpha,
        show_ci=show_ci,
        title=title,
        x=y_test.index.values,
    )


# Plot type M errors with violin plots (predictions_df, y_test,eps=1, base=10)
def plot_type_m_errors(true_value: pd.Series, estimate: np.ndarray, steps, per_step_errs, means,  base=10, title="Type M Error by Forecast Step"):
    
    """
    Plot Type M error distributions across forecast horizons using violin plots.

    Parameters
    ----------
    true_value : pandas.Series
        Observed values used to map forecast steps to x-axis labels.
    estimate : np.ndarray
        Predicted values. Currently unused in the plotting logic and kept for
        API compatibility.
    steps : np.ndarray
        Forecast horizons, typically returned by `type_m_error`.
    per_step_errs : list of np.ndarray
        Type M errors for each forecast step.
    means : np.ndarray
        Mean Type M error at each forecast step.
    base : float, default=10
        Base of the logarithm used in the Type M error computation.
    title : str, default="Type M Error by Forecast Step"
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly violin plot figure with one violin per forecast step and
        an overlay of the mean Type M error.

    Notes
    -----
    - The x-axis labels are taken from `true_value.index`.
    - `steps` is assumed to be 1-based.
    """

    # Map numeric steps to ground_truth index labels
    step_labels = true_value.index.values[steps - 1]  # steps are 1-based

    # Fijar tamaño final de la figura en píxeles (ancho x alto)
    fig = go.Figure(layout=dict(width=900, height=500))

    # Add one violin per step, but use ground_truth index labels
    for _, label, errs in zip(steps, step_labels, per_step_errs):
        fig.add_trace(go.Violin(
            y=errs,
            x=[label] * len(errs),   # use ground_truth index value instead of step number
            name=str(label),
            box_visible=True,
            meanline_visible=False,
            points=False,  # remove individual points if desired
            opacity=0.6,
            fillcolor="lightblue",   # <-- unified fill color
            line_color="blue",        # <-- unified border color
            showlegend=False
        ))

    # Add mean points overlay (aligned to ground_truth index values)
    fig.add_trace(go.Scatter(
        x=step_labels,
        y=means,
        mode="markers+lines",
        name="Mean Type M error",
        marker=dict(color="red", size=8, symbol="circle"),
        line=dict(dash="dash", color="red"),
        showlegend=False   # <-- hide this from the legend
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time-step",
        yaxis_title=f"Type M error (|log base {base}|)",
        violingap=0.3,
        violinmode="overlay",
        template="plotly_white"
    )

    return fig

# Plot scoring rules
def plot_scoring_rules(scores: pd.DataFrame) -> go.Figure:
    """
    Visualize probabilistic forecast scores over forecast steps.
    
    Parameters
    ----------
    scores : pandas.DataFrame
        DataFrame indexed by forecast step, typically returned by
        `scoring_rules`. Numeric columns are plotted as separate lines.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure showing one line per score across forecast steps.
    
    Notes
    -----
    - Only numeric columns are retained.
    - The DataFrame index is used as the x-axis and is sorted before plotting.
    """

    # --- Validation ---
    if not isinstance(scores, pd.DataFrame):
        raise ValueError("scores must be a pandas DataFrame")

    if scores.empty:
        raise ValueError("scores DataFrame is empty")

    # Keep only numeric columns
    scores = scores.select_dtypes(include=[np.number])

    if scores.shape[1] == 0:
        raise ValueError("scores must contain at least one numeric column")

    # Ensure ordered steps
    scores = scores.sort_index()

    fig = go.Figure()

    # Add one line per score
    for col in scores.columns:
        fig.add_trace(
            go.Scatter(
                x=scores.index.values,
                y=scores[col].values,
                mode="lines+markers",
                name=col,
                line=dict(width=2)
            )
        )

    # Layout
    fig.update_layout(
        title="Probabilistic Forecast Scores over Time",
        xaxis_title="Forecast step",
        yaxis_title="Score",
        template="plotly_white",
        legend=dict(title="Score"),
        hovermode="x unified"
    )

    return fig