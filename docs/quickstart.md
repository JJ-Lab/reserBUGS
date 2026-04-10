# Quick Start

This example follows the same general workflow as `notebooks/Example_biotime.ipynb`:

1. Load a time series dataset.
2. Split the series chronologically into training and testing sets.
3. Fit a reservoir computing model.
4. Generate forecast paths.
5. Evaluate forecast performance.

```python
import numpy as np
import pandas as pd

from reserbugs.reservoir_computing import ReservoirComputing
from reserbugs.evaluation import scoring_rules, type_m_error, type_s_error

data = pd.read_csv("your_timeseries.csv")

X = data[["temperature", "precipitation"]].values
y = data["count"].values

split_idx = int(0.8 * len(y))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

rc = ReservoirComputing()
rc.fit(X_train, y_train)

preds, stats = rc.sample_paths(
    X_train=X_train,
    Y_train=y_train,
    X_test=X_test,
    Y_init=y_train[-1:],
    n_lags=1,
    N=100,
)

last_y = float(y_train[-1])
y_gt_ext = np.concatenate(([last_y], y_test))
preds_ext = np.concatenate(
    (np.full((preds.shape[0], 1), last_y), preds),
    axis=1,
)

error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
    true_value=y_gt_ext,
    estimate=preds_ext,
    baseline="diff",
    return_ci=True,
)

steps, per_step_errs, means = type_m_error(
    estimate=preds,
    true_value=y_test,
    threshold=0.1,
    base=10,
)

scores = scoring_rules(
    true_value=y_test[: preds.shape[1]],
    estimate=preds,
    alpha=0.05,
)

print(scores.head())
```

## Multiple Lags

For workflows with more than one lag, pass the same lag count when fitting and
predicting:

```python
rc.fit(X_train, y_train, n_lags=3)
preds = rc.predict(X_test, Y_init=y_train[-3:], n_lags=3)
```
