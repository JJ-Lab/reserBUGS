import numpy as np
import pytest

from reserbugs.reservoir_computing.reservoir_computing import ReservoirComputing


def make_nonnegative_series(n=40):
    """
    Small deterministic dataset suitable for Poisson/Tweedie/Gamma tests.
    """
    x = np.linspace(0, 2 * np.pi, n)
    X = np.column_stack([
        np.sin(x) + 1.5,
        np.cos(x) + 1.5,
    ])
    y = 2.0 + 0.3 * np.sin(x) + 0.2 * np.cos(x)
    y = np.maximum(y, 0.1)  # strictly positive for gamma safety
    return X, y


def test_fit_creates_model():
    X, y = make_nonnegative_series()

    rc = ReservoirComputing(
        reservoir_size=50,
        seed=123,
        family="tweedie",
    )
    rc.fit(X[:30], y[:30], n_lags=1)

    assert rc.model is not None
    assert rc.scaler_X is not None


def test_predict_returns_expected_shape():
    X, y = make_nonnegative_series()

    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]

    rc = ReservoirComputing(
        reservoir_size=50,
        seed=123,
        family="tweedie",
    )
    rc.fit(X_train, y_train, n_lags=1)

    preds = rc.predict(X_test, Y_init=y_train[-1:], n_lags=1)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert np.all(np.isfinite(preds))


def test_predict_raises_when_y_init_is_too_short():
    X, y = make_nonnegative_series()

    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]

    rc = ReservoirComputing(
        reservoir_size=50,
        seed=123,
        family="tweedie",
    )
    rc.fit(X_train, y_train, n_lags=2)

    with pytest.raises(ValueError, match="Y_init must have at least 2 values"):
        rc.predict(X_test, Y_init=y_train[-1:], n_lags=2)


def test_gamma_fit_raises_on_nonpositive_targets():
    X, y = make_nonnegative_series()
    y_bad = y.copy()
    y_bad[5] = 0.0

    rc = ReservoirComputing(
        reservoir_size=30,
        seed=123,
        family="gamma",
    )

    with pytest.raises(ValueError, match="Gamma requires y > 0"):
        rc.fit(X[:30], y_bad[:30], n_lags=1)


def test_sample_paths_returns_expected_shapes_and_stats():
    X, y = make_nonnegative_series()

    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]

    rc = ReservoirComputing(
        reservoir_size=40,
        seed=123,
        family="tweedie",
    )

    preds, stats = rc.sample_paths(
        X_train=X_train,
        Y_train=y_train,
        X_test=X_test,
        Y_init=y_train[-1:],
        n_lags=1,
        N=5,
        base_seed=500,
        return_stats=True,
    )

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5, len(X_test))

    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "quantiles" in stats
    assert "seeds" in stats

    assert stats["mean"].shape == (len(X_test),)
    assert len(stats["seeds"]) == 5
    assert stats["seeds"] == [500, 501, 502, 503, 504]


def test_invalid_family_raises():
    X, y = make_nonnegative_series()

    rc = ReservoirComputing(
        reservoir_size=20,
        seed=123,
        family="invalid_family",
    )

    with pytest.raises(ValueError, match="family must be 'gamma', 'poisson' or 'tweedie'"):
        rc.fit(X[:30], y[:30], n_lags=1)


def test_sample_paths_reproducibility():
    X, y = make_nonnegative_series()

    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]

    rc = ReservoirComputing(seed=123, reservoir_size=30)

    preds1, _ = rc.sample_paths(
        X_train, y_train, X_test, y_train[-1:], N=3, base_seed=100
    )

    preds2, _ = rc.sample_paths(
        X_train, y_train, X_test, y_train[-1:], N=3, base_seed=100
    )

    assert np.allclose(preds1, preds2)


def test_sample_paths_parallel_matches_shape():
    X, y = make_nonnegative_series()

    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]

    rc = ReservoirComputing(seed=123, reservoir_size=30)

    preds, stats = rc.sample_paths_parallel(
        X_train=X_train,
        Y_train=y_train,
        X_test=X_test,
        Y_init=y_train[-1:],
        N=4,
        n_jobs=1,
        return_stats=True,
    )

    assert preds.shape == (4, len(X_test))
    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "quantiles" in stats
    assert "seeds" in stats