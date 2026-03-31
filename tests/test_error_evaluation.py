import numpy as np
import pandas as pd
import pytest

from reserbugs.evaluation.error_evaluation import (
    type_s_error,
    type_m_error,
    scoring_rules,
)


def test_type_s_error_basic_case():
    true_value = np.array([10.0, 12.0, 15.0, 14.0])

    estimate = np.array([
        [10.0, 11.0, 16.0, 13.0],
        [10.0, 13.0, 14.0, 15.0],
        [10.0,  9.0, 13.0, 12.0],
    ])

    error_rate, ci_low, ci_high, x_out, n_valid = type_s_error(
        true_value=true_value,
        estimate=estimate,
        baseline="diff",
        return_ci=True,
    )

    assert error_rate.shape == (3,)
    assert ci_low.shape == (3,)
    assert ci_high.shape == (3,)
    assert x_out.shape == (3,)
    assert n_valid.shape == (3,)


def test_type_m_error_returns_expected_shapes():
    true_value = np.array([10.0, 20.0, 40.0])

    estimate = np.array([
        [10.0, 18.0, 35.0],
        [12.0, 22.0, 45.0],
        [ 9.0, 19.0, 50.0],
    ])

    steps, per_step_errs, means = type_m_error(
        estimate=estimate,
        true_value=true_value,
        threshold=0.1,
        base=10,
    )

    assert np.array_equal(steps, np.array([1, 2, 3]))
    assert len(per_step_errs) == 3
    assert means.shape == (3,)


def test_scoring_rules_returns_dataframe():
    true_value = np.array([10.0, 12.0, 14.0])

    estimate = np.array([
        [ 9.5, 11.0, 13.5],
        [10.5, 12.5, 14.5],
        [10.0, 12.0, 14.0],
        [ 9.8, 11.8, 14.2],
    ])

    scores = scoring_rules(
        true_value=true_value,
        estimate=estimate,
        alpha=0.05,
    )

    assert isinstance(scores, pd.DataFrame)
    assert list(scores.columns) == ["CRPS", "DSS", "IntervalScore"]
    assert list(scores.index) == [1, 2, 3]


def test_scoring_rules_raises_for_invalid_alpha():
    true_value = np.array([1.0, 2.0, 3.0])
    estimate = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
    ])

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        scoring_rules(true_value=true_value, estimate=estimate, alpha=1.0)

def test_type_s_error_raises_on_wrong_shape():
    import numpy as np
    from reserbugs.evaluation.error_evaluation import type_s_error

    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.0, 2.0, 3.0])  # not 2D

    import pytest
    with pytest.raises(ValueError):
        type_s_error(true_value=y, estimate=preds)


def test_type_s_error_level_mode():
    import numpy as np
    from reserbugs.evaluation.error_evaluation import type_s_error

    y = np.array([10.0, 12.0, 15.0])
    preds = np.array([
        [10.0, 13.0, 16.0],
        [10.0, 11.0, 14.0],
    ])

    error_rate, *_ = type_s_error(
        true_value=y,
        estimate=preds,
        baseline="level",
    )

    assert error_rate.shape == (2,)
