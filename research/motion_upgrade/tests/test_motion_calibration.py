"""
Calibration tests.

Written against synthetic logits with a known true temperature, so
"calibration works" is a recoverable quantity rather than an impression.
The key property tests: temperature scaling cannot change accuracy (so it
can never be blamed for a worse score), and it does reduce ECE on an
overconfident model.
"""
import numpy as np
import pytest

from motion.calibration import (
    apply_temperature,
    brier_score,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    softmax,
)


def make_logits(n=800, n_classes=4, true_temperature=2.5, seed=0):
    """
    Logits from a model that is right ~70% of the time but reports far
    higher confidence — the overconfident regime a small deep model on a
    small dataset actually lands in.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    correct = rng.random(n) < 0.7
    for i in range(n):
        target = labels[i] if correct[i] else (labels[i] + 1) % n_classes
        logits[i, target] += 4.0
    return logits * true_temperature, labels


def test_softmax_rows_sum_to_one_and_is_stable():
    probs = softmax(np.array([[1000.0, 1001.0], [-1000.0, -1001.0]]))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()


def test_fit_temperature_recovers_the_scaling():
    # The base logits are themselves overconfident, so the absolute
    # temperature is not 1.0 to begin with. The recoverable property is
    # that scaling the logits by k scales the fitted temperature by k:
    # calibration has to track how sharp the logits are, whatever their
    # starting point.
    base_logits, labels = make_logits(true_temperature=1.0, seed=1)
    base_t = fit_temperature(base_logits, labels)
    assert base_t > 1.0, "the synthetic model is deliberately overconfident"

    for k in (0.5, 2.0, 3.0):
        assert fit_temperature(base_logits * k, labels) == pytest.approx(base_t * k, rel=0.08)


def test_fitting_a_well_calibrated_model_leaves_it_alone():
    # Sanity floor: calibrate once, then calibrating the already-scaled
    # logits must return a temperature of about 1.
    logits, labels = make_logits(seed=11)
    t = fit_temperature(logits, labels)
    assert fit_temperature(logits / t, labels) == pytest.approx(1.0, rel=0.05)


def test_fitted_temperature_minimises_nll():
    logits, labels = make_logits(seed=2)
    t = fit_temperature(logits, labels)
    best = negative_log_likelihood(logits, labels, t)
    for other in (t * 0.5, t * 0.8, t * 1.25, t * 2.0):
        assert negative_log_likelihood(logits, labels, other) >= best - 1e-9


def test_temperature_scaling_cannot_change_predictions():
    # One scalar divided into every logit preserves the argmax, so
    # calibration can never cost accuracy — the reason to prefer it over
    # anything fancier on a small validation split.
    logits, _ = make_logits(seed=3)
    for t in (0.2, 0.7, 1.0, 4.0, 9.0):
        np.testing.assert_array_equal(
            apply_temperature(logits, t).argmax(axis=1), logits.argmax(axis=1)
        )


def test_calibration_reduces_ece_on_an_overconfident_model():
    val_logits, val_labels = make_logits(seed=4)
    test_logits, test_labels = make_logits(seed=5)

    report = calibrate(val_logits, val_labels, test_logits, test_labels)
    assert report.temperature > 1.0, "an overconfident model needs softening"
    assert report.ece_after < report.ece_before
    assert report.brier_after <= report.brier_before
    assert report.nll_after <= report.nll_before
    assert report.improved is True
    assert "temperature" in report.summary()


def test_calibration_report_serialises():
    val_logits, val_labels = make_logits(n=200, seed=6)
    d = calibrate(val_logits, val_labels).to_dict()
    assert set(["temperature", "ece_before", "ece_after", "bins_after"]).issubset(d)
    assert isinstance(d["bins_after"], list)


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # 200 samples at exactly 0.8 confidence, correct exactly 80% of the time.
    n = 200
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.8
    probs[:, 1] = 0.2
    labels = np.zeros(n, dtype=int)
    labels[int(n * 0.8):] = 1
    ece, bins = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert len(bins) == 1 and bins[0]["count"] == n


def test_ece_is_large_for_a_confidently_wrong_model():
    probs = np.tile([0.99, 0.01], (100, 1))
    labels = np.ones(100, dtype=int)  # always the class it gave 0.01 to
    ece, _ = expected_calibration_error(probs, labels)
    assert ece == pytest.approx(0.99, abs=0.01)
    assert maximum_calibration_error(probs, labels) == pytest.approx(0.99, abs=0.01)


def test_empty_bins_are_omitted_rather_than_counted_as_calibrated():
    probs = np.tile([0.9, 0.1], (50, 1))
    labels = np.zeros(50, dtype=int)
    _, bins = expected_calibration_error(probs, labels, n_bins=20)
    assert len(bins) == 1, "19 empty bins must not be reported as perfect"


def test_brier_score_bounds():
    onehot = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(onehot, labels) == pytest.approx(0.0)
    assert brier_score(onehot[:, ::-1], labels) == pytest.approx(2.0)


def test_reliability_curve_matches_the_bins():
    probs, labels = softmax(make_logits(seed=7)[0]), make_logits(seed=7)[1]
    conf, acc, count = reliability_curve(probs, labels, n_bins=10)
    assert conf.shape == acc.shape == count.shape
    assert count.sum() == probs.shape[0]


def test_degenerate_inputs_are_handled():
    assert fit_temperature(np.zeros((1, 3)), np.zeros(1, dtype=int)) == 1.0
    assert expected_calibration_error(np.zeros((0, 3)), np.zeros(0, dtype=int))[0] == 0.0
    assert brier_score(np.zeros((0, 3)), np.zeros(0, dtype=int)) == 0.0


def test_shape_mismatches_raise():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((10,)), np.zeros(10, dtype=int))
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((10, 3)), np.zeros(9, dtype=int))
