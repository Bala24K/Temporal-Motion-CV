import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.model_comparison import run_comparison  # noqa: E402


def test_comparison_returns_all_four_models_with_valid_reports():
    results = run_comparison(n_subjects=8, windows_per_subject_per_exercise=6,
                              n_frames=10, seed=0, cnn_epochs=5)
    expected_models = {"random_forest", "xgboost", "mlp", "temporal_cnn"}
    assert set(results.keys()) == expected_models
    for name, result in results.items():
        assert 0.0 <= result.report.accuracy <= 1.0, f"{name} accuracy out of range"
        assert result.n_train > 0 and result.n_test > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
