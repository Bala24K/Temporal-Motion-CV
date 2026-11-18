import numpy as np
import pytest

from exercise_system.registry import ExerciseRegistry
from motion_engine.learned import LearnedRecognizer, window_features
from motion_engine.synthetic import _pose, smoothstep, synth_sequence
from motion_engine.uncertainty import (ConformalClassifier, MahalanobisOOD, PrototypeClassifier, auroc, decide, softmax)

NOVEL = {
    "y_raise": lambda p: dict(arm_elev=8 + 152 * p, elbow_int=172),
    "jumping_jack": lambda p: dict(arm_elev=8 + 160 * p, elbow_int=172, knee_int=172 - 20 * p),
    "side_bend": lambda p: dict(lean=30 * p, arm_elev=10),
    "squat_curl_combo": lambda p: dict(knee_int=172 - 60 * p, elbow_int=170 - 120 * p),
    "hinge_press": lambda p: dict(lean=60 * p, arm_elev=8 + 120 * p, elbow_int=170 - 60 * p),
}


def seq(fn, n=240):
    return [dict(_pose(**fn(smoothstep(1 - abs(2 * ((i % 80) / 80) - 1))))) for i in range(n)]


def gauss(n, mu, rng, d=6):
    return rng.normal(mu, 1.0, (n, d))


def test_auroc_basic():
    assert auroc([0, 1, 2], [3, 4]) == 1.0 and auroc([3, 4], [0, 1]) == 0.0 and auroc([1, 1], [1, 1]) == 0.5


def test_mahalanobis_flags_far_points_and_keeps_id():
    rng = np.random.default_rng(0)
    X = np.vstack([gauss(300, 0, rng), gauss(300, 5, rng)])
    y = np.array([0] * 300 + [1] * 300)
    val = np.vstack([gauss(200, 0, rng), gauss(200, 5, rng)])
    ood = MahalanobisOOD.fit(X, y).calibrate(val)
    test = np.vstack([gauss(200, 0, rng), gauss(200, 5, rng)])
    assert ood.is_ood(test).mean() < 0.05
    assert ood.is_ood(gauss(50, 15, rng)).all()
    assert auroc(ood.score(test), ood.score(gauss(100, 12, rng))) == 1.0


def test_conformal_coverage_holds_on_noisy_classes():
    rng = np.random.default_rng(1)
    def data(n):
        y = rng.integers(0, 3, n)
        return np.eye(3)[y] * 2.0 + rng.normal(0, 1.2, (n, 3)), y
    Xc, yc = data(600)
    Xt, yt = data(3000)
    conf = ConformalClassifier.fit(softmax(Xc), yc, alpha=0.1)
    sets = conf.predict_set(softmax(Xt))
    cov = np.mean([y in s for s, y in zip(sets, yt)])
    assert cov >= 0.88                                       # target 0.90, finite-sample slack
    assert np.mean([len(s) for s in sets]) > 1.0             # noisy classes -> sets must widen, not lie


def test_decide_rule_orders_ood_before_confidence():
    rng = np.random.default_rng(2)
    X = np.vstack([gauss(200, 0, rng), gauss(200, 6, rng)]); y = np.array([0] * 200 + [1] * 200)
    ood = MahalanobisOOD.fit(X, y).calibrate(X)
    conf = ConformalClassifier(qhat=0.2, alpha=0.1)
    far = np.full((1, 6), 40.0)
    assert decide(np.array([0.99, 0.01]), conf, ood, far).action == "fallback"       # confident but far: still abstain
    assert decide(np.array([0.95, 0.05]), conf, ood, X[:1]).action == "analyze"
    assert decide(np.array([0.55, 0.45]), conf, ood, X[:1]).action == "fallback"      # nothing reaches p>=0.8
    assert decide(np.array([0.55, 0.45]), ConformalClassifier(0.6, 0.1), ood, X[:1]).action == "confirm"
    assert decide(np.array([0.5, 0.5]), ConformalClassifier(0.01, 0.1), ood, X[:1]).action == "fallback"


def test_prototype_classifier_is_few_shot():
    rng = np.random.default_rng(3)
    X = np.vstack([gauss(50, 0, rng), gauss(50, 6, rng)]); labels = ["a"] * 50 + ["b"] * 50
    pc = PrototypeClassifier.fit(X, labels)
    new = gauss(3, -6, rng)                                  # only three examples of a brand-new class
    assert pc.predict(new)[0] in ("a", "b")                  # before: forced into a known class
    pc2 = pc.add_class("c", new)
    assert pc2.predict(gauss(20, -6, rng)).count("c") >= 19 and pc2.predict(gauss(20, 0, rng)).count("a") >= 19


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from ml.train_recognizer import main
    p = tmp_path_factory.mktemp("u") / "rec.onnx"
    meta = main(str(p), fast=True)
    return LearnedRecognizer(p), meta


def test_regression_confidently_wrong_movements_now_abstain(model):
    """The original failure: Y-raise / jumping-jack scored conf=1.00 as lateral/front raise."""
    L, _ = model
    for name, fn in NOVEL.items():
        r = L.recognize(seq(fn))
        assert r.action != "analyze", f"{name} was analysed as {r.top.members if r.top else None}"
    assert sum(L.recognize(seq(fn)).action == "fallback" for fn in NOVEL.values()) >= 4


def test_known_exercises_still_recognised_with_low_false_abstention(model):
    L, meta = model
    ok = 0
    for ex in ("squat", "shoulder_press", "romanian_deadlift", "overhead_triceps_extension", "goblet_squat"):
        r = L.recognize(synth_sequence(ex, reps=4, seed=777, scale=0.85)[0])
        ok += r.status in ("recognized", "ambiguous") and ex in r.top.members
    assert ok >= 4
    assert meta["id_ood_false_flag_rate"] < 0.1 and meta["conformal_coverage_synthetic"] >= 0.85
