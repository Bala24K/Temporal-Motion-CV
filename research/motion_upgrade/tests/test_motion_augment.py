"""
Augmentation tests.

The property that matters is that every transform preserves the label.
A mirror that moves coordinates without permuting the left/right joint
indices still trains — it just teaches the model that a left-arm curl and
a right-arm curl are anatomically different, and the damage shows up only
as a slightly disappointing accuracy number.
"""
import numpy as np
import pytest

from motion.augment import (
    AugmentConfig,
    add_noise,
    augment_window,
    joint_dropout,
    mirror,
    rotate_xy,
    scale,
    time_mask,
    time_scale,
    translate,
)
from motion.graph import JOINT_INDEX, NUM_JOINTS
from motion.phase import derive_phase_labels, phase_labels_to_onehot, segment_reps_from_phases
from motion.tensors import build_window
from tests.motion_fixtures import curl_clip, squat_clip


@pytest.fixture
def window():
    w = build_window(curl_clip(48, cycles=2.0), target_len=48)
    assert w is not None
    return w


def test_mirror_is_self_inverse(window):
    back = mirror(mirror(window))
    np.testing.assert_allclose(back.coords, window.coords, atol=1e-6)
    np.testing.assert_array_equal(back.valid, window.valid)


def test_mirror_swaps_left_and_right_joints(window):
    flipped = mirror(window)
    left, right = JOINT_INDEX["left_wrist"], JOINT_INDEX["right_wrist"]
    # The mirrored left wrist should land where the negated right wrist is.
    np.testing.assert_allclose(flipped.coords[:, left, 0], -window.coords[:, right, 0], atol=1e-6)
    np.testing.assert_allclose(flipped.coords[:, left, 1], window.coords[:, right, 1], atol=1e-6)


def test_mirror_preserves_the_exercise_signal_and_rep_count(window):
    # The strongest label-preservation check available without a model:
    # a mirrored curl must still read as the same number of curls.
    def reps(w):
        labels, _ = derive_phase_labels(w.coords, "bicep_curl")
        return len(segment_reps_from_phases(phase_labels_to_onehot(labels)))

    assert reps(mirror(window)) == reps(window) == 2


def test_mirror_preserves_rep_count_for_a_squat():
    w = build_window(squat_clip(48, cycles=2.0), target_len=48)

    def reps(win):
        labels, _ = derive_phase_labels(win.coords, "squat")
        return len(segment_reps_from_phases(phase_labels_to_onehot(labels)))

    assert reps(mirror(w)) == reps(w) == 2


def test_rotation_preserves_distances(window):
    rotated = rotate_xy(window, 15.0)
    a, b = JOINT_INDEX["left_shoulder"], JOINT_INDEX["left_elbow"]
    before = np.linalg.norm(window.coords[:, a, :2] - window.coords[:, b, :2], axis=1)
    after = np.linalg.norm(rotated.coords[:, a, :2] - rotated.coords[:, b, :2], axis=1)
    np.testing.assert_allclose(before, after, atol=1e-5)


def test_rotation_leaves_z_untouched(window):
    rotated = rotate_xy(window, 30.0)
    np.testing.assert_allclose(rotated.coords[:, :, 2], window.coords[:, :, 2], atol=1e-6)


def test_scale_multiplies_all_coordinates(window):
    np.testing.assert_allclose(scale(window, 2.0).coords, window.coords * 2.0, atol=1e-5)


def test_translate_shifts_uniformly(window):
    offset = np.array([0.1, -0.2, 0.05])
    shifted = translate(window, offset)
    np.testing.assert_allclose(shifted.coords - window.coords,
                               np.broadcast_to(offset, window.coords.shape), atol=1e-5)


def test_time_scale_keeps_the_window_length(window):
    for factor in (0.8, 1.0, 1.3):
        out = time_scale(window, factor)
        assert out.num_frames == window.num_frames
        assert out.coords.shape == window.coords.shape


def test_noise_only_touches_detected_joints(window):
    w = joint_dropout(window, 2, np.random.default_rng(0))
    noisy = add_noise(w, 0.05, np.random.default_rng(1))
    dropped = ~w.valid[0]
    # Dropped joints must be bit-identical; detected ones must have moved.
    np.testing.assert_allclose(noisy.coords[:, dropped], w.coords[:, dropped], atol=0.0)
    assert not np.allclose(noisy.coords[:, ~dropped], w.coords[:, ~dropped])


def test_joint_dropout_marks_joints_invalid_and_freezes_them(window):
    out = joint_dropout(window, 3, np.random.default_rng(7))
    dropped = np.flatnonzero(~out.valid[0])
    assert len(dropped) == 3
    for j in dropped:
        assert np.allclose(out.coords[:, j], out.coords[0, j])
        assert not out.valid[:, j].any()


def test_time_mask_freezes_a_run_and_flags_the_frames(window):
    out = time_mask(window, max_frames=4, rng=np.random.default_rng(3))
    assert out.num_frames == window.num_frames
    assert (~out.frame_valid).sum() >= 1


def test_augment_window_is_reproducible_for_a_fixed_seed(window):
    a = augment_window(window, AugmentConfig(), np.random.default_rng(42))
    b = augment_window(window, AugmentConfig(), np.random.default_rng(42))
    np.testing.assert_allclose(a.coords, b.coords, atol=0.0)


def test_augment_off_is_a_no_op(window):
    out = augment_window(window, AugmentConfig.off(), np.random.default_rng(0))
    np.testing.assert_allclose(out.coords, window.coords, atol=0.0)
    np.testing.assert_array_equal(out.valid, window.valid)


def test_augment_never_changes_the_tensor_shape(window):
    rng = np.random.default_rng(11)
    for _ in range(25):
        out = augment_window(window, AugmentConfig(), rng)
        assert out.coords.shape == (window.num_frames, NUM_JOINTS, 3)
        assert out.valid.shape == (window.num_frames, NUM_JOINTS)
        assert np.isfinite(out.coords).all()


def test_augmentation_does_not_mutate_the_source_window(window):
    before = window.coords.copy()
    augment_window(window, AugmentConfig(), np.random.default_rng(5))
    np.testing.assert_allclose(window.coords, before, atol=0.0)
