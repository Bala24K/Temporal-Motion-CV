"""
Tensor construction tests.

Most of the risk in this layer is silent: a gap filled with a plausible
number, a mask that no longer lines up with the coordinates it describes,
a bone stream computed against the wrong parent. None of those raise, and
all of them degrade the model in ways that look like "the architecture
didn't help".
"""
import numpy as np
import pytest

from pose.landmarks import Landmark, PoseFrame
from motion.graph import JOINT_INDEX, NUM_JOINTS, ROOT_INDEX, parent_array
from motion.tensors import (
    NUM_COORDS,
    SkeletonWindow,
    STREAM_NAMES,
    bone_stream,
    build_streams,
    build_window,
    describe_channels,
    fill_gaps,
    frames_to_raw_array,
    motion_stream,
    resample_mask,
    resample_time,
    stream_channel_count,
)
from tests.motion_fixtures import NEUTRAL, curl_clip, make_pose_frame, neutral_clip


def test_frames_to_raw_array_shapes_and_validity():
    frames = neutral_clip(n_frames=10)
    coords, valid, frame_valid = frames_to_raw_array(frames)

    assert coords.shape == (10, NUM_JOINTS, NUM_COORDS)
    assert valid.shape == (10, NUM_JOINTS)
    assert frame_valid.shape == (10,)
    assert frame_valid.all(), "a fully visible neutral pose must normalise"
    # All 13 detected landmarks plus the 2 synthetic centres.
    assert valid.sum(axis=1).min() == NUM_JOINTS


def test_unnormalisable_frames_are_kept_in_place_and_marked_invalid():
    # Dropping them would compress the time axis and corrupt velocity.
    frames = neutral_clip(n_frames=6)
    broken = PoseFrame(timestamp=frames[3].timestamp)
    broken.landmarks["nose"] = Landmark(x=0.5, y=0.1, visibility=0.9)
    frames[3] = broken

    coords, valid, frame_valid = frames_to_raw_array(frames)
    assert coords.shape[0] == 6
    assert not frame_valid[3]
    assert frame_valid[[0, 1, 2, 4, 5]].all()
    assert not valid[3].any()
    assert np.isnan(coords[3]).all()


def test_landmarks_below_visibility_threshold_are_not_trusted():
    pts = dict(NEUTRAL)
    frame = make_pose_frame(pts, visibility=0.95)
    frame.landmarks["left_wrist"] = Landmark(x=0.39, y=0.56, visibility=0.1)

    _, valid, _ = frames_to_raw_array([frame], min_visibility=0.4)
    assert not valid[0, JOINT_INDEX["left_wrist"]]
    assert valid[0, JOINT_INDEX["right_wrist"]]


def test_fill_gaps_interpolates_short_runs():
    coords = np.zeros((6, 1, 1), dtype=float)
    coords[:, 0, 0] = [0.0, np.nan, np.nan, np.nan, 4.0, 5.0]
    filled = fill_gaps(coords, max_gap=4)
    np.testing.assert_allclose(filled[:, 0, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_fill_gaps_holds_across_long_runs_instead_of_inventing_a_trajectory():
    coords = np.zeros((8, 1, 1), dtype=float)
    coords[:, 0, 0] = [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0]
    filled = fill_gaps(coords, max_gap=2)
    values = filled[:, 0, 0]
    assert set(np.unique(values)) == {0.0, 7.0}, "a long gap must be held, not interpolated"
    assert values[1] == 0.0 and values[-2] == 7.0


def test_fill_gaps_holds_at_the_edges():
    coords = np.zeros((5, 1, 1), dtype=float)
    coords[:, 0, 0] = [np.nan, np.nan, 3.0, np.nan, np.nan]
    filled = fill_gaps(coords)
    np.testing.assert_allclose(filled[:, 0, 0], [3.0, 3.0, 3.0, 3.0, 3.0])


def test_fill_gaps_leaves_never_observed_channels_as_nan():
    coords = np.full((4, 1, 1), np.nan)
    assert np.isnan(fill_gaps(coords)).all()


def test_resample_time_preserves_endpoints_and_length():
    arr = np.linspace(0, 10, 11).reshape(11, 1, 1)
    out = resample_time(arr, 5)
    assert out.shape == (5, 1, 1)
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[-1, 0, 0] == pytest.approx(10.0)

    up = resample_time(arr, 21)
    assert up.shape == (21, 1, 1)
    assert up[-1, 0, 0] == pytest.approx(10.0)


def test_resample_time_handles_degenerate_lengths():
    assert resample_time(np.zeros((0, 2, 3)), 4).shape == (4, 2, 3)
    single = resample_time(np.ones((1, 2, 3)), 4)
    assert single.shape == (4, 2, 3) and np.all(single == 1.0)


def test_resample_mask_stays_boolean():
    mask = np.array([[True], [False], [True], [False]])
    out = resample_mask(mask, 8)
    assert out.dtype == bool
    assert out.shape == (8, 1)


def test_bone_stream_matches_parent_differences_and_zeroes_the_root():
    coords = np.random.default_rng(0).normal(size=(5, NUM_JOINTS, 3)).astype(np.float32)
    bones = bone_stream(coords)
    parent = parent_array()

    assert np.all(bones[:, ROOT_INDEX, :] == 0.0)
    j = JOINT_INDEX["left_wrist"]
    np.testing.assert_allclose(bones[:, j], coords[:, j] - coords[:, parent[j]], rtol=1e-6)


def test_bone_stream_is_translation_invariant():
    coords = np.random.default_rng(1).normal(size=(4, NUM_JOINTS, 3)).astype(np.float32)
    shifted = coords + np.array([0.3, -0.2, 0.1], dtype=np.float32)
    np.testing.assert_allclose(bone_stream(coords), bone_stream(shifted), atol=1e-5)


def test_motion_stream_first_frame_is_zero_not_wrapped():
    arr = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    motion = motion_stream(arr)
    assert np.all(motion[0] == 0.0)
    np.testing.assert_allclose(motion[1:], arr[1:] - arr[:-1])


def test_build_window_end_to_end():
    window = build_window(curl_clip(n_frames=30), target_len=48)
    assert window is not None
    assert window.coords.shape == (48, NUM_JOINTS, 3)
    assert window.valid.shape == (48, NUM_JOINTS)
    assert window.num_frames == 48
    assert window.valid_ratio == pytest.approx(1.0)
    assert window.coords.dtype == np.float32


def test_build_window_returns_none_when_nothing_normalises():
    bad = PoseFrame()
    bad.landmarks["nose"] = Landmark(x=0.5, y=0.1, visibility=0.9)
    assert build_window([bad, bad, bad]) is None
    assert build_window([]) is None


def test_build_window_centres_the_pelvis():
    # Normalisation puts the hip midpoint at the origin; if this drifts,
    # every geometric augmentation about the origin is wrong too.
    window = build_window(neutral_clip(n_frames=12), target_len=16)
    hip = window.coords[:, JOINT_INDEX["hip_center"], :2]
    assert np.abs(hip).max() < 1e-4


def test_build_streams_channel_layout():
    window = build_window(curl_clip(n_frames=24), target_len=32)
    x = build_streams(window, STREAM_NAMES, include_validity_channel=True)

    expected_c = stream_channel_count(STREAM_NAMES, True)
    assert expected_c == 4 * 3 + 1
    assert x.shape == (expected_c, 32, NUM_JOINTS)
    assert x.dtype == np.float32
    assert len(describe_channels(STREAM_NAMES, True)) == expected_c
    # The validity channel is the last one and is 0/1 valued.
    assert set(np.unique(x[-1])) <= {0.0, 1.0}


def test_build_streams_subset_and_no_validity_channel():
    window = build_window(curl_clip(n_frames=20), target_len=24)
    x = build_streams(window, ("joint",), include_validity_channel=False)
    assert x.shape == (3, 24, NUM_JOINTS)


def test_build_streams_rejects_unknown_or_empty_stream_lists():
    window = build_window(curl_clip(n_frames=20), target_len=24)
    with pytest.raises(ValueError):
        build_streams(window, ("joint", "velocity"))
    with pytest.raises(ValueError):
        build_streams(window, ())


def test_z_channel_is_attenuated_relative_to_xy():
    # Single-camera depth is much noisier than x/y; the scaling is
    # deliberate and worth pinning so it is not silently removed.
    frames = neutral_clip(n_frames=8)
    for f in frames:
        for lm in f.landmarks.values():
            lm.z = 1.0
    window = build_window(frames, target_len=8)
    assert np.abs(window.coords[:, :, 2]).max() < 1.0
