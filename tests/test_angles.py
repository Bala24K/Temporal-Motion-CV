import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomechanics.angles import angle_3pt, angular_velocity, distance, normalized_displacement  # noqa: E402


def test_right_angle():
    # shoulder(0,0) - elbow(0,-1) - wrist(1,-1): a clean 90-degree bend
    a = (0, 0)
    b = (0, -1)
    c = (1, -1)
    assert abs(angle_3pt(a, b, c) - 90.0) < 1e-6


def test_straight_arm_is_180():
    a = (0, 0)
    b = (1, 0)
    c = (2, 0)
    assert abs(angle_3pt(a, b, c) - 180.0) < 1e-6


def test_fully_folded_is_near_zero():
    a = (0, 0)
    b = (1, 0)
    c = (0, 0.001)  # nearly folded back onto itself
    angle = angle_3pt(a, b, c)
    assert angle < 5.0


def test_degenerate_zero_length_segment_returns_none():
    a = (0, 0)
    b = (0, 0)  # elbow coincides with shoulder — degenerate
    c = (1, 1)
    assert angle_3pt(a, b, c) is None


def test_angular_velocity_basic():
    v = angular_velocity(160.0, 120.0, dt=0.5)
    assert abs(v - (-80.0)) < 1e-6


def test_angular_velocity_none_when_missing_reading():
    assert angular_velocity(None, 120.0, dt=0.5) is None
    assert angular_velocity(120.0, None, dt=0.5) is None
    assert angular_velocity(120.0, 130.0, dt=0.0) is None


def test_distance_basic():
    assert abs(distance((0, 0), (3, 4)) - 5.0) < 1e-6


def test_normalized_displacement_scales_correctly():
    # same raw displacement, different body scale -> different normalized value
    close_up = normalized_displacement((0, 0), (0.1, 0), scale=0.5)
    far_away = normalized_displacement((0, 0), (0.1, 0), scale=0.2)
    assert close_up < far_away  # smaller torso scale -> relatively bigger movement


def test_normalized_displacement_none_without_scale():
    assert normalized_displacement((0, 0), (1, 1), scale=None) is None
    assert normalized_displacement((0, 0), (1, 1), scale=0) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
