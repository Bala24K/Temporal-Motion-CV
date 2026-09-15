import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from biomechanics.kinematics import compute_velocity, compute_acceleration, compute_kinematics  # noqa: E402


def test_velocity_of_linear_ramp_is_constant():
    # angle(t) = 10*t  -> velocity should be a constant 10 deg/s everywhere
    times = list(np.linspace(0, 2, 50))
    angles = [10.0 * t for t in times]
    v = compute_velocity(times, angles)
    assert all(abs(x - 10.0) < 1e-6 for x in v)


def test_velocity_matches_analytic_sine_derivative():
    # angle(t) = A*sin(w*t) -> velocity = A*w*cos(w*t)
    A, w = 40.0, 2.0
    times = list(np.linspace(0, 3, 400))
    angles = [A * np.sin(w * t) for t in times]
    v = compute_velocity(times, angles)
    analytic = [A * w * np.cos(w * t) for t in times]
    # skip the very first/last couple samples where finite differencing is least accurate
    err = np.max(np.abs(np.array(v[3:-3]) - np.array(analytic[3:-3])))
    assert err < 0.05, f"velocity deviates from analytic derivative by {err}"


def test_acceleration_matches_analytic_sine_second_derivative():
    # angle(t) = A*sin(w*t) -> acceleration = -A*w^2*sin(w*t)
    A, w = 40.0, 2.0
    times = list(np.linspace(0, 3, 400))
    angles = [A * np.sin(w * t) for t in times]
    v = compute_velocity(times, angles)
    acc = compute_acceleration(times, v)
    analytic = [-A * w * w * np.sin(w * t) for t in times]
    err = np.max(np.abs(np.array(acc[5:-5]) - np.array(analytic[5:-5])))
    assert err < 0.5, f"acceleration deviates from analytic 2nd derivative by {err}"


def test_zero_velocity_for_constant_angle():
    times = list(np.linspace(0, 1, 20))
    angles = [90.0] * 20
    v = compute_velocity(times, angles)
    assert all(abs(x) < 1e-9 for x in v)


def test_compute_kinematics_bundles_all_three():
    times = list(np.linspace(0, 1, 30))
    angles = [30.0 * t for t in times]
    k = compute_kinematics(times, angles)
    assert len(k.angle_deg) == len(k.angular_velocity_deg_s) == len(k.angular_acceleration_deg_s2) == 30
    assert abs(k.angular_velocity_deg_s[10] - 30.0) < 1e-6


def test_handles_short_series_without_crashing():
    assert compute_velocity([0.0], [90.0]) == [0.0]
    assert compute_acceleration([0.0], [0.0]) == [0.0]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
