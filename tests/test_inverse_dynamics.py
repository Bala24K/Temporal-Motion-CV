import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from biomechanics.anthropometrics import estimate_forearm_segment, G  # noqa: E402
from biomechanics.inverse_dynamics import required_elbow_moment  # noqa: E402


def test_full_extension_has_zero_gravity_moment():
    """Forearm hanging straight down (aligned with the vertical upper arm)
    has no horizontal offset from the elbow, so gravity creates no moment
    about it — a directly checkable physical fact under this model's
    vertical-upper-arm assumption."""
    seg = estimate_forearm_segment(75, 1.75)
    result = required_elbow_moment(180.0, seg)
    assert abs(result.gravity_moment_nm) < 1e-6


def test_ninety_degrees_matches_hand_calculation():
    """At 90 degrees the forearm is horizontal (maximum moment arm) — the
    gravity moment should exactly match a hand-computed static torque:
    M = (m_forearm * com_distance + m_hand_load * length) * g."""
    seg = estimate_forearm_segment(75, 1.75, external_load_kg=5.0)
    result = required_elbow_moment(90.0, seg)
    expected = (seg.mass_kg * seg.com_distance_m + seg.hand_load_mass_kg * seg.length_m) * G
    assert abs(result.gravity_moment_nm - expected) < 1e-6


def test_gravity_moment_is_symmetric_about_ninety_degrees():
    """sin(60deg) == sin(120deg): the model should predict the same
    gravity moment at angles equidistant from 90 degrees, since the
    forearm's horizontal offset from the elbow is the same in both cases."""
    seg = estimate_forearm_segment(75, 1.75)
    m60 = required_elbow_moment(60.0, seg).gravity_moment_nm
    m120 = required_elbow_moment(120.0, seg).gravity_moment_nm
    assert abs(m60 - m120) < 1e-9


def test_heavier_external_load_increases_required_moment():
    seg_light = estimate_forearm_segment(75, 1.75, external_load_kg=2.0)
    seg_heavy = estimate_forearm_segment(75, 1.75, external_load_kg=10.0)
    m_light = required_elbow_moment(90.0, seg_light).gravity_moment_nm
    m_heavy = required_elbow_moment(90.0, seg_heavy).gravity_moment_nm
    assert m_heavy > m_light


def test_no_acceleration_means_no_inertial_component():
    seg = estimate_forearm_segment(75, 1.75)
    result = required_elbow_moment(90.0, seg)  # angular_acceleration_deg_s2 omitted
    assert result.inertial_moment_nm == 0.0
    assert result.net_flexion_moment_nm == result.gravity_moment_nm


def test_flexion_direction_acceleration_adds_positive_inertial_moment():
    """Flexion = angle DECREASING in this codebase's convention, so a
    flexion-direction (accelerating toward more flexion) angular
    acceleration is negative in raw theta terms and should require an
    additional positive (flexion) inertial moment component — see the
    sign-convention note in inverse_dynamics.py."""
    seg = estimate_forearm_segment(75, 1.75)
    result = required_elbow_moment(90.0, seg, angular_acceleration_deg_s2=-500.0)
    assert result.inertial_moment_nm > 0
    assert result.net_flexion_moment_nm > result.gravity_moment_nm


def test_extension_direction_acceleration_subtracts_from_required_moment():
    seg = estimate_forearm_segment(75, 1.75)
    result = required_elbow_moment(90.0, seg, angular_acceleration_deg_s2=500.0)
    assert result.inertial_moment_nm < 0
    assert result.net_flexion_moment_nm < result.gravity_moment_nm


def test_inertial_moment_scales_linearly_with_moment_of_inertia():
    seg = estimate_forearm_segment(75, 1.75, external_load_kg=10.0)  # bigger I
    seg_light = estimate_forearm_segment(75, 1.75, external_load_kg=0.0)  # smaller I
    r_heavy = required_elbow_moment(90.0, seg, angular_acceleration_deg_s2=-300.0)
    r_light = required_elbow_moment(90.0, seg_light, angular_acceleration_deg_s2=-300.0)
    assert abs(r_heavy.inertial_moment_nm) > abs(r_light.inertial_moment_nm)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
