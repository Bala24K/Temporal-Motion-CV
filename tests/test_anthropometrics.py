import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomechanics.anthropometrics import estimate_forearm_segment  # noqa: E402


def test_mass_scales_linearly_with_body_mass():
    a = estimate_forearm_segment(body_mass_kg=70, body_height_m=1.75)
    b = estimate_forearm_segment(body_mass_kg=140, body_height_m=1.75)
    assert abs(b.mass_kg - 2 * a.mass_kg) < 1e-9
    assert abs(b.hand_load_mass_kg - 2 * a.hand_load_mass_kg) < 1e-9


def test_length_scales_linearly_with_height():
    a = estimate_forearm_segment(body_mass_kg=75, body_height_m=1.60)
    b = estimate_forearm_segment(body_mass_kg=75, body_height_m=1.80)
    assert b.length_m > a.length_m
    ratio = b.length_m / a.length_m
    assert abs(ratio - (1.80 / 1.60)) < 1e-9


def test_external_load_increases_hand_mass_and_inertia():
    bare = estimate_forearm_segment(body_mass_kg=75, body_height_m=1.75, external_load_kg=0.0)
    loaded = estimate_forearm_segment(body_mass_kg=75, body_height_m=1.75, external_load_kg=8.0)
    assert loaded.hand_load_mass_kg > bare.hand_load_mass_kg
    assert loaded.moment_of_inertia_kg_m2 > bare.moment_of_inertia_kg_m2
    # the mass difference should be exactly the added load
    assert abs((loaded.hand_load_mass_kg - bare.hand_load_mass_kg) - 8.0) < 1e-9


def test_com_distance_is_within_segment_length():
    seg = estimate_forearm_segment(body_mass_kg=75, body_height_m=1.75)
    assert 0 < seg.com_distance_m < seg.length_m


def test_invalid_inputs_raise():
    for bad_mass, bad_height in [(0, 1.75), (-10, 1.75), (75, 0), (75, -1.8)]:
        try:
            estimate_forearm_segment(bad_mass, bad_height)
            assert False, f"expected ValueError for mass={bad_mass}, height={bad_height}"
        except ValueError:
            pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
