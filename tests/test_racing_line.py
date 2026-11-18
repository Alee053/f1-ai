from dataclasses import replace

import numpy as np
import pytest

from f1_ai.geometry import CircuitGeometry
from f1_ai.racing_line import ReferenceConfig, _speed_profile, longitudinal_limits, plan_reference
from f1_ai.vehicle import FormulaConfig


def circle(radius=40.0, samples=64):
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


@pytest.fixture(scope="module")
def circle_plan():
    return plan_reference(
        circle(), 5.0, FormulaConfig(), ReferenceConfig(samples=32, optimization_steps=2)
    )


def test_time_selection_can_leave_the_center_and_shorten_a_circular_lap(circle_plan):
    assert circle_plan.candidate != "centerline"
    assert np.max(np.abs(circle_plan.offsets_m)) > 0.5
    assert circle_plan.estimated_lap_seconds < circle_plan.centerline_estimated_lap_seconds
    usable = 5 - circle_plan.vehicle.width_m / 2 - circle_plan.config.road_margin_m
    road = CircuitGeometry(circle())
    for point in circle_plan.points:
        projection = road.project(point)
        nearest = road.sample(np.array([projection.distance]))[0]
        assert np.linalg.norm(point - nearest) <= usable + 1e-6
    assert not circle_plan.points.flags.writeable
    assert circle_plan.to_dict()["optimality_proven"] is False
    assert circle_plan.to_dict()["box2d_driving_validated"] is False


def test_downforce_allows_more_corner_speed_with_the_same_mass():
    settings = ReferenceConfig(samples=32, optimization_steps=0)
    aero = plan_reference(circle(), 5, FormulaConfig(), settings)
    no_aero = plan_reference(circle(), 5, replace(FormulaConfig(), downforce_cla_m2=0), settings)
    assert aero.speed_m_s.min() > no_aero.speed_m_s.max()
    assert aero.vehicle.mass_kg == no_aero.vehicle.mass_kg


def test_cornering_leaves_less_grip_for_acceleration_and_braking():
    vehicle = FormulaConfig()
    straight = longitudinal_limits(vehicle, 20, 0, 0.75)
    turning = longitudinal_limits(vehicle, 20, 0.025, 0.75)
    assert straight[2] and turning[2]
    assert 0 < turning[0] < straight[0]
    assert 0 < turning[1] < straight[1]
    assert not longitudinal_limits(vehicle, 60, 0.1, 0.75)[2]


def test_closed_profile_brakes_before_tight_turns_and_respects_both_limits():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    ellipse = np.column_stack((80 * np.cos(angles), 30 * np.sin(angles)))
    plan = plan_reference(
        ellipse, 4, FormulaConfig(), ReferenceConfig(samples=64, optimization_steps=2)
    )
    distances = CircuitGeometry(plan.points).lengths
    speed = plan.speed_m_s
    curvature = plan.curvature_m_inv
    assert speed.max() > 1.2 * speed.min()
    assert speed[0] > 0  # La unión final/inicio corresponde a una vuelta lanzada.
    anticipates_braking = False
    for index, length in enumerate(distances):
        following = (index + 1) % len(speed)
        accel, _, feasible = longitudinal_limits(
            plan.vehicle, speed[index], curvature[index], plan.config.grip_fraction
        )
        _, brake, _ = longitudinal_limits(
            plan.vehicle, speed[following], curvature[following], plan.config.grip_fraction
        )
        assert feasible
        assert speed[following] ** 2 <= speed[index] ** 2 + 2 * accel * length + 1e-4
        assert speed[index] ** 2 <= speed[following] ** 2 + 2 * brake * length + 1e-4
        anticipates_braking |= bool(
            speed[index] > speed[following] + 0.1
            and abs(curvature[index]) < abs(curvature[following])
        )
    assert anticipates_braking


def test_braking_margin_reduces_entry_speed_before_the_curve():
    curvature = np.zeros(32)
    curvature[12:15] = 0.06
    lengths = np.full(32, 10.0)
    config = ReferenceConfig(samples=32)
    nominal, _ = _speed_profile(lengths, curvature, FormulaConfig(), config)
    early, _ = _speed_profile(
        lengths, curvature, FormulaConfig(), replace(config, braking_fraction=0.60)
    )
    assert early[8] < nominal[8]
    assert early[11] < nominal[11]
    assert early[13] == pytest.approx(nominal[13])


def test_time_search_preserves_corridor_and_improves_estimate_on_asymmetric_turns():
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    track = np.column_stack((75 * np.cos(angles), 38 * np.sin(angles)))
    settings = ReferenceConfig(samples=64, optimization_steps=2)
    baseline = plan_reference(track, 6.0, FormulaConfig(), settings)
    searched = plan_reference(track, 6.0, FormulaConfig(), replace(settings, time_search_passes=3))
    assert searched.estimated_lap_seconds <= baseline.estimated_lap_seconds
    assert np.isfinite(searched.speed_m_s).all()
    usable = searched.half_width_m - searched.vehicle.width_m / 2 - settings.road_margin_m
    assert np.max(np.abs(searched.offsets_m)) <= usable


@pytest.mark.parametrize(
    "settings",
    [
        {"samples": 4},
        {"samples": True},
        {"optimization_steps": 100},
        {"speed_limit_m_s": float("nan")},
        {"grip_fraction": 0},
        {"road_margin_m": -1},
        {"braking_fraction": 1.5},
        {"time_search_passes": 7},
    ],
)
def test_invalid_planning_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ReferenceConfig(**settings)


def test_too_narrow_or_nonfinite_corridors_are_rejected():
    for width in (0.5, float("nan"), True):
        with pytest.raises(ValueError):
            plan_reference(circle(), width, FormulaConfig())
