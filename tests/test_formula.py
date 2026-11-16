from dataclasses import asdict, replace

import numpy as np
import pytest
from Box2D.b2 import world
from gymnasium.utils.env_checker import check_env

from f1_ai.config import EnvironmentConfig, load_config
from f1_ai.environment import make_env
from f1_ai.formula_car import FormulaCar
from f1_ai.vehicle import GRAVITY, FormulaConfig, aerodynamic_forces, normal_loads, tyre_forces


class Road:
    road_friction = 1.0


def flat_car(config=None, velocity=(0, 0)):
    physics_world = world(gravity=(0, 0))
    car = FormulaCar(physics_world, 0, 0, 0, config or FormulaConfig())
    road = Road()
    car.hull.linearVelocity = velocity
    for wheel in car.wheels:
        wheel.tiles.add(road)
        wheel.linearVelocity = velocity
        wheel.omega = velocity[1] / wheel.wheel_rad
    return physics_world, car


def advance(physics_world, car, steps):
    for _ in range(steps):
        car.step(0.02)
        physics_world.Step(0.02, 180, 60)


def test_downforce_and_drag_grow_with_speed_squared():
    config = FormulaConfig()
    assert aerodynamic_forces(0, config) == (0, 0)
    slow = aerodynamic_forces(100 / 3.6, config)
    fast = aerodynamic_forces(200 / 3.6, config)
    assert fast == pytest.approx(tuple(4 * x for x in slow))
    assert fast == pytest.approx((9452.160493827163, 2268.518518518519))


@pytest.mark.parametrize("ax,ay", [(0, 0), (30, 50), (-40, -60)])
def test_vertical_load_is_conserved_and_transfers_between_wheels(ax, ay):
    config = FormulaConfig()
    loads = normal_loads(config, 70, ax, ay)
    assert min(loads) >= 0
    assert sum(loads) == pytest.approx(config.mass_kg * GRAVITY + aerodynamic_forces(70, config)[0])
    if ax > 0:
        static = normal_loads(config, 70, 0, 0)
        assert sum(loads[:2]) < sum(static[:2])
    if ay > 0:
        assert loads[0] > loads[1] and loads[2] > loads[3]


@pytest.mark.parametrize("slip,angle", [(0, 0), (1, 0.3), (-1, -0.5), (20, 1)])
def test_tyre_combined_force_respects_available_grip(slip, angle):
    fx, fy, limit = tyre_forces(FormulaConfig(), slip, angle, 4000, 1)
    assert np.hypot(fx, fy) <= limit * (1 + 1e-12)
    if slip:
        assert np.sign(fx) == np.sign(slip)
    if angle:
        assert np.sign(fy) == -np.sign(angle)


def test_downforce_improves_lateral_response_without_changing_mass():
    results = []
    for cla in [0, 5]:
        physics_world, car = flat_car(replace(FormulaConfig(), downforce_cla_m2=cla), (10, 50))
        try:
            mass = car.hull.mass + sum(w.mass for w in car.wheels)
            advance(physics_world, car, 1)
            results.append(float(car.hull.linearVelocity[0]))
            assert car.hull.mass + sum(w.mass for w in car.wheels) == mass
            assert mass == pytest.approx(770, rel=1e-6)
        finally:
            car.destroy()
    assert results[1] < results[0] - 0.1


def test_full_throttle_can_spin_rear_wheels_without_traction_control():
    physics_world, car = flat_car()
    try:
        car.gas(1)
        advance(physics_world, car, 50)
        assert car.hull.linearVelocity[1] > 5
        assert max(abs(x) for x in car.telemetry["slip_ratios"][2:]) > 0.5
        assert car.wheels[0].gas == car.wheels[1].gas == 0
    finally:
        car.destroy()


def test_brakes_can_lock_wheels_and_stop_without_launching_backwards():
    physics_world, car = flat_car(velocity=(0, 50))
    try:
        car.brake(1)
        advance(physics_world, car, 15)
        assert any(car.telemetry["locked_wheels"])
        assert 0 < car.hull.linearVelocity[1] < 50
        advance(physics_world, car, 300)
        assert car.hull.linearVelocity.length < 0.1
    finally:
        car.destroy()


def test_drag_dissipates_energy_in_forward_and_reverse_motion():
    for speed in [50, -50]:
        physics_world, car = flat_car(velocity=(0, speed))
        try:
            advance(physics_world, car, 20)
            assert 0 < car.hull.linearVelocity.length < abs(speed)
        finally:
            car.destroy()


@pytest.mark.parametrize(
    "changes",
    [{"mass_kg": float("nan")}, {"power_kw": -1}, {"downforce_cla_m2": -1}, {"mass_kg": True}],
)
def test_invalid_physical_parameters_are_rejected(changes):
    with pytest.raises(ValueError):
        FormulaConfig(**changes)


def test_formula_profile_works_through_gymnasium_and_serialization():
    config = load_config("configs/formula.toml")
    assert EnvironmentConfig(**asdict(config.environment)) == config.environment
    env = make_env(config.environment)
    try:
        check_env(env, skip_render_check=True)
        assert isinstance(env.car, FormulaCar)
        first, _ = env.reset(seed=0)
        result = env.step(np.array([0.1, 0.2], dtype=np.float32))
        second, _ = env.reset(seed=0)
        again = env.step(np.array([0.1, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(result[0], again[0])
        assert result[1:] == again[1:]
        assert env.surf is None
    finally:
        env.close()
