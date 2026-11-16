import math
from dataclasses import replace

import numpy as np
import pytest

from f1_ai.racing_controller import FollowerConfig, RacingController, ReferenceState
from f1_ai.racing_line import ReferenceConfig, plan_reference
from f1_ai.vehicle import FormulaConfig


@pytest.fixture(scope="module")
def plan():
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    return plan_reference(
        40 * np.column_stack((np.cos(angles), np.sin(angles))),
        5,
        FormulaConfig(),
        ReferenceConfig(samples=32, optimization_steps=0),
    )


def state_on_circle(plan, speed=10, **changes):
    state = ReferenceState(
        position=tuple(plan.points[0]),
        heading_rad=0,
        velocity=(0, speed),
        yaw_rate_rad_s=speed / np.linalg.norm(plan.points[0]),
        slip_ratios=(0, 0, 0, 0),
        locked_wheels=(False, False, False, False),
    )
    return replace(state, **changes)


def test_follower_turns_left_for_a_counterclockwise_path_and_rates_its_controls(plan):
    controller = RacingController(plan)
    state = state_on_circle(plan)
    previous = np.zeros(2)
    for _ in range(20):
        action = controller.action(state)
        assert action.dtype == np.float32 and np.isfinite(action).all()
        assert np.max(np.abs(action)) <= 1
        assert -0.075001 <= action[0] - previous[0] <= 0.075001
        assert -0.080001 <= action[1] - previous[1] <= 0.040001
        previous = action
    assert action[0] < 0 and action[1] > 0
    controller.reset()
    np.testing.assert_array_equal(controller.action(state), RacingController(plan).action(state))


def test_excess_rear_slip_releases_throttle_without_waiting_for_the_ramp(plan):
    controller = RacingController(plan)
    state = state_on_circle(plan)
    for _ in range(20):
        action = controller.action(state)
    assert action[1] > 0.08
    action = controller.action(replace(state, slip_ratios=(0, 0, 0.9, 0.9)))
    assert action[1] <= 0


def test_wheel_lock_reduces_requested_braking(plan):
    commands = []
    for locked in (False, True):
        controller = RacingController(plan)
        # Mismo exceso de velocidad, todavía dentro de la envolvente lateral.
        state = state_on_circle(plan, speed=30, yaw_rate_rad_s=0)
        state = replace(state, locked_wheels=(locked,) * 4)
        for _ in range(30):
            action = controller.action(state)
        commands.append(action[1])
    assert commands[0] < commands[1] < 0


def test_pure_pursuit_can_follow_the_path_in_a_small_kinematic_rollout(plan):
    # Solo verifica signos y geometría. No modela neumáticos ni valida Box2D.
    controller = RacingController(plan)
    position = plan.points[0].copy()
    heading = 0.0
    speed, dt = 12.0, 0.02
    yaw_rate = speed / np.linalg.norm(position)
    radii = []
    for _ in range(400):
        state = ReferenceState(
            tuple(position),
            heading,
            (-speed * math.sin(heading), speed * math.cos(heading)),
            yaw_rate,
            (0, 0, 0, 0),
            (False,) * 4,
        )
        steering = controller.action(state, dt)[0] * 0.4
        yaw_rate = -speed * math.tan(steering) / plan.vehicle.wheelbase_m
        heading += yaw_rate * dt
        position += speed * dt * np.array([-math.sin(heading), math.cos(heading)])
        radii.append(np.linalg.norm(position))
    assert heading > 2
    assert max(abs(np.array(radii) - np.linalg.norm(plan.points[0]))) < 1


@pytest.mark.parametrize("dt", [0, -0.1, 1, float("nan"), True])
def test_invalid_timestep_is_rejected(plan, dt):
    with pytest.raises(ValueError):
        RacingController(plan).action(state_on_circle(plan), dt)


def test_nonfinite_state_or_invalid_feedback_is_rejected(plan):
    state = state_on_circle(plan)
    for changes in ({"position": (float("nan"), 0)}, {"locked_wheels": (1,) * 4}):
        with pytest.raises(ValueError):
            replace(state, **changes)
    with pytest.raises(ValueError):
        FollowerConfig(drive_slip_cut=0.1)
