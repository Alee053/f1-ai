from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.envs.box2d.car_racing import PLAYFIELD, TRACK_WIDTH, CarRacing
from gymnasium.utils.env_checker import check_env

from f1_ai.config import EnvironmentConfig, load_config
from f1_ai.environment import CircuitPool, make_env
from f1_ai.track_layout import TrackLayout


def test_gymnasium_contract_and_no_internal_rgb_render():
    env = make_env(EnvironmentConfig())
    try:
        check_env(env, skip_render_check=True)
        assert env.surf is None
        assert env.screen is None
        assert env.clock is None
    finally:
        env.close()


def test_reproducible_reset_and_step():
    env = make_env(EnvironmentConfig())
    try:
        first, _ = env.reset(seed=4)
        first_step = env.step(np.array([0.1, 0.5], dtype=np.float32))
        second, _ = env.reset(seed=4)
        second_step = env.step(np.array([0.1, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]
    finally:
        env.close()


def test_timeout_is_truncation_not_a_completed_lap():
    env = make_env(EnvironmentConfig(max_steps=2))
    try:
        env.reset(seed=0)
        env.step(np.zeros(2, dtype=np.float32))
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated and truncated
        assert not info["lap_complete"]
        assert info["termination_reason"] == "time_limit"
    finally:
        env.close()


@pytest.mark.parametrize("action", [[0, float("nan")], [0, 2], [0, 0, 0]])
def test_invalid_action_is_rejected(action):
    env = make_env(EnvironmentConfig())
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="acción"):
            env.step(np.array(action, dtype=np.float32))
    finally:
        env.close()


def test_validation_circuits_cycle_in_fixed_order():
    env = CircuitPool(make_env(EnvironmentConfig()), (100, 101), shuffle=False)
    try:
        assert [env.reset()[1]["circuit_seed"] for _ in range(3)] == [100, 101, 100]
    finally:
        env.close()


def test_train_and_validation_circuits_cannot_overlap():
    config = load_config("configs/ppo.toml")
    with pytest.raises(ValueError, match="repetirse"):
        replace(config.circuits, validation=(config.circuits.train[0],))


def test_numeric_adapter_preserves_native_physics_and_native_reward():
    numeric = make_env(EnvironmentConfig())
    native = CarRacing()
    try:
        numeric.reset(seed=3)
        native.reset(seed=3)
        for _ in range(20):
            _, _, _, _, info = numeric.step(np.array([0.1, 0.5], dtype=np.float32))
            _, reward, _, _, _ = native.step(np.array([0.1, 0.5, 0.0], dtype=np.float32))
            np.testing.assert_allclose(numeric.car.hull.position, native.car.hull.position)
            assert info["native_reward"] == pytest.approx(reward)
    finally:
        numeric.close()
        native.close()


def test_leaving_playfield_is_failure_not_success():
    env = make_env(EnvironmentConfig())
    try:
        env.reset(seed=0)
        env.car.hull.position = (100000, 100000)
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert terminated and not truncated
        assert info["termination_reason"] == "out_of_bounds"
        assert not info["lap_complete"]
    finally:
        env.close()


def test_monaco_uses_its_actual_road_edges_outside_the_procedural_arena():
    config = load_config("configs/formula-road-learning-braking-cuda.toml")
    layout = TrackLayout.load(Path("tracks/monaco.json"))
    env = make_env(config.environment, track_layout=layout)
    try:
        observation, info = env.reset(seed=0)
        assert info["track_length"] == pytest.approx(3337.0, abs=0.01)
        assert abs(info["position"][0]) > PLAYFIELD
        assert env.road_contact_fraction == 1.0
        left, right = observation[16:20].reshape(2, 2)
        assert np.linalg.norm(right - left) * (6.0 + TRACK_WIDTH) == pytest.approx(10, rel=0.01)
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert not (terminated or truncated)
        assert info["native_reward"] != -100
    finally:
        env.close()


def test_normalized_steering_maps_to_physical_joint_range():
    numeric = make_env(EnvironmentConfig(steering_scale=0.4))
    native = CarRacing()
    try:
        numeric.reset(seed=3)
        native.reset(seed=3)
        for _ in range(20):
            numeric.step(np.array([0.5, 0.3], dtype=np.float32))
            native.step(np.array([0.2, 0.3, 0.0], dtype=np.float32))
            np.testing.assert_allclose(numeric.car.hull.position, native.car.hull.position)
    finally:
        numeric.close()
        native.close()


def test_track_reward_never_credits_off_track_progress():
    env = make_env(EnvironmentConfig(reward_profile="track"))
    outside_steps = 0
    try:
        env.reset(seed=100)
        for _ in range(600):
            _, reward, terminated, truncated, _ = env.step(np.array([0, 1], dtype=np.float32))
            if env.road_contact_fraction < 0.5:
                outside_steps += 1
                assert reward <= -0.1
            if terminated or truncated:
                break
        assert outside_steps > 0
    finally:
        env.close()
