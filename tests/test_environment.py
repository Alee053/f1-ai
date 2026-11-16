from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1_ai.config import EnvironmentConfig, load_config
from f1_ai.environment import CircuitPool, make_env


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
