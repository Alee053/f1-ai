from dataclasses import replace

import numpy as np
import pytest
import torch

from f1_ai.config import load_config
from f1_ai.environment import make_env, observation_contract
from f1_ai.racing_controller import RacingController, ReferenceState
from f1_ai.training import create_model, load_model, train


def test_neutral_correction_reproduces_reference_even_on_a_partial_final_step():
    config = load_config("configs/formula-residual-cuda.toml").environment
    config = replace(config, max_steps=13)
    hybrid = make_env(config)
    direct = make_env(replace(config, control_profile="direct"))
    try:
        observation, _ = hybrid.reset(seed=0)
        version, names = observation_contract(config)
        assert version == 3 and observation.shape == (len(names),)
        assert hybrid.observation_space.contains(observation)
        direct.reset(seed=0)
        controller = RacingController(hybrid.controller.plan)
        initial = observation.copy()
        for ticks in (5, 5, 3):
            reward_sum = 0.0
            for _ in range(ticks):
                command = controller.action(ReferenceState.from_car(direct.car))
                _, reward, direct_terminated, direct_truncated, direct_info = direct.step(command)
                reward_sum += reward
            observation, reward, terminated, truncated, info = hybrid.step(
                np.zeros(2, dtype=np.float32)
            )
            assert reward == pytest.approx(reward_sum)
            assert (terminated, truncated) == (direct_terminated, direct_truncated)
            assert info["position"] == direct_info["position"]
            np.testing.assert_array_equal(info["applied_action"], command)
            assert info["simulated_seconds"] == direct_info["simulated_seconds"]
        assert hybrid.unwrapped.elapsed_steps == 13 and truncated
        reset, _ = hybrid.reset(seed=0)
        np.testing.assert_array_equal(reset, initial)
    finally:
        direct.close()
        hybrid.close()


def test_corrections_are_filtered_and_observed_until_reset():
    config = load_config("configs/formula-residual-cuda.toml").environment
    env = make_env(config)
    try:
        observation, _ = env.reset(seed=1)
        assert observation[-2:].tolist() == [0, 0]
        observation, _, _, _, _ = env.step(np.array([1, -1], dtype=np.float32))
        assert 0 < observation[-2] < 1 and -1 < observation[-1] < 0
        with pytest.raises(ValueError, match="corrección"):
            env.step([np.nan, 0])
        assert env.unwrapped.elapsed_steps == 5
        observation, _ = env.reset(seed=1)
        assert observation[-2:].tolist() == [0, 0]
    finally:
        env.close()


def test_residual_ppo_starts_at_the_reference_and_cuda_checkpoint_has_correct_contract(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Esta comprobación requiere CUDA.")
    config = load_config("configs/formula-residual-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=32),
        circuits=replace(config.circuits, train=(0,)),
    )
    env = make_env(config.environment)
    try:
        model = create_model(config, env)
        observation, _ = env.reset(seed=0)
        initial, _ = model.predict(observation, deterministic=True)
        np.testing.assert_array_equal(initial, [0, 0])
        output = tmp_path / "residual"
        result = train(config, output, smoke=True)
        assert result["actual_timesteps"] == 512 and result["save_load_predictions_identical"]
        trained = load_model(output / "model.zip", config)
        assert trained.device.type == "cuda"
        action, _ = trained.predict(observation, deterministic=True)
        assert env.action_space.contains(action) and np.any(action != initial)
        with pytest.raises(ValueError, match="versión de observaciones"):
            load_model(
                output / "model.zip",
                replace(config, environment=replace(config.environment, control_profile="direct")),
            )
    finally:
        env.close()
