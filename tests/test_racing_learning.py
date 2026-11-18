import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from f1_ai import imitation
from f1_ai.config import load_config
from f1_ai.curriculum import CornerCurriculum, CornerStart
from f1_ai.environment import CircuitPool, make_env, observation_contract
from f1_ai.imitation import validate_dataset
from f1_ai.racing_line import ReferenceConfig
from f1_ai.rewards import FrontLockPenalty, RearSlipPenalty, lap_time_step_reward
from f1_ai.training import load_model, train


def test_planned_policy_controls_steering_and_pedal_directly():
    config = load_config("configs/formula-direct-cuda.toml")
    env = make_env(config.environment)
    try:
        observation, _ = env.reset(seed=0)
        version, names = observation_contract(config.environment)
        assert version == 4 and observation.shape == (len(names),) == (41,)
        assert "target_speed_error" in names and "previous_pedal" not in names
        action = np.array([0.25, 0.3], dtype=np.float32)
        observation, _, _, _, _ = env.step(action)
        np.testing.assert_array_equal(env.previous_action, action)
        assert env.observation_space.contains(observation) and env.elapsed_steps == 1
    finally:
        env.close()


@pytest.mark.parametrize("braking", [False, True])
def test_corner_spawn_keeps_track_and_starts_with_consistent_velocity(braking):
    config = load_config("configs/formula-curriculum-cuda.toml")
    env = make_env(config.environment)
    try:
        _, normal = env.reset(seed=7)
        points = env.geometry.points.copy()
        options = {"corner_start": CornerStart(seed=100, stage=0, braking=braking)}
        observation, sector = env.reset(seed=7, options=options)
        assert normal["speed"] == 0.0 < sector["speed"]
        assert sector["position"] != normal["position"]
        assert sector["simulated_seconds"] == sector["forward_progress"] == 0.0
        assert env.road_contact_fraction == 1.0 and sector["off_track_steps"] == 0
        assert max(abs(value) for value in sector["slip_ratios"]) < 1e-5
        np.testing.assert_array_equal(env.geometry.points, points)
        repeated, _ = env.reset(seed=7, options=options)
        np.testing.assert_array_equal(repeated, observation)
        np.testing.assert_array_equal(env.reset(seed=7)[1]["position"], normal["position"])
    finally:
        env.close()


def test_corner_curriculum_truncates_practice_without_crediting_a_lap():
    config = load_config("configs/formula-curriculum-cuda.toml")
    env = CornerCurriculum(CircuitPool(make_env(config.environment), (7,), shuffle=True), 4)
    try:
        _, info = env.reset(seed=42)
        assert info["training_sector"] and info["curriculum_stage"] == 0
        assert info["circuit_seed"] == 7
        for _ in range(450):
            _, _, terminated, truncated, info = env.step(np.array([0, -0.6], dtype=np.float32))
            if terminated or truncated:
                break
        assert truncated and not terminated and not info["lap_complete"]
        assert info["termination_reason"] == "sector_limit"
        assert info["simulated_seconds"] == 9.0
        env.set_curriculum_timesteps(81_920)
        assert env.reset(seed=42)[1]["curriculum_stage"] == 2
    finally:
        env.close()


def test_racing_observation_exposes_wheel_feedback_and_actual_previous_command():
    config = load_config("configs/formula-racing-cuda.toml")
    env = make_env(config.environment)
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=0)
        version, names = observation_contract(config.environment)
        assert version == 2 and len(names) == len(observation) == 34
        np.testing.assert_array_equal(observation[-2:], [0, 0])
        command = np.array([0.3, 0.2], dtype=np.float32)
        observation, _, _, _, info = env.step(command)
        np.testing.assert_array_equal(observation[-2:], command)
        np.testing.assert_allclose(
            observation[24:28], np.clip(np.array(info["slip_ratios"]) / 2, -1, 1)
        )
        observation, _ = env.reset(seed=0)
        np.testing.assert_array_equal(observation[-2:], [0, 0])
        assert env.surf is None
    finally:
        env.close()


def test_lap_objective_prefers_less_time_for_equal_valid_progress_and_tolerates_useful_slip():
    def total(steps):
        return sum(
            lap_time_step_reward(
                1 / steps,
                on_road=True,
                action_delta=np.zeros(2),
                max_abs_slip=0.15,
                sideslip_rad=0.05,
            )
            for _ in range(steps)
        )

    assert total(2000) > total(2500)
    steady = dict(on_road=True, action_delta=np.zeros(2), max_abs_slip=0.15, sideslip_rad=0.05)
    baseline = lap_time_step_reward(0.001, **steady)
    assert lap_time_step_reward(0.001, **{**steady, "max_abs_slip": 0}) == baseline
    assert lap_time_step_reward(0.001, **{**steady, "max_abs_slip": 1}) < baseline
    assert lap_time_step_reward(0.001, **{**steady, "action_delta": np.ones(2)}) < baseline
    assert lap_time_step_reward(0.001, **{**steady, "sideslip_rad": 0.6}) < baseline
    assert lap_time_step_reward(0.1, **{**steady, "on_road": False}) < 0


def test_training_penalty_only_charges_excess_slip_of_driven_wheels(monkeypatch):
    config = load_config("configs/formula-curriculum-cuda.toml")
    env = make_env(config.environment)
    wrapper = RearSlipPenalty(env, coefficient=0.6)
    try:
        observation, _ = wrapper.reset(seed=7)
        action = np.array([0.8, 0.3], dtype=np.float32)
        info = {"slip_ratios": (0.9, -0.95, 0.2, -0.25)}

        def unchanged_controls(command):
            np.testing.assert_array_equal(command, action)
            return observation, 1.0, False, False, info

        monkeypatch.setattr(env, "step", unchanged_controls)
        assert wrapper.step(action)[1] == 1.0
        info["slip_ratios"] = (0.1, -0.2, 1.0, 0.15)
        _, reward, terminated, truncated, result = wrapper.step(action)
        assert reward == pytest.approx(1.0 - 0.6 * 0.7**2)
        assert result is info and not (terminated or truncated)
    finally:
        wrapper.close()


def test_front_lock_penalty_charges_only_excess_slip_and_front_locks_on_braking(monkeypatch):
    config = load_config("configs/formula-braking-ppo-cuda.toml")
    env = make_env(config.environment)
    wrapper = FrontLockPenalty(env, coefficient=config.ppo.front_lock_penalty)
    try:
        observation, _ = wrapper.reset(seed=7)
        brake = np.array([0.9, -0.24], dtype=np.float32)
        gas = np.array([0.9, 0.24], dtype=np.float32)
        info = {"slip_ratios": (-0.15, -0.2, -0.9, 0.0), "locked_wheels": (0, 0, 1, 0)}

        def unchanged_controls(action):
            assert action is brake or action is gas
            return observation, 1.0, False, False, info

        monkeypatch.setattr(env, "step", unchanged_controls)
        assert wrapper.step(brake)[1] == 1.0
        info["slip_ratios"] = (-0.9, -0.25, 0.0, 0.0)
        info["locked_wheels"] = (1, 0, 0, 0)
        assert wrapper.step(brake)[1] == pytest.approx(1.0 - 0.8 * (0.6**2 + 0.35))
        assert wrapper.step(gas)[1] == 1.0
    finally:
        wrapper.close()


def test_racing_dagger_labels_rate_changes_from_the_students_actual_controls(monkeypatch):
    config = load_config("configs/formula-racing-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=8),
        circuits=replace(config.circuits, train=(0,)),
    )

    class FixedStudent:
        def __init__(self, model):
            pass

        def action(self, observation):
            return np.array([0.4, 0.2], dtype=np.float32)

    monkeypatch.setattr(imitation, "PPOAgent", FixedStudent)
    data, episodes = imitation.collect_dataset(
        config,
        model=object(),
        expert_probability=0,
        teacher_kind="racing",
        reference_settings=ReferenceConfig(samples=64, optimization_steps=2),
    )
    validate_dataset(data, config)
    np.testing.assert_array_equal(
        data["observations"][1:, -2:], np.tile([0.4, 0.2], (7, 1)).astype(np.float32)
    )
    assert np.max(np.abs(data["actions"][1:, 0] - 0.4)) <= 0.075001
    assert np.max(np.abs(data["actions"][1:, 1] - 0.2)) <= 0.080001
    assert len(episodes) == 1 and data["observations"].shape == (8, 34)


def test_new_observation_cannot_load_legacy_datasets():
    config = load_config("configs/formula-racing-cuda.toml")
    data = {
        "observations": np.zeros((2, 16), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "seeds": np.zeros(2, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="n,34"):
        validate_dataset(data, config)


def test_new_formula_profile_trains_and_reloads_on_cuda_without_loading_as_legacy(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Esta comprobación requiere una GPU CUDA.")
    config = load_config("configs/formula-racing-cuda.toml")
    config = replace(config, environment=replace(config.environment, max_steps=32))
    output = tmp_path / "new-profile"
    result = train(config, output, smoke=True)
    assert result["actual_timesteps"] == 512
    assert result["parameters_finite"] and result["save_load_predictions_identical"]
    model = load_model(output / "model.zip", config)
    assert model.device.type == "cuda" and model.observation_space.shape == (34,)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["observation_version"] == 2
    assert metadata["execution"]["model_device"].startswith("cuda")
    with pytest.raises(ValueError, match="versión de observaciones"):
        load_model(
            output / "model.zip",
            replace(config, environment=replace(config.environment, observation_profile="legacy")),
        )
