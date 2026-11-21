from dataclasses import replace

import numpy as np
import pytest
import torch

from f1_ai.agents import PPOAgent, PursuitAgent
from f1_ai.artifacts import provenance, write_json
from f1_ai.config import load_config
from f1_ai.environment import NumericCarRacing, make_env
from f1_ai.imitation import (
    calibrate_launch,
    clone,
    fit_actor,
    load_dataset,
    record_demonstrations,
    validate_dataset,
)
from f1_ai.training import create_model, load_model


def test_validation_circuits_cannot_enter_imitation_dataset():
    config = load_config("configs/ppo.toml")
    data = {
        "observations": np.zeros((2, 16), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "seeds": np.array([config.circuits.train[0], config.circuits.validation[0]]),
    }
    with pytest.raises(ValueError, match="fuera de entrenamiento"):
        validate_dataset(data, config)


def test_dagger_retains_the_initial_and_aggregated_labels(tmp_path):
    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=4),
        circuits=replace(config.circuits, train=(0,), validation=(100,)),
    )
    dataset, output = tmp_path / "dataset", tmp_path / "imitation"
    record_demonstrations(config, dataset)
    clone(config, dataset, output, epochs=1, dagger_rounds=1)
    with np.load(output / "dataset-round-0.npz", allow_pickle=False) as initial:
        with np.load(output / "dataset-round-1.npz", allow_pickle=False) as aggregated:
            assert initial["observations"].shape == (4, 16)
            assert aggregated["observations"].shape == (8, 16)
            np.testing.assert_array_equal(aggregated["observations"][:4], initial["observations"])
            validate_dataset(dict(aggregated), config)


def test_road_student_learns_from_a_corridor_checkpoint_without_reading_its_plan(
    tmp_path, monkeypatch
):
    config = load_config("configs/formula-road-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=4),
        circuits=replace(config.circuits, train=(0,), validation=(100,)),
        ppo=replace(config.ppo, device="cpu"),
    )
    teacher_config = replace(
        config, environment=replace(config.environment, observation_profile="corridor-v4")
    )
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    checkpoint = teacher_dir / "model.zip"
    real_plan = NumericCarRacing._track_plan

    def plan_only_for_teacher(env, settings):
        assert env.config.observation_profile == "corridor-v4"
        return real_plan(env, settings)

    monkeypatch.setattr(NumericCarRacing, "_track_plan", plan_only_for_teacher)
    teacher_env = make_env(teacher_config.environment)
    road_env = make_env(config.environment)
    try:
        teacher = create_model(teacher_config, teacher_env)
        teacher.save(checkpoint)
        write_json(teacher_dir / "metadata.json", provenance(teacher_config, kind="ppo"))
        guided, _ = teacher_env.reset(seed=0)
        road, _ = road_env.reset(seed=0)
        dataset, output = tmp_path / "dataset", tmp_path / "student"
        record_demonstrations(config, dataset, teacher_kind="ppo", teacher_model=checkpoint)
        data = load_dataset(dataset, config)
        np.testing.assert_array_equal(data["observations"][0], road)
        np.testing.assert_array_equal(data["actions"][0], PPOAgent(teacher).action(guided))
        with pytest.raises(ValueError, match="--teacher-model"):
            clone(config, dataset, output, epochs=1, dagger_rounds=1)
        clone(config, dataset, output, epochs=1, dagger_rounds=1, teacher_model=checkpoint)
        with np.load(output / "dataset-round-1.npz", allow_pickle=False) as aggregate:
            assert aggregate["observations"].shape == (8, 36)
            assert set(aggregate["seeds"]) == {0}
        student = load_model(output / "best_model.zip", config)
        assert road_env.action_space.contains(PPOAgent(student).action(road))
    finally:
        road_env.close()
        teacher_env.close()


def test_launch_calibration_only_changes_the_pedal_head(tmp_path):
    config = load_config("configs/formula-road-cuda.toml")
    config = replace(
        config,
        circuits=replace(config.circuits, train=(0,), validation=(100,)),
        ppo=replace(config.ppo, device="cpu"),
    )
    source, dataset, output = (tmp_path / name for name in ("actor", "data", "calibrated"))
    source.mkdir()
    dataset.mkdir()
    checkpoint = source / "model.zip"
    env = make_env(config.environment)
    try:
        model = create_model(config, env)
        with torch.no_grad():
            model.policy.action_net.bias[1].fill_(-0.1)
        initial, _ = env.reset(seed=0)
        observation = initial
        action = np.array([0.0, 0.3], dtype=np.float32)
        observations = []
        for _ in range(100):
            observations.append(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
        write_json(dataset / "metadata.json", provenance(config, kind="demonstrations"))
        np.savez_compressed(
            dataset / "dataset.npz",
            observations=np.asarray(observations),
            actions=np.repeat(action[None, :], 100, axis=0),
            seeds=np.zeros(100, dtype=np.int64),
        )
        model.save(checkpoint)
        write_json(source / "metadata.json", provenance(config, kind="dagger"))

        result = calibrate_launch(config, checkpoint, dataset, output)
        restored = load_model(output / "model.zip", config)
        before, after = PPOAgent(model).action(initial), PPOAgent(restored).action(initial)
        assert 0 < result["launch_samples"] < len(observations)
        assert after[1] > before[1]
        for name, weights in model.policy.state_dict().items():
            calibrated = restored.policy.state_dict()[name]
            if name.startswith("action_net."):
                weights, calibrated = weights[:1], calibrated[:1]
            torch.testing.assert_close(weights, calibrated, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_learns_real_demonstrations_without_changing_critic(bounded, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Esta comprobación requiere una GPU CUDA.")
    config = load_config("configs/ppo.toml")
    config = replace(
        config, ppo=replace(config.ppo, use_sde=bounded, squash_output=bounded, device=device)
    )
    env = make_env(config.environment)
    torch.set_num_threads(1)
    try:
        model = create_model(config, env)
        observation, _ = env.reset(seed=config.circuits.train[0])
        expert = PursuitAgent()
        observations, actions = [], []
        for _ in range(128):
            action = expert.action(observation)
            observations.append(observation)
            actions.append(action)
            observation, _, _, _, _ = env.step(action)
        data = {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "seeds": np.zeros(128, dtype=np.int64),
        }
        critic = {
            key: value.clone()
            for key, value in model.policy.state_dict().items()
            if key.startswith(("value_net.", "mlp_extractor.value_net."))
        }
        std = model.policy.log_std.detach().clone()
        losses = fit_actor(model, data, epochs=30, learning_rate=1e-3, seed=42, batch_size=32)
        assert losses[-1] < losses[0] / 3
        for key, value in critic.items():
            torch.testing.assert_close(model.policy.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(model.policy.log_std, std, rtol=0, atol=0)
    finally:
        env.close()
