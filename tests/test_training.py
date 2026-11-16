import json
from dataclasses import replace
from pathlib import Path

from f1_ai.config import load_config
from f1_ai.training import load_model, train


def test_ppo_updates_and_save_load_round_trip(tmp_path: Path):
    config = load_config("configs/ppo.toml")
    output = tmp_path / "smoke"
    result = train(config, output, smoke=True)
    assert result["actual_timesteps"] == 512
    assert result["policy_parameter_delta_l2"] > 0
    assert result["parameters_finite"]
    assert result["save_load_predictions_identical"]
    assert result["demonstrates_driving_performance"] is False
    model = load_model(output / "model.zip", config)
    assert model.observation_space.shape == (16,)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["kind"] == "ppo_smoke"


def test_vectorized_training_saves_validation_and_checkpoint(tmp_path: Path):
    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=32),
        circuits=replace(config.circuits, validation=(100, 101)),
        ppo=replace(
            config.ppo,
            total_timesteps=32,
            n_envs=2,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_every=32,
        ),
    )
    output = tmp_path / "callbacks"
    result = train(config, output)
    assert result["actual_timesteps"] == 32
    assert (output / "best_model.zip").is_file()
    assert (output / "validation" / "evaluations.npz").is_file()
    assert list((output / "checkpoints").glob("*.zip"))


def test_warm_start_records_source_and_updates_weights(tmp_path: Path):
    config = load_config("configs/ppo.toml")
    initial = tmp_path / "initial"
    target = tmp_path / "target"
    train(config, initial, smoke=True)
    result = train(config, target, smoke=True, initial_model=initial / "model.zip")
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["initial_policy"]["root_kind"] == "ppo_smoke"
    assert result["policy_parameter_delta_l2"] > 0
    assert result["save_load_predictions_identical"]
