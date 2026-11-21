from dataclasses import replace

import pytest
from stable_baselines3 import SAC

from f1_ai.config import load_config
from f1_ai.soft_actor_critic import SAC_SETTINGS, train_sac
from f1_ai.training import load_model


def test_sac_checkpoint_restores_training_progress_and_replay(tmp_path, monkeypatch):
    monkeypatch.setitem(SAC_SETTINGS, "learning_starts", 8)
    monkeypatch.setitem(SAC_SETTINGS, "buffer_size", 128)
    monkeypatch.setitem(SAC_SETTINGS, "gradient_steps", 1)
    preset = load_config("configs/formula-road-learning-cuda.toml")
    config = replace(
        preset,
        environment=replace(preset.environment, max_steps=8),
        circuits=replace(preset.circuits, train=(0,), validation=(100,)),
        ppo=replace(
            preset.ppo,
            device="cpu",
            vec_env="dummy",
            n_envs=1,
            n_steps=16,
            batch_size=16,
            total_timesteps=32,
            eval_every=32,
        ),
    )
    first = tmp_path / "first"
    assert train_sac(config, first)["actual_timesteps"] == 32
    model = load_model(first / "model.zip", config)
    assert isinstance(model, SAC)
    model.load_replay_buffer(first / "replay_buffer.pkl")
    assert model._n_updates > 0
    assert model.replay_buffer.size() == 32
    assert (first / "checkpoints/sac_32_steps.zip").is_file()

    resumed = replace(config, ppo=replace(config.ppo, total_timesteps=48))
    output = tmp_path / "resumed"
    with pytest.raises(ValueError, match="final o interrumpido"):
        train_sac(resumed, output, resume_model=first / "best_model.zip")
    result = train_sac(resumed, output, resume_model=first / "model.zip")
    assert result["starting_timesteps"] == 32 and result["actual_timesteps"] == 48
    updated = load_model(output / "model.zip", config)
    updated.load_replay_buffer(output / "replay_buffer.pkl")
    assert updated._n_updates > model._n_updates
    assert updated.replay_buffer.size() == 48
