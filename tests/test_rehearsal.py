import json
from dataclasses import replace

import pytest
import torch

from f1_ai.config import load_config
from f1_ai.environment import make_env
from f1_ai.imitation import load_dataset, record_demonstrations
from f1_ai.supervision import ActorRehearsal, RehearsalPPO
from f1_ai.training import create_model, load_model, train


def test_rehearsal_pulls_the_actor_toward_demonstrations_without_changing_critic_or_noise(tmp_path):
    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=16),
        circuits=replace(config.circuits, train=(0,), validation=(100,)),
    )
    dataset = tmp_path / "data"
    record_demonstrations(config, dataset)
    data = load_dataset(dataset, config)
    env = make_env(config.environment)
    try:
        model = create_model(config, env)
        parameters = {
            name: value.clone()
            for name, value in model.policy.state_dict().items()
            if name.startswith(("value_net.", "mlp_extractor.value_net.")) or name == "log_std"
        }
        observations = torch.as_tensor(data["observations"])
        target = torch.as_tensor(data["actions"])
        with torch.no_grad():
            before = torch.mean(
                (model.policy.get_distribution(observations).distribution.mean - target) ** 2
            )
        rehearsal = ActorRehearsal(model, data, batch_size=16, updates=20, learning_rate=1e-3)
        rehearsal.update(model.policy)
        with torch.no_grad():
            after = torch.mean(
                (model.policy.get_distribution(observations).distribution.mean - target) ** 2
            )
        assert after < before / 2
        for name, value in parameters.items():
            torch.testing.assert_close(model.policy.state_dict()[name], value, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rehearsal_checkpoint_excludes_data_and_resume_requires_the_same_dataset(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Esta comprobación requiere CUDA.")
    config = load_config("configs/formula-racing-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=16),
        circuits=replace(config.circuits, train=(0,), validation=(100,)),
        ppo=replace(
            config.ppo,
            total_timesteps=32,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_every=32,
            device=device,
            rehearsal_learning_rate=6e-6,
        ),
    )
    dataset = tmp_path / "data"
    record_demonstrations(config, dataset, teacher_kind="racing")
    before = tmp_path / "before"
    result = train(config, before, rehearsal_dataset=dataset)
    assert result["actual_timesteps"] == 32 and result["save_load_predictions_identical"]
    model = load_model(before / "model.zip", config)
    assert isinstance(model, RehearsalPPO) and not hasattr(model, "actor_rehearsal")
    assert model.device.type == device
    config = replace(config, ppo=replace(config.ppo, total_timesteps=64))
    with pytest.raises(ValueError, match="variante de PPO"):
        train(config, tmp_path / "invalid", resume_model=before / "model.zip")
    result = train(
        config, tmp_path / "resumed", resume_model=before / "model.zip", rehearsal_dataset=dataset
    )
    assert result["actual_timesteps"] == 64 and result["parameters_finite"]
    metadata = json.loads((tmp_path / "resumed/metadata.json").read_text())
    assert metadata["training_variant"] == "ppo_rehearsal"
    assert metadata["resume"]["optimizer_restored"] is True
    assert metadata["rehearsal"]["optimizer_restored"] is False
    assert metadata["rehearsal"]["learning_rate"] == 6e-6
    assert metadata["rehearsal"]["samples"] == 16
