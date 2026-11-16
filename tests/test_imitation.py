from dataclasses import replace

import numpy as np
import pytest
import torch

from f1_ai.agents import PursuitAgent
from f1_ai.config import load_config
from f1_ai.environment import make_env
from f1_ai.imitation import fit_actor, validate_dataset
from f1_ai.training import create_model


def test_validation_circuits_cannot_enter_imitation_dataset():
    config = load_config("configs/ppo.toml")
    data = {
        "observations": np.zeros((2, 16), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "seeds": np.array([config.circuits.train[0], config.circuits.validation[0]]),
    }
    with pytest.raises(ValueError, match="fuera de entrenamiento"):
        validate_dataset(data, config)


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
