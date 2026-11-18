import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from f1_ai.config import load_config
from f1_ai.environment import NumericCarRacing, make_env


def test_evolution_shares_the_road_profile_and_counts_training_interactions(tmp_path, monkeypatch):
    pytest.importorskip("neat")
    from f1_ai import evolution

    config = load_config("configs/formula-road-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=4),
        circuits=replace(config.circuits, train=(0, 1), validation=(100,)),
        ppo=replace(config.ppo, total_timesteps=128, device="cpu"),
    )
    steps = Counter()
    validation_agents = []
    base_step = NumericCarRacing.step
    evaluate = evolution.evaluate_episode

    def count_steps(env, action):
        if action is not None:
            steps[env.np_random_seed] += 1
        return base_step(env, action)

    def reject_plan(*args):
        pytest.fail("NEAT road-v5 no debería consultar el plan.")

    def capture_candidate(env, agent, seed, **kwargs):
        validation_agents.append(agent)
        return evaluate(env, agent, seed, **kwargs)

    monkeypatch.setattr(NumericCarRacing, "step", count_steps)
    monkeypatch.setattr(NumericCarRacing, "_track_plan", reject_plan)
    monkeypatch.setattr(evolution, "evaluate_episode", capture_candidate)
    initial_random = random.getstate()
    output = tmp_path / "neat"
    settings = Path("configs/neat-road.ini")
    result = evolution.train_neat(config, output, settings, rollout_steps=4)
    assert result["actual_timesteps"] == sum(steps[seed] for seed in (0, 1)) == 128
    assert result["generations"] == 2
    assert steps[100] == 8
    assert set(steps).issubset({0, 1, 100})
    assert random.getstate() == initial_random
    selected = json.loads((output / "best_selection.json").read_text())
    agent = evolution.load_neat_agent(output / "best_model.json", config)
    env = make_env(config.environment)
    try:
        obs, _ = env.reset(seed=100)
        action = agent.action(obs)
        assert env.action_space.contains(action)
        np.testing.assert_array_equal(
            action, validation_agents[selected["generation"] - 1].action(obs)
        )
    finally:
        env.close()
    with pytest.raises(ValueError, match="múltiplo"):
        evolution.train_neat(config, tmp_path / "invalid", settings, rollout_steps=3)
    altered = replace(config, environment=replace(config.environment, max_steps=5))
    with pytest.raises(ValueError, match="configuración de evaluación"):
        evolution.load_neat_agent(output / "best_model.json", altered)
