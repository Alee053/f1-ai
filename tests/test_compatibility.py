import json
from dataclasses import replace
from pathlib import Path

import pytest

from f1_ai.config import load_config
from f1_ai.training import load_model, train


def test_old_checkpoints_default_to_legacy_action_semantics(tmp_path: Path):
    config = load_config("configs/ppo.toml")
    output = tmp_path / "legacy"
    train(config, output, smoke=True)
    path = output / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["configuration"]["environment"]["steering_scale"]
    del metadata["configuration"]["environment"]["reward_profile"]
    path.write_text(json.dumps(metadata))
    load_model(output / "model.zip", config)
    with pytest.raises(ValueError, match="configuración"):
        load_model(
            output / "model.zip",
            replace(
                config,
                environment=replace(config.environment, steering_scale=0.4),
            ),
        )


def test_formula_checkpoint_records_physics_and_rejects_changed_downforce(tmp_path: Path):
    config = load_config("configs/formula.toml")
    output = tmp_path / "formula"
    train(config, output, smoke=True)
    load_model(output / "model.zip", config)
    changed = replace(
        config,
        environment=replace(
            config.environment, formula=replace(config.environment.formula, downforce_cla_m2=0)
        ),
    )
    with pytest.raises(ValueError, match="configuración"):
        load_model(output / "model.zip", changed)
    classic = replace(
        config, environment=load_config("configs/ppo-imitation-conservative.toml").environment
    )
    with pytest.raises(ValueError, match="configuración"):
        load_model(output / "model.zip", classic)
