import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from f1_ai.config import load_config
from f1_ai.training import load_model, train


def test_ppo_updates_and_save_load_round_trip(tmp_path: Path):
    config = load_config("configs/ppo.toml")
    output = tmp_path / "smoke"
    previous_handler = signal.getsignal(signal.SIGTERM)
    result = train(config, output, smoke=True)
    assert signal.getsignal(signal.SIGTERM) is previous_handler
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
    checkpoint = next((output / "checkpoints").glob("*.zip"))
    restored = load_model(checkpoint, config)
    assert restored.observation_space.shape == (16,)


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


def test_sigterm_preserves_a_loadable_checkpoint_during_learning(tmp_path: Path):
    output = tmp_path / "terminated"
    script = """
import sys
from dataclasses import replace
from pathlib import Path
from f1_ai.config import load_config
from f1_ai.training import train
config = load_config('configs/ppo.toml')
config = replace(config, environment=replace(config.environment, max_steps=32),
    ppo=replace(config.ppo, total_timesteps=1000000, n_envs=1, n_steps=16,
                batch_size=16, n_epochs=1, eval_every=1000000))
train(config, Path(sys.argv[1]))
"""
    with (tmp_path / "child.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(output)], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 20
            ready = False
            while time.monotonic() < deadline and process.poll() is None:
                progress = output / "progress.csv"
                if progress.exists():
                    with progress.open() as stream:
                        rows = list(csv.DictReader(stream))
                    ready = any(float(row.get("time/total_timesteps") or 0) > 0 for row in rows)
                if ready:
                    break
                time.sleep(0.02)
            assert ready, "El proceso no llegó al bucle de aprendizaje."
            os.kill(process.pid, signal.SIGTERM)
            returncode = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert (output / "interrupted_model.zip").is_file()
    interruption = json.loads((output / "interrupted.json").read_text())
    assert interruption["reason"] == "SIGTERM"
    assert 0 < interruption["actual_timesteps"] < 1000000
    assert returncode == 128 + signal.SIGTERM
    config = load_config("configs/ppo.toml")
    config = replace(config, environment=replace(config.environment, max_steps=32))
    model = load_model(output / "interrupted_model.zip", config)
    assert model.observation_space.shape == (16,)
