import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from f1_ai.config import load_config
from f1_ai.environment import make_env
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


def test_plateau_keeps_the_best_incident_free_checkpoint(tmp_path: Path, monkeypatch):
    config = load_config("configs/formula-road-learning-fast-cuda.toml")
    config = replace(
        config,
        environment=replace(config.environment, max_steps=12),
        circuits=replace(config.circuits, validation=(100,)),
        ppo=replace(
            config.ppo,
            device="cpu",
            vec_env="dummy",
            total_timesteps=128,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_every=16,
            corner_curriculum=False,
            launch_curriculum=False,
            road_start_curriculum=False,
        ),
    )
    laps = iter([(49.0, 0), (48.7, 0), (43.0, 5), (48.64, 0)])

    def evaluate(_environment, _agent, circuit_seed, *, agent_name):
        seconds, locks = next(laps)
        return {
            "circuit_seed": circuit_seed,
            "return": 1.0,
            "steps": round(seconds * 50),
            "coverage": 1.0,
            "lap_complete": True,
            "lap_time_seconds": seconds,
            "off_track_steps": 0,
            "off_track_fraction": 0.0,
            "wheel_lock_steps": locks,
        }

    monkeypatch.setattr("f1_ai.callbacks.evaluate_episode", evaluate)
    output = tmp_path / "plateau"
    result = train(config, output, plateau_patience=2)
    assert result["stop_reason"] == "validation_plateau"
    assert result["actual_timesteps"] == 48
    best = json.loads((output / "best_selection.json").read_text())
    assert best["timesteps"] == 16
    selected = load_model(output / "best_model.zip", config)
    assert selected.num_timesteps == 16


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


def test_missing_initial_metadata_fails_before_starting_simulators(tmp_path, monkeypatch):
    config = load_config("configs/ppo.toml")

    def unexpected_environment(_config):
        pytest.fail("Se creó un simulador antes de validar el modelo inicial.")

    monkeypatch.setattr("f1_ai.training.make_training_envs", unexpected_environment)
    output = tmp_path / "training"
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        train(config, output, initial_model=tmp_path / "missing" / "model.zip")
    assert not output.exists()


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


@pytest.mark.parametrize(
    "device,backend", [("cpu", "dummy"), ("cuda", "dummy"), ("cuda", "subproc")]
)
def test_resume_keeps_optimizer_steps_and_aligns_checkpoints(tmp_path: Path, device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Esta comprobación requiere una GPU CUDA.")
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
    initial = tmp_path / "before-resume"
    train(config, initial)
    source = load_model(initial / "model.zip", config)
    optimizer_steps = sorted(
        {float(state["step"]) for state in source.policy.optimizer.state.values()}
    )
    config = replace(
        config, ppo=replace(config.ppo, device=device, vec_env=backend, total_timesteps=96)
    )
    output = tmp_path / "resumed"
    result = train(config, output, resume_model=initial / "model.zip")
    assert result["starting_timesteps"] == 32
    assert result["actual_timesteps"] == 96
    assert result["trained_timesteps"] == 64
    assert result["policy_parameter_delta_l2"] > 0
    assert result["save_load_predictions_identical"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["resume"]["optimizer_state_entries"] > 0
    assert metadata["resume"]["optimizer_steps_at_start"] == optimizer_steps
    assert metadata["resume"]["environment_state_restored"] is False
    assert {torch.device(value).type for value in metadata["execution"]["optimizer_devices"]} == {
        device
    }
    with np.load(output / "validation" / "evaluations.npz", allow_pickle=False) as records:
        np.testing.assert_array_equal(records["timesteps"], [32, 64, 96])
    assert {path.name for path in (output / "checkpoints").glob("*.zip")} == {
        "ppo_32_steps.zip",
        "ppo_64_steps.zip",
        "ppo_96_steps.zip",
    }
    first = load_model(output / "checkpoints" / "ppo_32_steps.zip", config, device=device)
    env = make_env(config.environment)
    try:
        observation, _ = env.reset(seed=config.circuits.train[0])
        np.testing.assert_allclose(
            source.predict(observation, deterministic=True)[0],
            first.predict(observation, deterministic=True)[0],
            atol=1e-6,
        )
    finally:
        env.close()


def test_cuda_request_cannot_silently_fall_back_to_cpu(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config("configs/ppo.toml")
    config = replace(config, ppo=replace(config.ppo, device="cuda"))
    with pytest.raises(RuntimeError, match="solicitó CUDA"):
        train(config, tmp_path / "unavailable", smoke=True)


def test_device_configuration_is_explicit():
    with pytest.raises(ValueError, match="device"):
        replace(load_config("configs/ppo.toml").ppo, device="auto")
    with pytest.raises(ValueError, match="vec_env"):
        replace(load_config("configs/ppo.toml").ppo, vec_env="threads")
