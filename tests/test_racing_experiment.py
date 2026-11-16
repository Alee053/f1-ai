import csv
import hashlib
import json

import numpy as np
import pytest

from f1_ai import racing_experiment
from f1_ai.cli import parser
from f1_ai.config import load_config
from f1_ai.racing_line import ReferenceConfig


def test_saved_map_planning_creates_no_environment_and_preserves_previous_outputs(
    tmp_path, monkeypatch
):
    def forbidden_environment(*args, **kwargs):
        raise AssertionError("La planificación offline no debe crear un entorno.")

    monkeypatch.setattr(racing_experiment, "make_env", forbidden_environment)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    track = tmp_path / "track.json"
    track.write_text(json.dumps((40 * np.column_stack((np.cos(angles), np.sin(angles)))).tolist()))
    output = tmp_path / "plan"
    config = load_config("configs/formula.toml")
    settings = ReferenceConfig(samples=32, optimization_steps=0)
    result = racing_experiment.plan_saved_track(config, track, output, settings)
    assert result["estimated_flying_lap_seconds"] > 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source_sha256"] == hashlib.sha256(track.read_bytes()).hexdigest()
    assert metadata["physical_steps"] == 0 and metadata["training_performed"] is False
    assert metadata["observation_version"] is None
    with pytest.raises(FileExistsError):
        racing_experiment.plan_saved_track(config, track, output, settings)


def test_twelve_step_formula_smoke_exports_real_state_and_never_claims_a_lap(tmp_path):
    output = tmp_path / "smoke"
    result = racing_experiment.run_reference(
        load_config("configs/formula.toml"),
        output,
        seed=0,
        steps=12,
        settings=ReferenceConfig(samples=64, optimization_steps=2),
    )
    assert result["steps"] == 12
    assert not result["lap_complete"] and result["lap_time_seconds"] is None
    assert result["termination_reason"] == "time_limit"
    with (output / "trajectory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert float(rows[-1]["pedal"]) > 0
    assert float(rows[-1]["speed_m_s"]) > float(rows[0]["speed_m_s"])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["information_regime"] == "full_map_and_physical_state"
    assert metadata["neural_policy"] is False
    assert metadata["reward_used_for_control"] is False


def test_reference_development_protects_held_out_tracks_and_rejects_classic(tmp_path):
    with pytest.raises(ValueError, match="entrenamiento"):
        racing_experiment.run_reference(load_config("configs/formula.toml"), tmp_path, seed=200)
    with pytest.raises(ValueError, match="formula-v1"):
        racing_experiment.run_reference(load_config("configs/ppo.toml"), tmp_path)


def test_reference_commands_parse_explicit_units_and_bounded_work():
    args = parser().parse_args(
        ["reference", "--output", "runs/reference-smoke", "--steps", "12", "--max-speed", "40"]
    )
    assert args.steps == 12 and args.max_speed == 40
    args = parser().parse_args(
        ["plan-reference", "--track", "track.json", "--output", "runs/plan", "--half-width", "5"]
    )
    assert args.half_width == 5
