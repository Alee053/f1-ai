"""Imitación y agregación de datos usando únicamente circuitos de entrenamiento.

El controlador etiqueta estados durante la recolección. La inferencia del modelo
guardado utiliza exclusivamente la red neuronal. Esta etapa no se llama PPO.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from f1_ai.agents import PPOAgent, PursuitAgent
from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import (
    LAP_TIME_REWARDS,
    RACING_RESIDUAL,
    EnvironmentConfig,
    ExperimentConfig,
    ReferenceDrivingConfig,
)
from f1_ai.environment import make_env, observation_contract
from f1_ai.evaluation import driving_score, evaluate_episode, summarize
from f1_ai.plan_features import track_plan
from f1_ai.racing_controller import FollowerConfig, RacingController, ReferenceState
from f1_ai.racing_line import ReferenceConfig
from f1_ai.supervision import fit_actor as fit_actor
from f1_ai.training import create_model


def validate_dataset(data: dict[str, np.ndarray], config: ExperimentConfig) -> None:
    observations, actions, seeds = data["observations"], data["actions"], data["seeds"]
    n = len(observations)
    width = len(observation_contract(config.environment)[1])
    if n == 0 or observations.shape != (n, width) or actions.shape != (n, 2):
        raise ValueError(f"El dataset debe contener observaciones [n,{width}] y acciones [n,2].")
    if (
        seeds.shape != (n,)
        or not np.issubdtype(seeds.dtype, np.integer)
        or not set(np.unique(seeds)).issubset(config.circuits.train)
    ):
        raise ValueError("Las demostraciones contienen circuitos fuera de entrenamiento.")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Las demostraciones contienen valores no finitos.")
    if np.any(np.abs(observations) > 1) or np.any(np.abs(actions) > 1):
        raise ValueError("Las demostraciones exceden los espacios normalizados.")


def collect_dataset(
    config: ExperimentConfig,
    *,
    model: PPO | None = None,
    expert_probability: float = 1.0,
    target_speed: float = 30.0,
    lateral_acceleration: float = 8.0,
    teacher_kind: str = "pursuit",
    reference_settings: ReferenceConfig | None = None,
    follower_settings: FollowerConfig | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    if config.environment.control_profile == RACING_RESIDUAL:
        raise ValueError("La recolección de etiquetas de imitación requiere mandos directos.")
    if not 0 <= expert_probability <= 1:
        raise ValueError("La probabilidad de usar al experto debe pertenecer a [0, 1].")
    rng = np.random.default_rng(config.ppo.seed)
    if teacher_kind not in {"pursuit", "racing"}:
        raise ValueError("El experto debe ser pursuit o racing.")
    if teacher_kind == "racing" and config.environment.vehicle_model != "formula-v1":
        raise ValueError("El experto racing requiere Formula.")
    reference = config.environment.racing_reference or ReferenceDrivingConfig()
    reference_settings = reference_settings or reference.plan
    follower_settings = follower_settings or reference.follower
    teacher = (
        PursuitAgent(target_speed, lateral_acceleration, environment=config.environment)
        if teacher_kind == "pursuit"
        else None
    )
    student = PPOAgent(model) if model is not None else None
    env = make_env(config.environment)
    observations, actions, seeds = [], [], []
    episodes = []
    try:
        for seed in config.circuits.train:
            observation, _ = env.reset(seed=seed)
            if teacher_kind == "racing":
                teacher = RacingController(
                    track_plan(
                        env.geometry.points.tobytes(),
                        config.environment.formula,
                        reference_settings,
                    ),
                    follower_settings,
                )
            terminated = truncated = False
            max_slip, lock_steps = 0.0, 0
            while not (terminated or truncated):
                if teacher_kind == "racing":
                    expert_action = teacher.action(
                        ReferenceState.from_car(env.car), previous_action=env.previous_action
                    )
                else:
                    expert_action = teacher.action(observation)
                observations.append(observation.copy())
                actions.append(expert_action)
                seeds.append(seed)
                action = expert_action
                if student is not None and rng.random() >= expert_probability:
                    action = student.action(observation)
                observation, _, terminated, truncated, info = env.step(action)
                if config.environment.vehicle_model == "formula-v1":
                    max_slip = max(max_slip, max(abs(value) for value in info["slip_ratios"]))
                    lock_steps += int(any(info["locked_wheels"]))
            episodes.append(
                {
                    "seed": seed,
                    "steps": env.elapsed_steps,
                    "coverage": info["coverage"],
                    "lap_complete": info["lap_complete"],
                    "reason": info["termination_reason"],
                    "off_track_steps": info["off_track_steps"],
                    "max_abs_slip_ratio": max_slip,
                    "wheel_lock_steps": lock_steps,
                }
            )
            print(
                f"[demonstrations] teacher={teacher_kind} seed={seed} "
                f"lap={info['lap_complete']} coverage={info['coverage']:.3f} "
                f"off_track={info['off_track_steps']}",
                flush=True,
            )
    finally:
        env.close()
    data = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    validate_dataset(data, config)
    return data, episodes


def load_dataset(directory: Path, config: ExperimentConfig) -> dict[str, np.ndarray]:
    with (directory / "metadata.json").open() as stream:
        metadata = json.load(stream)
    recorded = EnvironmentConfig(**metadata["configuration"]["environment"])
    if metadata["observation_version"] != observation_contract(config.environment)[0]:
        raise ValueError("La versión de observaciones del dataset no coincide.")
    if recorded != config.environment:
        raise ValueError("El entorno de las demostraciones no coincide con la configuración.")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("observations", "actions", "seeds")}
    validate_dataset(data, config)
    return data


def record_demonstrations(
    config: ExperimentConfig,
    output: Path,
    *,
    target_speed: float = 30.0,
    lateral_acceleration: float = 8.0,
    teacher_kind: str = "pursuit",
) -> dict:
    output = create_directory(output)
    started = time.perf_counter()
    data, episodes = collect_dataset(
        config,
        target_speed=target_speed,
        lateral_acceleration=lateral_acceleration,
        teacher_kind=teacher_kind,
    )
    metadata = provenance(config, kind="demonstrations")
    metadata["teacher"] = {
        "target_speed": target_speed,
        "lateral_acceleration": lateral_acceleration,
        "teacher_kind": teacher_kind,
    }
    if teacher_kind == "racing":
        reference = config.environment.racing_reference or ReferenceDrivingConfig()
        metadata["teacher"].update(
            reference_settings=asdict(reference.plan),
            follower_settings=asdict(reference.follower),
        )
        metadata["teacher_information_regime"] = "full_map_and_physical_state"
    write_json(output / "metadata.json", metadata)
    np.savez_compressed(output / "dataset.npz", **data)
    result = {
        "samples": len(data["observations"]),
        "episodes": episodes,
        "wall_seconds": time.perf_counter() - started,
        "training_seeds_only": True,
    }
    write_json(output / "results.json", result)
    return result


def clone(
    config: ExperimentConfig,
    dataset: Path,
    output: Path,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    dagger_rounds: int = 0,
) -> dict:
    if dagger_rounds < 0:
        raise ValueError("El número de rondas DAgger no puede ser negativo.")
    data = load_dataset(dataset, config)
    output = create_directory(output)
    torch.set_num_threads(config.ppo.torch_threads)
    env = make_env(config.environment)
    started = time.perf_counter()
    try:
        model = create_model(config, env)
        metadata = provenance(config, kind="dagger" if dagger_rounds else "behavior_cloning")
        metadata["imitation"] = {
            "dataset": str(dataset),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dagger_rounds": dagger_rounds,
            "only_training_labels": True,
            "checkpoint_selection": "clean_laps, completed_laps, clean_lap_time, coverage"
            if config.environment.reward_profile in LAP_TIME_REWARDS
            else "completed_laps, coverage, lap_time",
        }
        metadata["execution"] = {
            "model_device": str(model.device),
            "policy_devices": sorted(
                {str(parameter.device) for parameter in model.policy.parameters()}
            ),
            "cuda_device_name": torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda"
            else None,
        }
        write_json(output / "metadata.json", metadata)
        with (dataset / "metadata.json").open() as stream:
            teacher = json.load(stream)["teacher"]
        if teacher.get("teacher_kind") == "racing":
            teacher["reference_settings"] = ReferenceConfig(**teacher["reference_settings"])
            teacher["follower_settings"] = FollowerConfig(**teacher["follower_settings"])
        history = []
        best_score = None
        for round_index in range(dagger_rounds + 1):
            if round_index:
                fresh, _ = collect_dataset(
                    config,
                    model=model,
                    expert_probability=0.5**round_index,
                    **teacher,
                )
                data = {key: np.concatenate((data[key], fresh[key])) for key in data}
                validate_dataset(data, config)
            # Permite continuar y auditar DAgger sin perder sus etiquetas agregadas.
            np.savez_compressed(output / f"dataset-round-{round_index}.npz", **data)
            losses = fit_actor(
                model,
                data,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=config.ppo.seed + round_index,
            )
            model.save(output / f"round-{round_index}.zip")
            episodes = [
                evaluate_episode(config.environment, PPOAgent(model), seed, agent_name="bc")
                for seed in config.circuits.validation
            ]
            summary = summarize(episodes)
            score = driving_score(
                summary, prioritize_clean_time=config.environment.reward_profile in LAP_TIME_REWARDS
            )
            record = {
                "round": round_index,
                "samples": len(data["observations"]),
                "training_mse": losses,
                "validation": summary,
                "episodes": episodes,
            }
            history.append(record)
            write_json(output / f"round-{round_index}.json", record)
            if best_score is None or score > best_score:
                best_score = score
                model.save(output / "best_model.zip")
                (output / "best_selection.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
            print(
                f"[imitation] round={round_index} laps={summary['completed_laps']} "
                f"coverage={summary['mean_coverage']:.3f}",
                flush=True,
            )
        model.save(output / "model.zip")
        result = {
            "kind": "dagger" if dagger_rounds else "behavior_cloning",
            "history": history,
            "wall_seconds": time.perf_counter() - started,
            "rl_interactions": 0,
            "only_training_labels": True,
            "environment": asdict(config.environment),
        }
        write_json(output / "imitation_results.json", result)
        return result
    finally:
        env.close()
