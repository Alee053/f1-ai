"""Imitación y agregación de datos usando únicamente circuitos de entrenamiento.

El controlador etiqueta estados durante la recolección. La inferencia del modelo
guardado utiliza exclusivamente la red neuronal. Esta etapa no se llama PPO.
"""

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from f1_ai.agents import PPOAgent, PursuitAgent
from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import (
    CORRIDOR_OBSERVATION,
    LAP_TIME_REWARDS,
    RACING_RESIDUAL,
    ROAD_OBSERVATION,
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
from f1_ai.training import create_model, load_model, read_model_metadata


def _teacher_environment(environment: EnvironmentConfig) -> EnvironmentConfig:
    if environment.control_profile == RACING_RESIDUAL or (
        environment.observation_profile not in {CORRIDOR_OBSERVATION, ROAD_OBSERVATION}
    ):
        raise ValueError("El maestro PPO requiere mandos directos con corridor-v4 o road-v5.")
    return replace(environment, observation_profile=CORRIDOR_OBSERVATION)


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
    checkpoint_teacher: PPO | None = None,
    reference_settings: ReferenceConfig | None = None,
    follower_settings: FollowerConfig | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    if config.environment.control_profile == RACING_RESIDUAL:
        raise ValueError("La recolección de etiquetas de imitación requiere mandos directos.")
    if not 0 <= expert_probability <= 1:
        raise ValueError("La probabilidad de usar al experto debe pertenecer a [0, 1].")
    rng = np.random.default_rng(config.ppo.seed)
    if teacher_kind not in {"pursuit", "racing", "ppo"}:
        raise ValueError("El experto debe ser pursuit, racing o ppo.")
    if (teacher_kind == "ppo") != (checkpoint_teacher is not None):
        raise ValueError("El maestro PPO requiere su checkpoint de confianza.")
    if teacher_kind == "racing" and config.environment.vehicle_model != "formula-v1":
        raise ValueError("El experto racing requiere Formula.")
    reference = config.environment.racing_reference or ReferenceDrivingConfig()
    reference_settings = reference_settings or reference.plan
    follower_settings = follower_settings or reference.follower
    environment = config.environment
    projection = None
    if teacher_kind == "pursuit":
        teacher = PursuitAgent(target_speed, lateral_acceleration, environment=environment)
    elif teacher_kind == "ppo":
        teacher = PPOAgent(checkpoint_teacher)
        environment = _teacher_environment(environment)
        if environment != config.environment:
            _, source_names = observation_contract(environment)
            _, student_names = observation_contract(config.environment)
            projection = [source_names.index(name) for name in student_names]
    student = PPOAgent(model) if model is not None else None
    env = make_env(environment)
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
                measured = observation if projection is None else observation[projection]
                observations.append(measured.copy())
                actions.append(expert_action)
                seeds.append(seed)
                action = expert_action
                if student is not None and rng.random() >= expert_probability:
                    action = student.action(measured)
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
    remove_reference = (
        config.environment.observation_profile == ROAD_OBSERVATION
        and recorded.observation_profile == CORRIDOR_OBSERVATION
    )
    source = (
        replace(
            config,
            environment=replace(config.environment, observation_profile=CORRIDOR_OBSERVATION),
        )
        if remove_reference
        else config
    )
    version, source_names = observation_contract(source.environment)
    if metadata["observation_version"] != version:
        raise ValueError("La versión de observaciones del dataset no coincide.")
    if recorded != source.environment:
        raise ValueError("El entorno de las demostraciones no coincide con la configuración.")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("observations", "actions", "seeds")}
    validate_dataset(data, source)
    if remove_reference:
        _, names = observation_contract(config.environment)
        data["observations"] = data["observations"][:, [source_names.index(name) for name in names]]
        validate_dataset(data, config)
    return data


def record_demonstrations(
    config: ExperimentConfig,
    output: Path,
    *,
    target_speed: float = 30.0,
    lateral_acceleration: float = 8.0,
    teacher_kind: str = "pursuit",
    teacher_model: Path | None = None,
) -> dict:
    if (teacher_kind == "ppo") != (teacher_model is not None):
        raise ValueError("--teacher-model se requiere únicamente con --teacher ppo.")
    checkpoint = None
    if teacher_model is not None:
        source = replace(config, environment=_teacher_environment(config.environment))
        checkpoint = load_model(teacher_model, source, device="cpu")
    output = create_directory(output)
    started = time.perf_counter()
    data, episodes = collect_dataset(
        config,
        target_speed=target_speed,
        lateral_acceleration=lateral_acceleration,
        teacher_kind=teacher_kind,
        checkpoint_teacher=checkpoint,
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
    elif teacher_model is not None:
        metadata["teacher"] = {
            "teacher_kind": "ppo",
            "model_sha256": hashlib.sha256(teacher_model.read_bytes()).hexdigest(),
        }
        metadata["teacher_information_regime"] = CORRIDOR_OBSERVATION
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
    teacher_model: Path | None = None,
) -> dict:
    if dagger_rounds < 0:
        raise ValueError("El número de rondas DAgger no puede ser negativo.")
    data = load_dataset(dataset, config)
    with (dataset / "metadata.json").open() as stream:
        teacher = json.load(stream)["teacher"]
    teacher_kind = teacher.get("teacher_kind", "pursuit")
    requires_teacher = teacher_kind == "ppo" and dagger_rounds > 0
    if requires_teacher != (teacher_model is not None):
        raise ValueError("--teacher-model se requiere para DAgger con un maestro PPO.")
    if teacher_model is not None:
        if hashlib.sha256(teacher_model.read_bytes()).hexdigest() != teacher["model_sha256"]:
            raise ValueError("El maestro PPO no coincide con el del dataset.")
        source = replace(config, environment=_teacher_environment(config.environment))
        teacher = {
            "teacher_kind": "ppo",
            "checkpoint_teacher": load_model(teacher_model, source, device="cpu"),
        }
    elif teacher_kind == "racing":
        teacher["reference_settings"] = ReferenceConfig(**teacher["reference_settings"])
        teacher["follower_settings"] = FollowerConfig(**teacher["follower_settings"])
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
            "teacher_kind": teacher_kind,
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


def calibrate_launch(
    config: ExperimentConfig, checkpoint: Path, dataset: Path, output: Path
) -> dict:
    """Ajusta la última salida de pedal con estados de arranque de train."""
    source_kind = read_model_metadata(checkpoint)["kind"]
    if (
        source_kind not in {"behavior_cloning", "dagger"}
        or config.environment.observation_profile != ROAD_OBSERVATION
        or config.environment.control_profile != "direct"
    ):
        raise ValueError("El ajuste de arranque requiere un alumno de imitación road-v5.")
    observations = load_dataset(dataset, config)["observations"]
    speed_limit = 1.8
    pedal_floor = 0.08
    launch_weight = 10.0
    ridge = 0.00001
    speed = np.abs(observations[:, 0]) * config.environment.forward_speed_scale
    launch = speed < speed_limit
    if not launch.any() or launch.all():
        raise ValueError("Se requieren estados de arranque y de marcha en el dataset.")
    model = load_model(checkpoint, config, device="cpu")
    policy = model.policy
    with torch.no_grad():
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(tensor))
        features = np.column_stack((latent.numpy().astype(np.float64), np.ones(len(tensor))))
        logits = policy.action_net(latent)[:, 1].numpy()
    minimum = pedal_floor * (1.0 - speed[launch] / speed_limit)
    if policy.squash_output:
        minimum = np.arctanh(minimum)
    desired_delta = np.maximum(minimum - logits[launch], 0.0)
    stopped, moving = features[launch], features[~launch]
    gram = moving.T @ moving / len(moving) + launch_weight * stopped.T @ stopped / len(stopped)
    rhs = launch_weight * stopped.T @ desired_delta / len(stopped)
    delta = np.linalg.solve(gram + ridge * np.eye(features.shape[1]), rhs)
    with torch.no_grad():
        weights = policy.action_net.weight[1]
        weights.add_(torch.as_tensor(delta[:-1], dtype=weights.dtype))
        policy.action_net.bias[1].add_(float(delta[-1]))

    results = {
        "launch_samples": len(stopped),
        "moving_samples": len(moving),
        "launch_speed_m_s": speed_limit,
        "pedal_floor": pedal_floor,
        "launch_weight": launch_weight,
        "ridge": ridge,
        "moving_logit_rmse": float(np.linalg.norm(moving @ delta) / np.sqrt(len(moving))),
    }
    metadata = provenance(config, kind=source_kind)
    metadata["launch_calibration"] = {
        **results,
        "source_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((dataset / "dataset.npz").read_bytes()).hexdigest(),
    }
    output = create_directory(output)
    write_json(output / "metadata.json", metadata)
    model.save(output / "model.zip")
    return results
