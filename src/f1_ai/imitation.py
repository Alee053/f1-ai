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
from torch.nn import functional as F

from f1_ai.agents import PPOAgent, PursuitAgent
from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import EnvironmentConfig, ExperimentConfig
from f1_ai.environment import make_env
from f1_ai.evaluation import evaluate_episode, summarize
from f1_ai.training import create_model


def validate_dataset(data: dict[str, np.ndarray], config: ExperimentConfig) -> None:
    observations, actions, seeds = data["observations"], data["actions"], data["seeds"]
    n = len(observations)
    if n == 0 or observations.shape != (n, 16) or actions.shape != (n, 2):
        raise ValueError("El dataset debe contener observaciones [n,16] y acciones [n,2].")
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
) -> tuple[dict[str, np.ndarray], list[dict]]:
    if not 0 <= expert_probability <= 1:
        raise ValueError("La probabilidad de usar al experto debe pertenecer a [0, 1].")
    rng = np.random.default_rng(config.ppo.seed)
    teacher = PursuitAgent(target_speed, lateral_acceleration)
    student = PPOAgent(model) if model is not None else None
    env = make_env(config.environment)
    observations, actions, seeds = [], [], []
    episodes = []
    try:
        for seed in config.circuits.train:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                expert_action = teacher.action(observation)
                observations.append(observation.copy())
                actions.append(expert_action)
                seeds.append(seed)
                action = expert_action
                if student is not None and rng.random() >= expert_probability:
                    action = student.action(observation)
                observation, _, terminated, truncated, info = env.step(action)
            episodes.append(
                {
                    "seed": seed,
                    "steps": env.elapsed_steps,
                    "coverage": info["coverage"],
                    "lap_complete": info["lap_complete"],
                    "reason": info["termination_reason"],
                }
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
) -> dict:
    output = create_directory(output)
    started = time.perf_counter()
    data, episodes = collect_dataset(
        config,
        target_speed=target_speed,
        lateral_acceleration=lateral_acceleration,
    )
    metadata = provenance(config, kind="demonstrations")
    metadata["teacher"] = {
        "target_speed": target_speed,
        "lateral_acceleration": lateral_acceleration,
    }
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


def fit_actor(
    model: PPO,
    data: dict[str, np.ndarray],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    batch_size: int = 256,
) -> list[float]:
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate < 1:
        raise ValueError("La configuración de ajuste supervisado no es válida.")
    observations = torch.as_tensor(data["observations"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    parameters = [
        *model.policy.mlp_extractor.policy_net.parameters(),
        *model.policy.action_net.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        order = torch.randperm(len(observations), generator=generator)
        total = 0.0
        for indices in order.split(batch_size):
            mean = model.policy.get_distribution(observations[indices]).distribution.mean
            loss = F.mse_loss(mean, actions[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(observations))
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"[imitation] epoch={epoch + 1} mse={losses[-1]:.6f}", flush=True)
    return losses


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
        }
        write_json(output / "metadata.json", metadata)
        with (dataset / "metadata.json").open() as stream:
            teacher = json.load(stream)["teacher"]
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
            score = (
                summary["completed_laps"],
                summary["mean_coverage"],
                -(summary["mean_lap_time_seconds"] or float("inf")),
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
