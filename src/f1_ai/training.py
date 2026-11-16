"""PPO reproducible en CPU, con evaluación separada y una prueba breve de integración."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from gymnasium.wrappers import TransformReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.callbacks import DrivingEvaluation
from f1_ai.config import EnvironmentConfig, ExperimentConfig
from f1_ai.environment import OBSERVATION_VERSION, CircuitPool, make_env


def create_model(config: ExperimentConfig, env: Env | VecEnv) -> PPO:
    settings = config.ppo
    return PPO(
        "MlpPolicy",
        env,
        seed=settings.seed,
        device="cpu",
        verbose=0,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        n_epochs=settings.n_epochs,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        ent_coef=settings.ent_coef,
        clip_range=settings.clip_range,
        target_kl=settings.target_kl,
        policy_kwargs={
            "net_arch": {"pi": [settings.hidden_size] * 2, "vf": [settings.hidden_size] * 2},
            "log_std_init": settings.log_std_init,
        },
    )


def train(
    config: ExperimentConfig,
    output: Path,
    *,
    smoke: bool = False,
    initial_model: Path | None = None,
) -> dict:
    if smoke:
        config = replace(
            config,
            ppo=replace(
                config.ppo,
                total_timesteps=512,
                n_envs=1,
                n_steps=128,
                batch_size=64,
                n_epochs=2,
                eval_every=1024,
            ),
        )
    output = create_directory(output)
    settings = config.ppo
    torch.set_num_threads(settings.torch_threads)
    envs = DummyVecEnv(
        [
            lambda: TransformReward(
                Monitor(
                    CircuitPool(
                        make_env(config.environment),
                        config.circuits.train,
                        shuffle=True,
                    )
                ),
                lambda reward: float(reward) * settings.reward_scale,
            )
            for _ in range(settings.n_envs)
        ]
    )
    metadata = provenance(config, kind="ppo_smoke" if smoke else "ppo")
    if initial_model is not None:
        metadata["initial_policy"] = {
            "model_sha256": hashlib.sha256(initial_model.read_bytes()).hexdigest(),
            "source": str(initial_model),
        }
    metadata["checkpoint_selection"] = "completed_laps, coverage, lap_time, return"
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        model = create_model(config, envs)
        if initial_model is not None:
            source = load_model(initial_model, config)
            model.policy.load_state_dict(source.policy.state_dict())
            with torch.no_grad():
                model.policy.log_std.fill_(settings.log_std_init)
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        before = torch.cat(
            [parameter.detach().flatten().clone() for parameter in model.policy.parameters()]
        )
        callbacks = []
        if not smoke:
            callbacks = [
                DrivingEvaluation(config, output),
                CheckpointCallback(
                    save_freq=max(1, settings.eval_every // settings.n_envs),
                    save_path=str(output / "checkpoints"),
                    name_prefix="ppo",
                ),
            ]
        try:
            model.learn(total_timesteps=settings.total_timesteps, callback=callbacks)
        except KeyboardInterrupt:
            model.save(output / "interrupted_model.zip")
            write_json(
                output / "interrupted.json",
                {
                    "actual_timesteps": model.num_timesteps,
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            raise
        model_path = output / "model.zip"
        model.save(model_path)
        after = torch.cat([parameter.detach().flatten() for parameter in model.policy.parameters()])
        if not torch.isfinite(after).all():
            raise FloatingPointError("PPO produjo parámetros no finitos.")
        probe_env = make_env(config.environment)
        try:
            observation, _ = probe_env.reset(seed=config.circuits.train[0])
            expected, _ = model.predict(observation, deterministic=True)
            restored = PPO.load(model_path, device="cpu")
            actual, _ = restored.predict(observation, deterministic=True)
            np.testing.assert_array_equal(expected, actual)
        finally:
            probe_env.close()
        elapsed = time.perf_counter() - started
        results = {
            "kind": "integration_smoke" if smoke else "training",
            "requested_timesteps": settings.total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "wall_seconds": elapsed,
            "steps_per_second": model.num_timesteps / elapsed,
            "policy_parameter_delta_l2": float(torch.linalg.vector_norm(after - before)),
            "parameters_finite": True,
            "save_load_predictions_identical": True,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "demonstrates_driving_performance": False,
        }
        write_json(output / "training_results.json", results)
        return results
    finally:
        envs.close()


def load_model(path: Path, config: ExperimentConfig) -> PPO:
    """Carga exclusivamente un checkpoint local de confianza con metadatos compatibles."""
    from dataclasses import asdict

    with (path.parent / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["observation_version"] != OBSERVATION_VERSION:
        raise ValueError("La versión de observaciones del modelo no coincide.")
    recorded_environment = EnvironmentConfig(**metadata["configuration"]["environment"])
    if asdict(recorded_environment) != asdict(config.environment):
        raise ValueError("La configuración de evaluación no coincide con la del modelo.")
    if metadata["kind"] not in {"ppo", "ppo_smoke", "behavior_cloning", "dagger"}:
        raise ValueError("Los metadatos no corresponden a un entrenamiento de este proyecto.")
    torch.set_num_threads(config.ppo.torch_threads)
    return PPO.load(path, device="cpu")
