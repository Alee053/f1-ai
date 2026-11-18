"""PPO reproducible en CPU o CUDA, con evaluación separada y reanudación."""

import hashlib
import json
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from gymnasium import Env
from gymnasium.wrappers import TransformReward
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.callbacks import DrivingEvaluation
from f1_ai.config import (
    LAP_TIME_REWARDS,
    RACING_RESIDUAL,
    EnvironmentConfig,
    ExperimentConfig,
    PPOConfig,
)
from f1_ai.curriculum import CornerCurriculum
from f1_ai.environment import CircuitPool, make_env, observation_contract
from f1_ai.rewards import RearSlipPenalty
from f1_ai.supervision import ActorRehearsal, RehearsalPPO


class _TerminationRequested(KeyboardInterrupt):
    """Interrupción solicitada por SIGTERM mientras se ejecuta learn."""


@contextmanager
def _interrupt_on_sigterm():
    # Python solo permite instalar señales en el hilo principal.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_checkpoint(_signum, _frame):
        raise _TerminationRequested

    signal.signal(signal.SIGTERM, request_checkpoint)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def require_device(device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError("El dispositivo debe ser cpu o cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no puede utilizar la GPU.")
    return device


def create_model(config: ExperimentConfig, env: Env | VecEnv, *, rehearsal: bool = False) -> PPO:
    settings = config.ppo
    algorithm = RehearsalPPO if rehearsal else PPO
    model = algorithm(
        "MlpPolicy",
        env,
        seed=settings.seed,
        device=require_device(settings.device),
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
        use_sde=settings.use_sde,
        sde_sample_freq=settings.sde_sample_freq,
        policy_kwargs={
            "net_arch": {"pi": [settings.hidden_size] * 2, "vf": [settings.hidden_size] * 2},
            "log_std_init": settings.log_std_init,
            "squash_output": settings.squash_output,
        },
    )
    if config.environment.control_profile == RACING_RESIDUAL:
        # La política determinista inicial aplica la referencia sin correcciones.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    return model


def make_training_envs(config: ExperimentConfig) -> VecEnv:
    settings = config.ppo

    def training_environment():
        env = make_env(config.environment)
        if settings.rear_slip_penalty:
            env = RearSlipPenalty(env, settings.rear_slip_penalty)
        env = CircuitPool(env, config.circuits.train, shuffle=True)
        if settings.corner_curriculum:
            monitored = Monitor(
                CornerCurriculum(env, settings.n_envs),
                info_keywords=("training_sector", "curriculum_stage"),
            )
        else:
            monitored = Monitor(env)
        return TransformReward(
            monitored,
            lambda reward: float(reward) * settings.reward_scale,
        )

    factories = [training_environment for _ in range(settings.n_envs)]
    # spawn evita heredar un contexto CUDA al crear los procesos de física.
    return (
        SubprocVecEnv(factories, start_method="spawn")
        if settings.vec_env == "subproc"
        else DummyVecEnv(factories)
    )


def train(
    config: ExperimentConfig,
    output: Path,
    *,
    smoke: bool = False,
    initial_model: Path | None = None,
    resume_model: Path | None = None,
    rehearsal_dataset: Path | None = None,
) -> dict:
    if initial_model is not None and resume_model is not None:
        raise ValueError("Elija inicialización de pesos o reanudación, no ambas.")
    if rehearsal_dataset is not None and config.environment.control_profile == RACING_RESIDUAL:
        raise ValueError("El repaso BC usa mandos directos, no correcciones de la trazada.")
    if resume_model is not None:
        if smoke:
            raise ValueError("La prueba smoke no permite reanudar un entrenamiento.")
        source_metadata = read_model_metadata(resume_model)
        if bool(source_metadata.get("rehearsal")) != (rehearsal_dataset is not None):
            raise ValueError(
                "Reanudar conserva la variante de PPO y requiere su dataset de repaso."
            )
        if source_metadata["kind"] != "ppo":
            raise ValueError("La reanudación requiere un checkpoint de entrenamiento PPO.")
        recorded = asdict(PPOConfig(**source_metadata["configuration"]["ppo"]))
        requested = asdict(config.ppo)
        runtime_fields = {"device", "vec_env", "torch_threads", "total_timesteps", "eval_every"}
        if any(
            recorded[key] != value for key, value in requested.items() if key not in runtime_fields
        ):
            raise ValueError(
                "Reanudar conserva los hiperparámetros; para cambiarlos use --initial-model."
            )
    require_device(config.ppo.device)
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

    envs = make_training_envs(config)
    metadata = provenance(config, kind="ppo_smoke" if smoke else "ppo")
    source_path = resume_model if resume_model is not None else initial_model
    if source_path is not None:
        source_metadata = read_model_metadata(source_path)
        metadata["initial_policy"] = {
            "model_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source": str(source_path),
            "kind": source_metadata["kind"],
            "root_kind": source_metadata.get("initial_policy", {}).get(
                "root_kind",
                source_metadata["kind"],
            ),
        }
    metadata["checkpoint_selection"] = (
        "clean_laps, completed_laps, clean_lap_time, coverage, return"
        if config.environment.reward_profile in LAP_TIME_REWARDS
        else "completed_laps, coverage, lap_time, return"
    )
    started = time.perf_counter()
    try:
        starting_timesteps = 0
        if resume_model is not None:
            model = load_model(resume_model, config, device=settings.device, env=envs)
            starting_timesteps = model.num_timesteps
            if not 0 < starting_timesteps < settings.total_timesteps:
                raise ValueError("El presupuesto total debe superar los pasos del checkpoint.")
            if settings.corner_curriculum:
                envs.env_method("set_curriculum_timesteps", starting_timesteps)
            metadata["resume"] = {
                "starting_timesteps": starting_timesteps,
                "optimizer_restored": True,
                "optimizer_state_entries": len(model.policy.optimizer.state),
                "optimizer_steps_at_start": sorted(
                    {float(state["step"]) for state in model.policy.optimizer.state.values()}
                ),
                "environment_state_restored": False,
                "random_generator_state_restored": False,
            }
        else:
            model = create_model(config, envs, rehearsal=rehearsal_dataset is not None)
        if initial_model is not None:
            source = load_model(initial_model, config, device=settings.device)
            if type(source.policy.action_dist) is not type(model.policy.action_dist):
                raise ValueError("La distribución de acciones de la inicialización no coincide.")
            if source.policy.squash_output != model.policy.squash_output:
                raise ValueError("La transformación de acciones de la inicialización no coincide.")
            model.policy.load_state_dict(source.policy.state_dict())
            with torch.no_grad():
                model.policy.log_std.fill_(settings.log_std_init)
            del source
        if rehearsal_dataset is not None:
            from f1_ai.imitation import load_dataset

            data = load_dataset(rehearsal_dataset, config)
            digest = hashlib.sha256((rehearsal_dataset / "dataset.npz").read_bytes()).hexdigest()
            if (
                resume_model is not None
                and digest != source_metadata["rehearsal"]["dataset_sha256"]
            ):
                raise ValueError("El dataset de repaso no coincide con el del checkpoint.")
            model.actor_rehearsal = ActorRehearsal(model, data, seed=settings.seed)
            metadata["training_variant"] = "ppo_rehearsal"
            metadata["rehearsal"] = {
                "dataset": str(rehearsal_dataset),
                "dataset_sha256": digest,
                "samples": len(data["observations"]),
                "batch_size": 256,
                "updates_per_rollout": 4,
                "learning_rate": 1e-4,
                "critic_updated": False,
                "log_std_updated": False,
                "optimizer_restored": False,
                "sampling_rng_restored": False,
            }
        metadata["execution"] = {
            "requested_device": settings.device,
            "vector_environment": type(envs).__name__,
            "model_device": str(model.device),
            "policy_devices": sorted(
                {str(parameter.device) for parameter in model.policy.parameters()}
            ),
            "optimizer_devices": sorted(
                {
                    str(value.device)
                    for state in model.policy.optimizer.state.values()
                    for key, value in state.items()
                    if torch.is_tensor(value) and key != "step"
                }
            ),
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda"
            else None,
        }
        write_json(output / "metadata.json", metadata)
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        before = torch.cat(
            [parameter.detach().flatten().clone() for parameter in model.policy.parameters()]
        )
        callbacks = []
        if not smoke:
            callbacks = [DrivingEvaluation(config, output)]
        try:
            with _interrupt_on_sigterm():
                model.learn(
                    total_timesteps=settings.total_timesteps - starting_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=resume_model is None,
                )
        except KeyboardInterrupt as error:
            model.save(output / "interrupted_model.zip")
            write_json(
                output / "interrupted.json",
                {
                    "actual_timesteps": model.num_timesteps,
                    "wall_seconds": time.perf_counter() - started,
                    "reason": "SIGTERM"
                    if isinstance(error, _TerminationRequested)
                    else "KeyboardInterrupt",
                },
            )
            if isinstance(error, _TerminationRequested):
                raise SystemExit(128 + signal.SIGTERM) from None
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
            restored = PPO.load(model_path, device=model.device)
            actual, _ = restored.predict(observation, deterministic=True)
            np.testing.assert_array_equal(expected, actual)
        finally:
            probe_env.close()
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - started
        results = {
            "kind": "integration_smoke" if smoke else "training",
            "requested_timesteps": settings.total_timesteps,
            "actual_timesteps": model.num_timesteps,
            "starting_timesteps": starting_timesteps,
            "trained_timesteps": model.num_timesteps - starting_timesteps,
            "wall_seconds": elapsed,
            "steps_per_second": (model.num_timesteps - starting_timesteps) / elapsed,
            "policy_parameter_delta_l2": float(torch.linalg.vector_norm(after - before)),
            "parameters_finite": True,
            "save_load_predictions_identical": True,
            "device": str(model.device),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "demonstrates_driving_performance": False,
        }
        write_json(output / "training_results.json", results)
        return results
    finally:
        envs.close()


def read_model_metadata(path: Path) -> dict:
    directory = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def load_model(
    path: Path,
    config: ExperimentConfig,
    *,
    device: str | None = None,
    env: Env | VecEnv | None = None,
) -> PPO:
    """Carga exclusivamente un checkpoint local de confianza con metadatos compatibles."""
    metadata = read_model_metadata(path)
    if metadata["observation_version"] != observation_contract(config.environment)[0]:
        raise ValueError("La versión de observaciones del modelo no coincide.")
    recorded_environment = EnvironmentConfig(**metadata["configuration"]["environment"])
    if asdict(recorded_environment) != asdict(config.environment):
        raise ValueError("La configuración de evaluación no coincide con la del modelo.")
    if metadata["kind"] not in {"ppo", "ppo_smoke", "behavior_cloning", "dagger"}:
        raise ValueError("Los metadatos no corresponden a un entrenamiento de este proyecto.")
    torch.set_num_threads(config.ppo.torch_threads)
    requested_device = config.ppo.device if device is None else device
    algorithm = RehearsalPPO if metadata.get("training_variant") == "ppo_rehearsal" else PPO
    return algorithm.load(path, device=require_device(requested_device), env=env)
