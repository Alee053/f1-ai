"""SAC con el mismo simulador y particiones que PPO."""

import signal
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.callbacks import DrivingEvaluation
from f1_ai.config import ROAD_OBSERVATIONS, ExperimentConfig, PPOConfig
from f1_ai.training import (
    _interrupt_on_sigterm,
    _TerminationRequested,
    load_model,
    make_training_envs,
    read_model_metadata,
    require_device,
)

SAC_SETTINGS = {
    "learning_rate": 0.0003,
    "buffer_size": 400_000,
    "learning_starts": 8_192,
    "tau": 0.005,
    "train_freq": (4, "step"),
    "gradient_steps": 8,
    "ent_coef": "auto_0.05",
    "target_entropy": -3.3,
    "use_sde": True,
    "use_sde_at_warmup": True,
    "sde_sample_freq": 8,
    "policy_kwargs": {"net_arch": [128, 128], "log_std_init": -2.4},
}


def train_sac(config: ExperimentConfig, output: Path, *, resume_model: Path | None = None) -> dict:
    settings = config.ppo
    if settings.pedal_only:
        raise ValueError("pedal_only requiere entrenamiento PPO.")
    if (
        config.environment.observation_profile not in ROAD_OBSERVATIONS
        or config.environment.control_profile != "direct"
        or settings.time_progress_mix
        or (settings.corner_curriculum and not settings.road_start_curriculum)
    ):
        raise ValueError("SAC requiere un perfil vial directo sin trazada de referencia.")
    require_device(settings.device)
    metadata = provenance(config, kind="sac")
    metadata["sac"] = {**SAC_SETTINGS, "batch_size": settings.batch_size, "gamma": settings.gamma}
    buffer_path = None
    if resume_model is not None:
        source = read_model_metadata(resume_model)
        if source["kind"] != "sac" or resume_model.name not in {
            "model.zip",
            "interrupted_model.zip",
        }:
            raise ValueError("Use el modelo SAC final o interrumpido, con su búfer de esa corrida.")
        buffer_path = resume_model.parent / "replay_buffer.pkl"
        if not buffer_path.is_file():
            raise ValueError("La reanudación de SAC requiere el búfer de la misma corrida.")
        recorded = asdict(PPOConfig(**source["configuration"]["ppo"]))
        runtime_fields = {"device", "vec_env", "torch_threads", "total_timesteps", "eval_every"}
        if (
            any(
                recorded[key] != value
                for key, value in asdict(settings).items()
                if key not in runtime_fields
            )
            or tuple(source["configuration"]["circuits"]["train"]) != config.circuits.train
        ):
            raise ValueError(
                "La reanudación de SAC conserva los parámetros y pistas de entrenamiento."
            )
        metadata["sac"] = source["sac"]

    torch.set_num_threads(settings.torch_threads)
    with closing(make_training_envs(config)) as env:
        if resume_model is None:
            model = SAC(
                "MlpPolicy", env, device=settings.device, seed=settings.seed, **metadata["sac"]
            )
        else:
            model = load_model(resume_model, config, env=env)
            model.load_replay_buffer(buffer_path)
        starting_timesteps = model.num_timesteps
        if not starting_timesteps < settings.total_timesteps:
            raise ValueError("El presupuesto total debe superar los pasos del checkpoint.")
        if resume_model is not None:
            if settings.corner_curriculum:
                env.env_method("set_curriculum_timesteps", starting_timesteps)
            metadata["resume"] = {
                "source": str(resume_model),
                "starting_timesteps": starting_timesteps,
                "replay_restored": True,
            }
        output = create_directory(output)
        write_json(output / "metadata.json", metadata)
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        started = time.perf_counter()
        try:
            with _interrupt_on_sigterm():
                model.learn(
                    total_timesteps=settings.total_timesteps - starting_timesteps,
                    reset_num_timesteps=resume_model is None,
                    callback=DrivingEvaluation(config, output, agent_name="sac"),
                    log_interval=25,
                )
        except KeyboardInterrupt as error:
            model.save(output / "interrupted_model.zip")
            model.save_replay_buffer(output / "replay_buffer.pkl")
            if isinstance(error, _TerminationRequested):
                raise SystemExit(128 + signal.SIGTERM) from None
            raise
        model.save(output / "model.zip")
        model.save_replay_buffer(output / "replay_buffer.pkl")
        result = {
            "actual_timesteps": model.num_timesteps,
            "starting_timesteps": starting_timesteps,
            "wall_seconds": time.perf_counter() - started,
            "device": str(model.device),
        }
        write_json(output / "training_results.json", result)
        return result
