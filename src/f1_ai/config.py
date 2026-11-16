"""Configuración explícita y separación de circuitos antes de los experimentos."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int = 3000
    off_track_grace_steps: int = 75
    minimum_coverage: float = 0.95

    def __post_init__(self):
        if self.max_steps <= 0 or self.off_track_grace_steps <= 0:
            raise ValueError("Los límites de pasos deben ser positivos.")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("La cobertura mínima debe pertenecer a (0, 1].")


@dataclass(frozen=True)
class CircuitSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self):
        groups = (self.train, self.validation, self.test)
        if any(not group for group in groups):
            raise ValueError("Cada partición debe contener al menos un circuito.")
        combined = sum(groups, ())
        if any(type(seed) is not int or seed < 0 for seed in combined):
            raise ValueError("Las semillas de circuito deben ser enteros no negativos.")
        if len(combined) != len(set(combined)):
            raise ValueError(
                "Los circuitos no pueden repetirse dentro de las particiones ni entre ellas."
            )


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 500_000
    n_envs: int = 4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    eval_every: int = 25_000
    torch_threads: int = 1

    def __post_init__(self):
        integers = (
            self.total_timesteps,
            self.n_envs,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.eval_every,
            self.torch_threads,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("Los tamaños y frecuencias de PPO deben ser enteros positivos.")
        if self.batch_size <= 1 or self.n_steps * self.n_envs < self.batch_size:
            raise ValueError("El lote debe tener entre 2 y n_steps * n_envs muestras.")
        if (self.n_steps * self.n_envs) % self.batch_size:
            raise ValueError("batch_size debe dividir n_steps * n_envs.")
        if not 0 < self.learning_rate < 1:
            raise ValueError("La tasa de aprendizaje debe pertenecer a (0, 1).")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma y gae_lambda no pertenecen al intervalo permitido.")
        if self.ent_coef < 0 or self.seed < 0:
            raise ValueError("La entropía y la semilla no pueden ser negativas.")


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig
    circuits: CircuitSplit
    ppo: PPOConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"environment", "circuits", "ppo"}:
        raise ValueError("La configuración debe tener las secciones environment, circuits y ppo.")
    return ExperimentConfig(
        environment=EnvironmentConfig(**data["environment"]),
        circuits=CircuitSplit(**{key: tuple(value) for key, value in data["circuits"].items()}),
        ppo=PPOConfig(**data["ppo"]),
    )
