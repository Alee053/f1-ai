"""Configuración explícita y separación de circuitos antes de los experimentos."""

import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from f1_ai.racing_controller import FollowerConfig
from f1_ai.racing_line import ReferenceConfig
from f1_ai.vehicle import FormulaConfig

RACING_RESIDUAL = "racing-residual-v1"
REFERENCE_TIME = "reference-time-v1"
LAP_TIME_REWARDS = {"lap-time-v1", REFERENCE_TIME}


@dataclass(frozen=True)
class ReferenceDrivingConfig:
    plan: ReferenceConfig = ReferenceConfig()
    follower: FollowerConfig = FollowerConfig()

    def __post_init__(self):
        for name, cls in (("plan", ReferenceConfig), ("follower", FollowerConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls(**value)
                object.__setattr__(self, name, value)
            if not isinstance(value, cls):
                raise ValueError(f"racing_reference.{name} debe contener una configuración válida.")


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int = 3000
    off_track_grace_steps: int = 75
    minimum_coverage: float = 0.95
    steering_scale: float = 1.0
    reward_profile: str = "progress"
    vehicle_model: str = "classic"
    formula: FormulaConfig | None = None
    forward_speed_scale: float = 60.0
    observation_profile: str = "legacy"
    control_profile: str = "direct"
    racing_reference: ReferenceDrivingConfig | None = None

    def __post_init__(self):
        if self.vehicle_model not in {"classic", "formula-v1"}:
            raise ValueError("vehicle_model debe ser classic o formula-v1.")
        if isinstance(self.formula, dict):
            object.__setattr__(self, "formula", FormulaConfig(**self.formula))
        if self.vehicle_model == "formula-v1" and self.formula is None:
            object.__setattr__(self, "formula", FormulaConfig())
        if self.formula is not None and not isinstance(self.formula, FormulaConfig):
            raise ValueError("formula debe contener parámetros Formula válidos.")
        if self.vehicle_model == "classic" and self.formula is not None:
            raise ValueError("Los parámetros Formula requieren vehicle_model=formula-v1.")
        if (
            isinstance(self.forward_speed_scale, bool)
            or not isinstance(self.forward_speed_scale, (int, float))
            or not 1 <= self.forward_speed_scale <= 200
        ):
            raise ValueError("forward_speed_scale debe pertenecer a [1, 200].")
        if self.vehicle_model == "formula-v1" and self.steering_scale != 0.4:
            raise ValueError("Formula v1 requiere steering_scale=0.4.")
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_steps, self.off_track_grace_steps)
        ):
            raise ValueError("Los límites de pasos deben ser positivos.")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("La cobertura mínima debe pertenecer a (0, 1].")
        if not 0 < self.steering_scale <= 1:
            raise ValueError("La escala de dirección debe pertenecer a (0, 1].")
        if self.observation_profile not in {"legacy", "racing-v2"}:
            raise ValueError("observation_profile debe ser legacy o racing-v2.")
        if self.control_profile not in {"direct", RACING_RESIDUAL}:
            raise ValueError(f"control_profile debe ser direct o {RACING_RESIDUAL}.")
        if self.control_profile == RACING_RESIDUAL and self.observation_profile != "racing-v2":
            raise ValueError(f"{RACING_RESIDUAL} requiere las mediciones de racing-v2.")
        if isinstance(self.racing_reference, dict):
            object.__setattr__(
                self, "racing_reference", ReferenceDrivingConfig(**self.racing_reference)
            )
        if self.racing_reference is not None and (
            not isinstance(self.racing_reference, ReferenceDrivingConfig)
            or self.control_profile != RACING_RESIDUAL
        ):
            raise ValueError(
                "racing_reference requiere una configuración de control residual válida."
            )
        if self.reward_profile not in {"progress", "track", *LAP_TIME_REWARDS}:
            raise ValueError("Perfil de recompensa desconocido.")
        if self.reward_profile == REFERENCE_TIME and self.control_profile != RACING_RESIDUAL:
            raise ValueError(
                f"{REFERENCE_TIME} requiere la trazada de referencia del control residual."
            )
        if self.vehicle_model != "formula-v1" and (
            self.observation_profile == "racing-v2" or self.reward_profile in LAP_TIME_REWARDS
        ):
            raise ValueError(
                "Las observaciones racing-v2 y los perfiles de tiempo requieren Formula."
            )


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
    device: str = "cpu"
    vec_env: str = "dummy"
    hidden_size: int = 64
    log_std_init: float = 0.0
    reward_scale: float = 1.0
    clip_range: float = 0.2
    target_kl: float | None = None
    use_sde: bool = False
    sde_sample_freq: int = -1
    squash_output: bool = False

    def __post_init__(self):
        if not isinstance(self.device, str) or self.device not in {"cpu", "cuda"}:
            raise ValueError("device debe ser cpu o cuda.")
        if not isinstance(self.vec_env, str) or self.vec_env not in {"dummy", "subproc"}:
            raise ValueError("vec_env debe ser dummy o subproc.")
        integers = (
            self.total_timesteps,
            self.n_envs,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.eval_every,
            self.torch_threads,
            self.hidden_size,
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
        if not isfinite(self.ent_coef) or self.ent_coef < 0:
            raise ValueError("La entropía debe ser finita y no negativa.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo.")
        if not isfinite(self.log_std_init) or not -5 <= self.log_std_init <= 2:
            raise ValueError("log_std_init debe ser finito y pertenecer a [-5, 2].")
        if not 0 < self.reward_scale <= 1 or not 0 < self.clip_range <= 1:
            raise ValueError("Las escalas de recompensa y recorte deben pertenecer a (0, 1].")
        if self.target_kl is not None and not 0 < self.target_kl < 1:
            raise ValueError("target_kl debe ser None o pertenecer a (0, 1).")
        if type(self.use_sde) is not bool or type(self.squash_output) is not bool:
            raise ValueError("use_sde y squash_output deben ser booleanos.")
        if self.squash_output and not self.use_sde:
            raise ValueError("La salida acotada de esta política requiere use_sde.")
        if (
            type(self.sde_sample_freq) is not int
            or self.sde_sample_freq < -1
            or self.sde_sample_freq == 0
        ):
            raise ValueError("sde_sample_freq debe ser -1 o un entero positivo.")


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
