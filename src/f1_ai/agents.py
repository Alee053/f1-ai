"""Políticas que reciben exactamente la misma observación numérica."""

from typing import Protocol

import numpy as np

from f1_ai.config import EnvironmentConfig
from f1_ai.environment import OBSERVATION_NAMES, RACING_OBSERVATION_NAMES, WAYPOINT_SCALE
from f1_ai.vehicle import GRAVITY, aerodynamic_forces


class Agent(Protocol):
    def action(self, observation: np.ndarray) -> np.ndarray: ...


class RandomAgent:
    def __init__(self, seed: int, circuit_seed: int):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, circuit_seed]))

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=2).astype(np.float32)


class PursuitAgent:
    """Pure pursuit con regulación de velocidad, sin aprendizaje.

    Formula añade amortiguación de guiñada y convierte la aceleración deseada
    en demanda de motor o freno. Solo utiliza el estado observado y parámetros
    estáticos del coche; no accede al slip de las ruedas ni al futuro del episodio.
    Sus constantes se eligen usando circuitos de entrenamiento, nunca de prueba.
    """

    def __init__(
        self,
        target_speed: float = 30.0,
        lateral_acceleration: float = 8.0,
        *,
        environment: EnvironmentConfig | None = None,
    ):
        if environment is not None and environment.observation_profile != "legacy":
            raise ValueError(
                "PursuitAgent usa observaciones legacy; racing-v2 requiere otro experto."
            )
        if not np.isfinite([target_speed, lateral_acceleration]).all() or (
            target_speed <= 0 or lateral_acceleration <= 0
        ):
            raise ValueError("Los parámetros del controlador deben ser positivos.")
        self.target_speed = target_speed
        self.lateral_acceleration = lateral_acceleration
        self.speed_scale = environment.forward_speed_scale if environment else 60.0
        self.formula = environment.formula if environment else None
        self.wheelbase = self.formula.wheelbase_m if self.formula else 3.24

    def action(self, observation: np.ndarray) -> np.ndarray:
        speed = max(0.0, float(observation[0]) * self.speed_scale)
        waypoints = observation[8:].reshape(4, 2) * WAYPOINT_SCALE
        blend = np.clip((8.0 + speed * 0.25 - 8.0) / 8.0, 0, 1)
        target = waypoints[0] * (1 - blend) + waypoints[1] * blend
        curvature = 2.0 * float(target[0]) / max(float(np.dot(target, target)), 1.0)
        steering_angle = np.arctan(self.wheelbase * curvature)
        if self.formula:
            # Giro positivo apunta a la derecha; la guiñada Box2D positiva, a la izquierda.
            steering_angle += 0.08 * (speed * curvature + float(observation[2]) * 5.0)
        preview = waypoints if self.formula else waypoints[:3]
        preview_curvature = max(
            abs(2.0 * float(point[0]) / max(float(np.dot(point, point)), 1.0)) for point in preview
        )
        desired_speed = min(
            self.target_speed,
            np.sqrt(self.lateral_acceleration / max(preview_curvature, 0.001)),
        )
        acceleration = (desired_speed - speed) * 0.15
        if self.formula:
            acceleration = self._formula_pedal(speed, desired_speed)
        return np.clip([steering_angle / 0.4, acceleration], -1, 1).astype(np.float32)

    def _formula_pedal(self, speed: float, desired_speed: float) -> float:
        """Demanda conservadora de actuadores; no añade asistencias a la física."""
        config = self.formula
        assert config is not None
        acceleration = float(np.clip(0.8 * (desired_speed - speed), -9.0, 6.0))
        _, drag = aerodynamic_forces(speed, config)
        resistance = drag + config.rolling_resistance * config.mass_kg * GRAVITY
        force = config.mass_kg * acceleration + resistance
        if force < 0:
            return -min(-force / config.brake_force_limit_n, 0.4)
        available = min(
            config.drive_force_limit_n,
            config.power_kw * 1000 * config.drivetrain_efficiency / max(speed, 3.0),
        )
        return float(np.clip(force / available, 0.0, 0.7))


class PPOAgent:
    def __init__(self, model):
        valid_shapes = {(len(OBSERVATION_NAMES),), (len(RACING_OBSERVATION_NAMES),)}
        if model.observation_space.shape not in valid_shapes or model.action_space.shape != (2,):
            raise ValueError("El modelo no usa los espacios de observación y acción del proyecto.")
        self.model = model

    def action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)
