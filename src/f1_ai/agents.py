"""Políticas que reciben exactamente la misma observación numérica."""

from typing import Protocol

import numpy as np

from f1_ai.config import EnvironmentConfig
from f1_ai.environment import WAYPOINT_SCALE


class Agent(Protocol):
    def action(self, observation: np.ndarray) -> np.ndarray: ...


class RandomAgent:
    def __init__(self, seed: int, circuit_seed: int):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, circuit_seed]))

    def action(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=2).astype(np.float32)


class PursuitAgent:
    """Pure pursuit con control proporcional de velocidad, sin aprendizaje.

    La distancia entre ejes y el límite de dirección corresponden a car_dynamics.py.
    Sus constantes se eligen usando circuitos de entrenamiento, nunca de prueba.
    """

    def __init__(
        self,
        target_speed: float = 30.0,
        lateral_acceleration: float = 8.0,
        *,
        environment: EnvironmentConfig | None = None,
    ):
        if not np.isfinite([target_speed, lateral_acceleration]).all() or (
            target_speed <= 0 or lateral_acceleration <= 0
        ):
            raise ValueError("Los parámetros del controlador deben ser positivos.")
        self.target_speed = target_speed
        self.lateral_acceleration = lateral_acceleration
        self.speed_scale = environment.forward_speed_scale if environment else 60.0
        self.wheelbase = (
            environment.formula.wheelbase_m if environment and environment.formula else 3.24
        )

    def action(self, observation: np.ndarray) -> np.ndarray:
        speed = max(0.0, float(observation[0]) * self.speed_scale)
        waypoints = observation[8:].reshape(4, 2) * WAYPOINT_SCALE
        blend = np.clip((8.0 + speed * 0.25 - 8.0) / 8.0, 0, 1)
        target = waypoints[0] * (1 - blend) + waypoints[1] * blend
        curvature = 2.0 * float(target[0]) / max(float(np.dot(target, target)), 1.0)
        steering = np.arctan(self.wheelbase * curvature) / 0.4
        preview_curvature = max(
            abs(2.0 * float(point[0]) / max(float(np.dot(point, point)), 1.0))
            for point in waypoints[:3]
        )
        desired_speed = min(
            self.target_speed,
            np.sqrt(self.lateral_acceleration / max(preview_curvature, 0.001)),
        )
        acceleration = (desired_speed - speed) * 0.15
        return np.clip([steering, acceleration], -1, 1).astype(np.float32)


class PPOAgent:
    def __init__(self, model):
        if model.observation_space.shape != (16,) or model.action_space.shape != (2,):
            raise ValueError("El modelo no usa los espacios de observación y acción del proyecto.")
        self.model = model

    def action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)
