"""Objetivo de tiempo con costes menores por controles bruscos y slip excesivo."""

import gymnasium as gym
import numpy as np

LAP_PROGRESS_POINTS = 1000.0


class RearSlipPenalty(gym.Wrapper):
    """Refuerza el coste por patinamiento del eje motriz solo durante PPO."""

    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.road_contact_fraction >= 0.5:
            slip = max(abs(value) for value in info["slip_ratios"][2:])
            excess = min(2.0, max(0.0, slip - 0.3))
            reward -= self.coefficient * excess**2
        return observation, reward, terminated, truncated, info


def lap_time_step_reward(
    progress_fraction: float,
    *,
    on_road: bool,
    action_delta: np.ndarray,
    max_abs_slip: float,
    sideslip_rad: float,
) -> float:
    """Cada vuelta válida suma el mismo avance; menos pasos conserva más retorno.

    Los bonos terminales se aplican en el entorno. No se pondera por distancia al
    eje ni por alineación con su tangente. Se permite el slip útil del neumático.
    """
    if not on_road:
        return -1.0
    smoothness = 0.02 * float(np.dot(action_delta, action_delta))
    excessive_slip = 0.1 * min(max(0.0, max_abs_slip - 0.2), 5.0) ** 2
    excessive_drift = 0.1 * max(0.0, abs(sideslip_rad) - 0.15) ** 2
    return (
        LAP_PROGRESS_POINTS * progress_fraction
        - 0.1
        - smoothness
        - excessive_slip
        - excessive_drift
    )
