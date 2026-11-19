"""Objetivo de tiempo con costes menores por controles bruscos y slip excesivo."""

from math import radians

import gymnasium as gym
import numpy as np

from f1_ai.config import ReferenceDrivingConfig
from f1_ai.geometry import TurnPreview
from f1_ai.plan_features import ReferenceTimeProgress, track_plan

LAP_PROGRESS_POINTS = 1000.0
MINIMUM_PROGRESS_M = 4.0
NO_PROGRESS_COST = 120.0
STRAIGHT_LOOKAHEAD_M = 50.0
STRAIGHT_TURN_ANGLE = radians(12.0)
CURVE_TURN_ANGLE = radians(30.0)


class RoadLearningReward(gym.Wrapper):
    """Enseña avance sobre asfalto con un límite de ritmo suave."""

    off_track_steps = 25
    stall_cost = 24.0

    def __init__(self, env: gym.Env, pace_m_s: float, *, straight_pace_m_s: float = 0.0):
        super().__init__(env)
        self.pace_m_s = pace_m_s
        self.straight_pace_m_s = straight_pace_m_s
        self.turn_preview = None
        self.previous_progress_m = 0.0
        self.consecutive_off_track = 0

    def reset(self, *, seed=None, options=None):
        self.previous_progress_m = 0.0
        self.consecutive_off_track = 0
        observation, info = self.env.reset(seed=seed, options=options)
        if self.straight_pace_m_s:
            self.turn_preview = TurnPreview(self.env.unwrapped.geometry)
        return observation, info

    def step(self, action):
        physics = self.env.unwrapped
        previous_action = physics.previous_action
        observation, _, terminated, truncated, info = self.env.step(action)
        distance = info["forward_progress"] * info["track_length"]
        gain = distance - self.previous_progress_m
        self.previous_progress_m = distance
        contact = physics.road_contact_fraction
        if contact >= 0.5:
            self.consecutive_off_track = 0
            delta = physics.previous_action - previous_action
            slip_excess = min(2.0, max(0.0, max(map(abs, info["slip_ratios"])) - 0.30))
            pace = self.pace_m_s
            if self.turn_preview is not None:
                progress = physics.progress
                turn = self.turn_preview.ahead(
                    progress.start + progress.previous, STRAIGHT_LOOKAHEAD_M
                )
                fraction = (CURVE_TURN_ANGLE - turn) / (CURVE_TURN_ANGLE - STRAIGHT_TURN_ANGLE)
                pace += (self.straight_pace_m_s - pace) * min(1.0, max(0.0, fraction))
            speed_excess = max(0.0, info["speed"] - pace)
            reward = (
                gain
                - 0.04
                - 0.20 * (1.0 - contact)
                - 0.04 * float(delta @ delta)
                - 0.035 * slip_excess**2
                - 0.003 * speed_excess**2
            )
        else:
            self.consecutive_off_track += 1
            reward = -0.35
            if self.consecutive_off_track >= self.off_track_steps and not (terminated or truncated):
                terminated = True
                info = {**info, "termination_reason": "off_track"}
        if terminated:
            reward += 30.0 if info["lap_complete"] else -12.0
        return observation, reward, terminated, truncated, info


class NoProgressTermination(gym.Wrapper):
    """Acota los episodios de entrenamiento sin avance válido."""

    def __init__(self, env: gym.Env, patience_steps: int, *, cost: float = NO_PROGRESS_COST):
        super().__init__(env)
        self.patience_steps = patience_steps
        self.cost = cost
        self.last_progress_m = 0.0
        self.new_progress_m = 0.0
        self.steps_without_progress = 0

    def reset(self, *, seed=None, options=None):
        self.last_progress_m = 0.0
        self.new_progress_m = 0.0
        self.steps_without_progress = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        progress_m = info["forward_progress"] * info["track_length"]
        on_road = self.env.unwrapped.road_contact_fraction >= 0.5
        if on_road:
            self.new_progress_m += progress_m - self.last_progress_m
        self.last_progress_m = progress_m
        if self.new_progress_m >= MINIMUM_PROGRESS_M:
            self.new_progress_m = 0.0
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        if (
            on_road
            and not (terminated or truncated)
            and self.steps_without_progress >= self.patience_steps
        ):
            terminated = True
            reward -= self.cost
            info = {**info, "termination_reason": "no_progress"}
        return observation, reward, terminated, truncated, info


class RoadContactPenalty(gym.Wrapper):
    """Cobra la fracción de ruedas fuera del asfalto, incluso antes de salir por completo."""

    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.coefficient * (1 - self.env.unwrapped.road_contact_fraction)
        return observation, reward, terminated, truncated, info


class TimeProgressReward(gym.Wrapper):
    """Redistribuye el premio de avance hacia sectores lentos durante PPO."""

    def __init__(self, env: gym.Env, mix: float):
        super().__init__(env)
        self.mix = mix
        self.previous_distance = 0.0
        self.time_progress = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        physics = self.env.unwrapped
        reference = physics.config.racing_reference or ReferenceDrivingConfig()
        plan = track_plan(physics.geometry.points.tobytes(), physics.config.formula, reference.plan)
        self.time_progress = ReferenceTimeProgress(plan)
        self.previous_distance = 0.0
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        physics = self.env.unwrapped
        progress = physics.progress
        distance = progress.furthest
        if distance > self.previous_distance and physics.road_contact_fraction >= 0.5:
            extra = self.time_progress.adjustment(
                progress.start, self.previous_distance, distance, progress.length
            )
            reward += self.mix * LAP_PROGRESS_POINTS * extra
        self.previous_distance = distance
        return observation, reward, terminated, truncated, info


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


class FrontLockPenalty(gym.Wrapper):
    """Penaliza el deslizamiento de frenada y el bloqueo del eje delantero."""

    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if action[1] < 0 and self.env.unwrapped.road_contact_fraction >= 0.5:
            slip = -min(info["slip_ratios"][:2])
            excess = min(1.25, max(0.0, slip - 0.3))
            locked = any(info["locked_wheels"][:2])
            reward -= self.coefficient * (excess**2 + 0.35 * locked)
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
