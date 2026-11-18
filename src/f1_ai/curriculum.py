"""Entradas a curvas durante el entrenamiento de la política de mandos directos."""

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from f1_ai.geometry import CircuitGeometry
from f1_ai.racing_line import RacingPlan

SECTOR_PROBABILITY = (0.70, 0.55, 0.25)
SECTOR_METERS = 120.0
SECTOR_STEPS = 450


@dataclass(frozen=True)
class CornerStart:
    seed: int
    stage: int
    braking: bool = False

    def pose(self, road: CircuitGeometry, plan: RacingPlan) -> tuple[float, float, float, float]:
        rng = np.random.default_rng(self.seed)
        curvature = np.abs(plan.curvature_m_inv)
        peaks = np.flatnonzero(
            (curvature >= np.quantile(curvature, 0.75))
            & (curvature >= np.roll(curvature, 1))
            & (curvature >= np.roll(curvature, -1))
        )
        line = CircuitGeometry(plan.points)
        speeds = np.r_[plan.speed_m_s, plan.speed_m_s[0]]
        weights = None
        lead = (25.0, 55.0)
        speed_ranges = ((0.88, 1.00), (0.90, 1.08), (0.92, 1.12))
        if self.braking:
            entry = (line.cumulative[peaks] - 40.0) % line.length
            entry_speed = np.interp(entry, line.cumulative, speeds)
            drops = np.maximum(0.0, entry_speed - plan.speed_m_s[peaks])
            weights = 1.0 + drops**2
            weights /= weights.sum()
            lead = (20.0, 48.0)
            speed_ranges = ((0.95, 1.02), (0.97, 1.09), (0.98, 1.14))
        apex = road.project(plan.centerline[rng.choice(peaks, p=weights)]).distance
        center = road.sample(apex - rng.uniform(*lead))
        direction = road.project(center).tangent
        right = np.array([direction[1], -direction[0]])
        jitter = (0.25, 0.55, 0.70)[self.stage]
        x, y = center + rng.uniform(-jitter, jitter) * right
        heading_jitter_deg = (2.0, 3.5, 4.0)[self.stage]
        angle = math.atan2(-direction[0], direction[1]) + math.radians(
            rng.uniform(-heading_jitter_deg, heading_jitter_deg)
        )

        distance = line.project(center).distance
        reference_speed = np.interp(distance, line.cumulative, speeds)
        speed = float(reference_speed * rng.uniform(*speed_ranges[self.stage]))
        if self.braking:
            speed = min(speed, plan.config.speed_limit_m_s)
        return angle, float(x), float(y), speed


class CornerCurriculum(gym.Wrapper):
    """Mezcla sectores y vueltas. Aumenta la dificultad y vuelve a priorizar vueltas."""

    def __init__(self, env: gym.Env, n_envs: int, *, braking: bool = False):
        super().__init__(env)
        self.n_envs = n_envs
        self.braking = braking
        self.timesteps = 0
        self.rng = np.random.default_rng(0)
        self.stage = 0
        self.sector = False
        self.sector_steps = 0

    def set_curriculum_timesteps(self, timesteps: int):
        self.timesteps = timesteps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 159]))
        self.stage = 2 if self.timesteps >= 81_920 else int(self.timesteps >= 32_768)
        self.sector = bool(self.rng.random() < SECTOR_PROBABILITY[self.stage])
        self.sector_steps = 0
        if self.sector:
            start = CornerStart(int(self.rng.integers(2**32)), self.stage, self.braking)
            options = {**(options or {}), "corner_start": start}
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**info, "training_sector": self.sector, "curriculum_stage": self.stage}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.timesteps += self.n_envs
        self.sector_steps += 1
        if self.sector and not (terminated or truncated):
            meters = info["forward_progress"] * info["track_length"]
            if meters >= SECTOR_METERS or self.sector_steps >= SECTOR_STEPS:
                # Truncar permite al crítico usar la observación final. No se añade bono de vuelta.
                truncated = True
                info = {**info, "termination_reason": "sector_limit"}
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "training_sector": self.sector, "curriculum_stage": self.stage},
        )
