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
LAUNCH_STEPS = 200
WIDE_SECTOR_METERS = 145.0
WIDE_SECTOR_STEPS = 500
WIDE_SECTOR_TIME_BONUS = 0.10
ROAD_SECTOR_PROBABILITY = (0.85, 0.65, 0.35)
ROAD_LAUNCH_PROBABILITY = (0.20, 0.35, 0.50)
ROAD_SECTOR_METERS = (55.0, 90.0, 120.0)
ROAD_LAUNCH_METERS = 18.0


@dataclass(frozen=True)
class CornerStart:
    seed: int
    stage: int
    braking: bool = False
    recovery: bool = False
    wide_entry: bool = False

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
        if self.recovery:
            lead = (14.0, 38.0)
        if self.wide_entry:
            lead = (30.0, 50.0)
            speed_ranges = ((0.86, 0.98), (0.90, 1.02), (0.93, 1.06))
        apex = road.project(plan.centerline[rng.choice(peaks, p=weights)]).distance
        center = road.sample(apex - rng.uniform(*lead))
        direction = road.project(center).tangent
        right = np.array([direction[1], -direction[0]])
        jitter = (0.25, 0.55, 0.70)[self.stage]
        if self.recovery:
            jitter = 2.0
        if self.wide_entry:
            usable_width = plan.half_width_m - plan.vehicle.width_m / 2 - plan.config.road_margin_m
            upper_fraction = (0.63, 0.69, 0.74)[self.stage]
            offset = rng.choice((-1, 1)) * usable_width * rng.uniform(0.45, upper_fraction)
        else:
            offset = rng.uniform(-jitter, jitter)
        x, y = center + offset * right
        heading_jitter_deg = (2.0, 3.5, 4.0)[self.stage]
        if self.recovery:
            heading_jitter_deg = 10.0
        angle = math.atan2(-direction[0], direction[1]) + math.radians(
            rng.uniform(-heading_jitter_deg, heading_jitter_deg)
        )

        distance = line.project(center).distance
        reference_speed = np.interp(distance, line.cumulative, speeds)
        speed = float(reference_speed * rng.uniform(*speed_ranges[self.stage]))
        if self.braking:
            speed = min(speed, plan.config.speed_limit_m_s)
        return angle, float(x), float(y), speed


@dataclass(frozen=True)
class RoadStart:
    seed: int
    stage: int

    def pose(self, road: CircuitGeometry) -> tuple[float, float, float, float]:
        rng = np.random.default_rng(self.seed)
        turns = np.linalg.norm(road.tangents - np.roll(road.tangents, 1, axis=0), axis=1)
        corners = np.flatnonzero(turns >= np.quantile(turns, 0.75))
        apex = road.cumulative[rng.choice(corners)]
        position = road.sample(apex - rng.uniform(18.0, 42.0))
        direction = road.project(position).tangent
        right = np.array([direction[1], -direction[0]])
        lateral_range = (0.3, 0.8, 1.4)[self.stage]
        yaw_range = (3.0, 5.0, 8.0)[self.stage]
        heading = math.atan2(-direction[0], direction[1]) + math.radians(
            rng.uniform(-yaw_range, yaw_range)
        )
        x, y = position + right * rng.uniform(-lateral_range, lateral_range)
        speed_ranges = ((7.0, 13.0), (9.0, 17.0), (10.0, 21.0))
        speed = rng.uniform(*speed_ranges[self.stage])
        return heading, float(x), float(y), float(speed)


class CornerCurriculum(gym.Wrapper):
    """Mezcla sectores y vueltas. Aumenta la dificultad y vuelve a priorizar vueltas."""

    def __init__(
        self,
        env: gym.Env,
        n_envs: int,
        *,
        braking: bool = False,
        recovery: bool = False,
        launch: bool = False,
        wide_entry: bool = False,
        road_starts: bool = False,
        road_stage_scale: int = 1,
        road_sector_bonus: float = 0.0,
    ):
        super().__init__(env)
        self.n_envs = n_envs
        self.braking = braking
        self.recovery = recovery
        self.launch = launch
        self.wide_entry = wide_entry
        self.road_starts = road_starts
        self.stage_scale = road_stage_scale if road_starts else 1
        self.road_sector_bonus = road_sector_bonus if road_starts else 0.0
        self.timesteps = 0
        self.rng = np.random.default_rng(0)
        self.stage = 0
        self.sector = False
        self.sector_steps = 0
        self.recovery_start = False
        self.launch_start = False
        self.wide_start = False
        self.sector_lock = False
        self.wide_sector_seconds = 0.0

    def set_curriculum_timesteps(self, timesteps: int):
        self.timesteps = timesteps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 159]))
        stage_steps = self.timesteps // self.stage_scale
        self.stage = 2 if stage_steps >= 81_920 else int(stage_steps >= 32_768)
        sector_probability = ROAD_SECTOR_PROBABILITY if self.road_starts else SECTOR_PROBABILITY
        launch_probability = ROAD_LAUNCH_PROBABILITY[self.stage] if self.road_starts else 0.5
        self.sector = bool(self.rng.random() < sector_probability[self.stage])
        self.launch_start = bool(
            self.sector and self.launch and self.rng.random() < launch_probability
        )
        self.wide_start = bool(
            self.sector and not self.launch_start and self.wide_entry and self.rng.random() < 0.7
        )
        self.recovery_start = bool(
            self.sector
            and not self.launch_start
            and not self.wide_start
            and self.recovery
            and self.stage > 0
            and self.rng.random() < 0.5
        )
        self.sector_steps = 0
        self.sector_lock = False
        self.wide_sector_seconds = 0.0
        if self.sector and not self.launch_start:
            pose_seed = int(self.rng.integers(2**32))
            if self.road_starts:
                start = RoadStart(pose_seed, self.stage)
            else:
                start = CornerStart(
                    pose_seed,
                    self.stage,
                    braking=self.braking,
                    recovery=self.recovery_start,
                    wide_entry=self.wide_start,
                )
            options = {**(options or {}), "corner_start": start}
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self._episode_info(info)

    def _episode_info(self, info):
        return {
            **info,
            "training_sector": self.sector,
            "curriculum_stage": self.stage,
            "training_recovery": self.recovery_start,
            "training_launch": self.launch_start,
            "training_wide_entry": self.wide_start,
            "training_wide_sector_seconds": self.wide_sector_seconds,
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.timesteps += self.n_envs
        self.sector_steps += 1
        road_practice = self.sector and self.road_sector_bonus > 0
        if self.wide_start or road_practice:
            self.sector_lock = self.sector_lock or any(info["locked_wheels"])
        if self.sector and not (terminated or truncated):
            meters = info["forward_progress"] * info["track_length"]
            distance_limit = SECTOR_METERS
            reason = "sector_limit"
            if self.wide_start:
                distance_limit = WIDE_SECTOR_METERS
                step_limit = WIDE_SECTOR_STEPS
                reason = "wide_sector_limit"
            elif self.launch_start:
                if road_practice:
                    distance_limit = ROAD_LAUNCH_METERS
                step_limit = LAUNCH_STEPS
                reason = "launch_limit"
            elif self.road_starts:
                distance_limit = ROAD_SECTOR_METERS[self.stage]
                step_limit = SECTOR_STEPS
                reason = "road_sector_limit"
            else:
                step_limit = SECTOR_STEPS
            reached_exit = meters >= distance_limit
            if reached_exit or self.sector_steps >= step_limit:
                # El truncamiento permite al crítico usar la observación final.
                truncated = True
                clean_exit = reached_exit and info["off_track_steps"] == 0 and not self.sector_lock
                if self.wide_start and clean_exit:
                    self.wide_sector_seconds = info["simulated_seconds"]
                    reward += (step_limit - self.sector_steps) * WIDE_SECTOR_TIME_BONUS
                    reason = "wide_sector_complete"
                elif road_practice and clean_exit:
                    reward += self.road_sector_bonus
                    reason = "road_launch_complete" if self.launch_start else "road_sector_complete"
                info = {**info, "termination_reason": reason}
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._episode_info(info),
        )
