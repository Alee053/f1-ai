"""Correcciones de trazada y velocidad sobre el seguidor de Formula."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import FPS, TRACK_WIDTH

from f1_ai.config import REFERENCE_TIME, ReferenceDrivingConfig
from f1_ai.environment import RACING_OBSERVATION_NAMES, NumericCarRacing
from f1_ai.plan_features import PREVIEW_M, ReferenceTimeProgress, track_plan
from f1_ai.racing_controller import RacingController, ReferenceState
from f1_ai.rewards import LAP_PROGRESS_POINTS

RESIDUAL_ACTION_NAMES = ("raceline_offset", "target_speed_adjustment")
RESIDUAL_OBSERVATION_NAMES = (
    *RACING_OBSERVATION_NAMES[:8],
    *RACING_OBSERVATION_NAMES[-10:],
    "raceline_error",
    "target_speed_error",
    *(
        f"raceline_{int(distance)}_{field}"
        for distance in PREVIEW_M
        for field in ("right", "forward", "target_speed", "curvature")
    ),
    "filtered_offset",
    "filtered_speed_adjustment",
)


class ResidualRacingEnv(gym.Wrapper):
    """La política decide a 10 Hz. El seguidor conserva el control a 50 Hz."""

    steps_per_action = 5
    max_offset_m = 1.6
    max_speed_adjustment = 0.25
    filter_gain = 1 - math.exp(-1 / (FPS * 0.22))

    def __init__(self, env: NumericCarRacing):
        super().__init__(env)
        self.observation_space = spaces.Box(
            -1, 1, shape=(len(RESIDUAL_OBSERVATION_NAMES),), dtype=np.float32
        )
        self.metadata = {**env.metadata, "render_fps": FPS // self.steps_per_action}
        self.filtered = np.zeros(2, dtype=np.float64)
        self.controller = None
        self._timed_progress = env.config.reward_profile == REFERENCE_TIME
        self._reference = env.config.racing_reference or ReferenceDrivingConfig()

    def reset(self, *, seed=None, options=None):
        physical, info = self.env.reset(seed=seed, options=options)
        plan = track_plan(
            self.env.geometry.points.tobytes(), self.env.config.formula, self._reference.plan
        )
        self.controller = RacingController(plan, self._reference.follower)
        self._speeds = np.r_[plan.speed_m_s, plan.speed_m_s[0]]
        self._curvature = np.r_[plan.curvature_m_inv, plan.curvature_m_inv[0]]
        self.filtered.fill(0)
        if self._timed_progress:
            self._time_progress = ReferenceTimeProgress(plan)
        return self._observation(physical), info

    def _observation(self, physical):
        hull = self.env.car.hull
        position = np.array(hull.position)
        forward = np.array(hull.GetWorldVector((0, 1)))
        right = np.array([forward[1], -forward[0]])
        geometry = self.controller.geometry
        projection = geometry.project(position)
        distance = (projection.distance + PREVIEW_M) % geometry.length
        future = geometry.sample(distance) - position
        target_speeds = np.interp(distance, geometry.cumulative, self._speeds)
        target_curvature = np.interp(distance, geometry.cumulative, self._curvature)
        speed = physical[0] * self.env.config.forward_speed_scale
        target = np.interp(projection.distance, geometry.cumulative, self._speeds)
        preview = np.column_stack(
            (
                future @ right / PREVIEW_M,
                future @ forward / PREVIEW_M,
                target_speeds / 60,
                12 * target_curvature,
            )
        )
        return np.clip(
            np.concatenate(
                (
                    physical[:8],
                    physical[-10:],
                    [projection.lateral / TRACK_WIDTH, (target - speed) / 20],
                    preview.ravel(),
                    self.filtered,
                ),
                dtype=np.float32,
            ),
            -1,
            1,
        )

    def step(self, action):
        if self.controller is None or self.env._episode_done:
            raise gym.error.ResetNeeded("Se debe reiniciar el entorno antes de conducir.")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("La corrección debe contener dos valores finitos en [-1, 1].")
        reward_sum = native_sum = speed_sum = max_slip = variation = 0.0
        lock_steps = 0
        progress = self.env.progress
        for _ in range(self.steps_per_action):
            self.filtered += self.filter_gain * (action - self.filtered)
            command = self.controller.action(
                ReferenceState.from_car(self.env.car),
                1 / FPS,
                lateral_offset_m=self.max_offset_m * self.filtered[0],
                speed_factor=1 + self.max_speed_adjustment * self.filtered[1],
            )
            variation += abs(float(command[0] - self.env.previous_action[0]))
            previous_distance = progress.furthest
            previous_off_track = self.env.off_track_steps
            physical, reward, terminated, truncated, info = self.env.step(command)
            if self._timed_progress and info["off_track_steps"] == previous_off_track:
                current_distance = progress.furthest
                if current_distance > previous_distance:
                    reward += LAP_PROGRESS_POINTS * self._time_progress.adjustment(
                        progress.start, previous_distance, current_distance, progress.length
                    )
            reward_sum += reward
            native_sum += info["native_reward"]
            speed_sum += info["speed"]
            max_slip = max(max_slip, *(abs(value) for value in info["slip_ratios"]))
            lock_steps += int(any(info["locked_wheels"]))
            if terminated or truncated:
                break
        return (
            self._observation(physical),
            reward_sum,
            terminated,
            truncated,
            {
                **info,
                "native_reward": native_sum,
                "speed_sum": speed_sum,
                "applied_action": command,
                "steering_variation": variation,
                "max_abs_slip_ratio": max_slip,
                "wheel_lock_steps": lock_steps,
            },
        )
