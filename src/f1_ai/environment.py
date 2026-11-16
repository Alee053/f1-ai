"""Adaptación numérica de CarRacing 1.2.3; se conserva su física Box2D."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import FPS, PLAYFIELD, TRACK_WIDTH, CarRacing

from f1_ai.config import EnvironmentConfig
from f1_ai.geometry import CircuitGeometry, LapProgress

OBSERVATION_VERSION = 1
LOOKAHEAD_METERS = np.array([8.0, 16.0, 32.0, 48.0])
WAYPOINT_SCALE = 50.0
OBSERVATION_NAMES = (
    "forward_speed",
    "lateral_speed",
    "yaw_rate",
    "heading_sin",
    "heading_cos",
    "lateral_error",
    "road_contact",
    "steering",
    "waypoint_8_right",
    "waypoint_8_forward",
    "waypoint_16_right",
    "waypoint_16_forward",
    "waypoint_32_right",
    "waypoint_32_forward",
    "waypoint_48_right",
    "waypoint_48_forward",
)


class NumericCarRacing(CarRacing):
    """Estado instrumentado y dos controles normalizados: giro y aceleración firmada.

    La geometría del mapa es información privilegiada de simulación. No se infiere
    desde imágenes ni se representa un sensor físico. El único método privado
    adaptado es _render: se verifica contra la versión fijada de Gymnasium.
    """

    def __init__(self, config: EnvironmentConfig, render_mode: str | None = None):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("El modo de renderizado debe ser human, rgb_array o None.")
        super().__init__(
            render_mode=render_mode,
            continuous=True,
            lap_complete_percent=config.minimum_coverage,
        )
        self.config = config
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.geometry: CircuitGeometry | None = None
        self.progress: LapProgress | None = None
        self.elapsed_steps = 0
        self.off_track_steps = 0
        self.off_track_events = 0
        self._consecutive_off_track = 0
        self._episode_done = False

    def _render(self, mode: str):
        # CarRacing.step llama este modo incluso si render_mode es None.
        if mode == "state_pixels":
            return self._observation()
        return super()._render(mode)

    def _observation(self) -> np.ndarray:
        assert self.car is not None
        if self.geometry is None:
            self.geometry = CircuitGeometry(np.array([tile[2:4] for tile in self.track]))
        hull = self.car.hull
        position = np.array(hull.position)
        projection = self.geometry.project(position)
        forward = np.array(hull.GetWorldVector((0, 1)))
        right = np.array(hull.GetWorldVector((1, 0)))
        velocity = np.array(hull.linearVelocity)
        future = self.geometry.sample(projection.distance + LOOKAHEAD_METERS) - position
        waypoints = np.column_stack((future @ right, future @ forward)) / WAYPOINT_SCALE
        state = np.array(
            [
                np.dot(velocity, forward) / 60.0,
                np.dot(velocity, right) / 30.0,
                hull.angularVelocity / 5.0,
                np.dot(projection.tangent, right),
                np.dot(projection.tangent, forward),
                projection.lateral / TRACK_WIDTH,
                self.road_contact_fraction,
                self.car.wheels[0].joint.angle / 0.4,
                *waypoints.ravel(),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(state).all():
            raise FloatingPointError("El estado del simulador contiene valores no finitos.")
        return np.clip(state, -1.0, 1.0)

    @property
    def road_contact_fraction(self) -> float:
        assert self.car is not None
        return sum(bool(wheel.tiles) for wheel in self.car.wheels) / len(self.car.wheels)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.geometry = None
        self.progress = None
        self.elapsed_steps = 0
        self.off_track_steps = 0
        self.off_track_events = 0
        self._consecutive_off_track = 0
        self._episode_done = False
        observation, _ = super().reset(seed=seed, options=options)
        assert self.geometry is not None and self.car is not None
        start = self.geometry.project(np.array(self.car.hull.position)).distance
        self.progress = LapProgress(self.geometry.length, start, self.config.minimum_coverage)
        return observation, self._info("running", native_reward=0.0)

    def step(self, action: np.ndarray | None):
        if action is None:
            # Únicamente el paso de inicialización interno de CarRacing.reset.
            return super().step(None)
        if self.progress is None or self._episode_done:
            raise gym.error.ResetNeeded("Se debe llamar a reset antes del siguiente episodio.")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("La acción debe contener dos números finitos en [-1, 1].")
        steering, acceleration = action
        native_action = np.array(
            [
                steering * self.config.steering_scale,
                max(float(acceleration), 0.0),
                max(float(-acceleration), 0.0),
            ],
            dtype=np.float32,
        )
        observation, native_reward, _, _, _ = super().step(native_action)
        self.elapsed_steps += 1
        assert self.car is not None and self.geometry is not None
        coverage = self.tile_visited_count / len(self.track)
        projection = self.geometry.project(np.array(self.car.hull.position))
        gain, completed = self.progress.update(projection.distance, coverage)
        off_track = self.road_contact_fraction < 0.5
        if off_track:
            self.off_track_events += int(self._consecutive_off_track == 0)
            self._consecutive_off_track += 1
            self.off_track_steps += 1
        else:
            self._consecutive_off_track = 0

        x, y = self.car.hull.position
        outside = abs(x) > PLAYFIELD or abs(y) > PLAYFIELD
        off_track_failure = self._consecutive_off_track >= self.config.off_track_grace_steps
        reason = "running"
        if outside:
            reason = "out_of_bounds"
        elif off_track_failure:
            reason = "off_track"
        elif completed:
            reason = "lap_complete"
        elif self.elapsed_steps >= self.config.max_steps:
            reason = "time_limit"
        terminated = reason in {"out_of_bounds", "off_track", "lap_complete"}
        truncated = reason == "time_limit"
        self._episode_done = terminated or truncated

        # Cada metro de avance neto máximo se premia una sola vez por episodio.
        reward = 1000.0 * gain / self.geometry.length - 0.1
        if self.config.reward_profile == "track":
            # El progreso geométrico fuera de carretera no es avance válido.
            if off_track:
                reward = -1.0
            else:
                centered = max(0.0, 1.0 - abs(projection.lateral) / TRACK_WIDTH)
                forward = np.array(self.car.hull.GetWorldVector((0, 1)))
                aligned = max(0.0, float(np.dot(projection.tangent, forward)))
                reward = 1000.0 * gain / self.geometry.length * centered * aligned - 0.1
        reward += 100.0 * (reason == "lap_complete")
        reward -= 50.0 * (reason == "off_track") + 100.0 * outside
        return observation, float(reward), terminated, truncated, self._info(reason, native_reward)

    def _info(self, reason: str, native_reward: float) -> dict[str, Any]:
        assert self.geometry is not None and self.car is not None
        return {
            "lap_complete": reason == "lap_complete",
            "is_success": reason == "lap_complete",
            "termination_reason": reason,
            "coverage": self.tile_visited_count / len(self.track),
            "forward_progress": self.progress.furthest / self.geometry.length
            if self.progress
            else 0.0,
            "simulated_seconds": self.elapsed_steps / FPS,
            "off_track_steps": self.off_track_steps,
            "off_track_events": self.off_track_events,
            "native_reward": float(native_reward),
            "track_length": self.geometry.length,
            "position": tuple(float(value) for value in self.car.hull.position),
            "speed": float(np.linalg.norm(self.car.hull.linearVelocity)),
        }


def make_env(config: EnvironmentConfig, render_mode: str | None = None) -> NumericCarRacing:
    return NumericCarRacing(config, render_mode=render_mode)


class CircuitPool(gym.Wrapper):
    """Selecciona circuitos de una partición sin compartir semillas de evaluación."""

    def __init__(self, env: gym.Env, seeds: tuple[int, ...], *, shuffle: bool):
        super().__init__(env)
        self.seeds = seeds
        self.shuffle = shuffle
        self._selector = np.random.default_rng(0)
        self._index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._selector = np.random.default_rng(seed)
            self._index = 0
        circuit_seed = (
            int(self._selector.choice(self.seeds))
            if self.shuffle
            else self.seeds[self._index % len(self.seeds)]
        )
        self._index += 1
        observation, info = self.env.reset(seed=circuit_seed, options=options)
        return observation, {**info, "circuit_seed": circuit_seed}
