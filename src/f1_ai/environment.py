"""CarRacing numérico con física clásica o modelo Formula versionado."""

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import FPS, PLAYFIELD, TRACK_WIDTH, CarRacing

from f1_ai.config import (
    CORRIDOR_OBSERVATION,
    LAP_TIME_REWARDS,
    MAP_OBSERVATIONS,
    PLANNED_OBSERVATION,
    RACING_RESIDUAL,
    REFERENCE_OBSERVATIONS,
    ROAD_OBSERVATION,
    ROAD_OBSERVATIONS,
    ROAD_PREVIEW_OBSERVATION,
    EnvironmentConfig,
    ReferenceDrivingConfig,
)
from f1_ai.curriculum import CornerStart, RoadStart
from f1_ai.geometry import CircuitGeometry, LapProgress
from f1_ai.plan_features import PREVIEW_M, CorridorFeatures, PlanFeatures, track_plan
from f1_ai.rewards import lap_time_step_reward
from f1_ai.track_layout import TrackLayout

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
RACING_LOOKAHEAD_METERS = np.array([6.0, 12.0, 24.0, 40.0, 60.0, 90.0, 130.0, 180.0])
RACING_OBSERVATION_NAMES = (
    *OBSERVATION_NAMES[:8],
    *(
        f"waypoint_{int(distance)}_{axis}"
        for distance in RACING_LOOKAHEAD_METERS
        for axis in ("right", "forward")
    ),
    *(f"wheel_slip_{wheel}" for wheel in ("fl", "fr", "rl", "rr")),
    *(f"wheel_lock_{wheel}" for wheel in ("fl", "fr", "rl", "rr")),
    "previous_steering",
    "previous_pedal",
)


VEHICLE_OBSERVATION_NAMES = (
    *RACING_OBSERVATION_NAMES[:8],
    *RACING_OBSERVATION_NAMES[-10:-2],
)
PLANNED_OBSERVATION_NAMES = (
    *VEHICLE_OBSERVATION_NAMES,
    "raceline_error",
    "target_speed_error",
    "profile_acceleration",
    "near_target_right",
    "near_target_forward",
    *(
        f"raceline_{int(distance)}_{field}"
        for distance in PREVIEW_M
        for field in ("right", "forward", "target_speed", "curvature")
    ),
)

ROAD_EDGE_NAMES = tuple(
    f"road_{side}_{int(distance)}_{axis}"
    for distance in PREVIEW_M
    for side in ("left", "right")
    for axis in ("right", "forward")
)
CORRIDOR_OBSERVATION_NAMES = (*PLANNED_OBSERVATION_NAMES, *ROAD_EDGE_NAMES)
ROAD_OBSERVATION_NAMES = (*VEHICLE_OBSERVATION_NAMES, *ROAD_EDGE_NAMES)
ROAD_PREVIEW_OBSERVATION_NAMES = (
    *ROAD_OBSERVATION_NAMES,
    "previous_steering",
    "previous_pedal",
    *(f"road_turn_{int(distance)}_{axis}" for distance in PREVIEW_M for axis in ("sin", "cos")),
)


def _direct_observation_contract(config: EnvironmentConfig) -> tuple[int, tuple[str, ...]]:
    if config.observation_profile == ROAD_PREVIEW_OBSERVATION:
        return 7, ROAD_PREVIEW_OBSERVATION_NAMES
    if config.observation_profile == ROAD_OBSERVATION:
        return 6, ROAD_OBSERVATION_NAMES
    if config.observation_profile == CORRIDOR_OBSERVATION:
        return 5, CORRIDOR_OBSERVATION_NAMES
    if config.observation_profile == PLANNED_OBSERVATION:
        return 4, PLANNED_OBSERVATION_NAMES
    if config.observation_profile == "racing-v2":
        return 2, RACING_OBSERVATION_NAMES
    return OBSERVATION_VERSION, OBSERVATION_NAMES


def observation_contract(config: EnvironmentConfig) -> tuple[int, tuple[str, ...]]:
    if config.control_profile == RACING_RESIDUAL:
        from f1_ai.residual_env import RESIDUAL_OBSERVATION_NAMES

        return 3, RESIDUAL_OBSERVATION_NAMES
    return _direct_observation_contract(config)


class NumericCarRacing(CarRacing):
    """Estado instrumentado y dos controles normalizados: giro y aceleración firmada.

    La geometría del mapa es información privilegiada de simulación. No se infiere
    desde imágenes ni se representa un sensor físico. El único método privado
    adaptado es _render. Formula sustituye el coche en el paso inicial de reset;
    ambos contratos se verifican contra la versión fijada de Gymnasium.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        render_mode: str | None = None,
        *,
        track_layout: TrackLayout | None = None,
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("El modo de renderizado debe ser human, rgb_array o None.")
        if track_layout is not None and (
            config.observation_profile not in ROAD_OBSERVATIONS
            or config.control_profile != "direct"
        ):
            raise ValueError(
                "Un circuito importado requiere observaciones viales y mandos directos."
            )
        super().__init__(
            render_mode=render_mode,
            continuous=True,
            lap_complete_percent=config.minimum_coverage,
        )
        self.config = config
        self.track_layout = track_layout
        self.playfield = PLAYFIELD
        if track_layout is not None:
            self.playfield = max(PLAYFIELD, float(np.abs(track_layout.edges).max()) + 65.0)
            self.road_color = np.array([63, 69, 80])
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(len(_direct_observation_contract(config)[1]),), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.geometry: CircuitGeometry | None = None
        self._plan_features: PlanFeatures | None = None
        self._corridor_features: CorridorFeatures | None = None
        self._corner_start: CornerStart | RoadStart | None = None
        self.progress: LapProgress | None = None
        self.elapsed_steps = 0
        self.off_track_steps = 0
        self.off_track_events = 0
        self._consecutive_off_track = 0
        self._episode_done = False
        self.previous_action = np.zeros(2, dtype=np.float32)

    def _create_track(self):
        layout = self.track_layout
        if layout is None:
            return super()._create_track()
        self.road = []
        self.track = [
            (0.0, float(angle), float(x), float(y))
            for angle, (x, y) in zip(layout.angles, layout.geometry.points, strict=True)
        ]
        for index, (left, right) in enumerate(layout.edges):
            previous_left, previous_right = layout.edges[index - 1]
            vertices = [tuple(point) for point in (left, right, previous_right, previous_left)]
            self.fd_tile.shape.vertices = vertices
            tile = self.world.CreateStaticBody(fixtures=self.fd_tile)
            tile.userData = tile
            tile.color = self.road_color + 2 * (index % 3)
            tile.road_visited = False
            tile.road_friction = 1.0
            tile.idx = index
            tile.fixtures[0].sensor = True
            self.road_poly.append((vertices, tile.color))
            self.road.append(tile)

        start_left, start_right = layout.edges[0]
        squares = np.linspace(start_left, start_right, 11)
        depth = 1.0 * layout.geometry.tangents[0]
        for index, (left, right) in enumerate(zip(squares[:-1], squares[1:], strict=True)):
            color = (232, 235, 240) if index % 2 == 0 else (37, 43, 54)
            vertices = [tuple(point) for point in (left, right, right + depth, left + depth)]
            self.road_poly.append((vertices, color))
        return True

    def _render_road(self, zoom, translation, angle):
        if self.track_layout is None:
            super()._render_road(zoom, translation, angle)
            return
        self.surf.fill((113, 124, 133))
        for vertices, color in self.road_poly:
            self._draw_colored_polygon(
                self.surf, vertices, tuple(map(int, color)), zoom, translation, angle
            )

    def _render(self, mode: str):
        # CarRacing.step llama este modo incluso si render_mode es None.
        if mode == "state_pixels":
            return self._observation()
        return super()._render(mode)

    def _road_geometry(self) -> CircuitGeometry:
        if self.geometry is None:
            self.geometry = (
                self.track_layout.geometry
                if self.track_layout is not None
                else CircuitGeometry(np.array([tile[2:4] for tile in self.track]))
            )
        return self.geometry

    def _track_plan(self, reference: ReferenceDrivingConfig):
        return track_plan(
            self._road_geometry().points.tobytes(), self.config.formula, reference.plan
        )

    def _observation(self) -> np.ndarray:
        assert self.car is not None
        geometry = self._road_geometry()
        hull = self.car.hull
        position = np.array(hull.position)
        projection = geometry.project(position)
        forward = np.array(hull.GetWorldVector((0, 1)))
        right = np.array(hull.GetWorldVector((1, 0)))
        velocity = np.array(hull.linearVelocity)
        profile = self.config.observation_profile
        planned = profile in REFERENCE_OBSERVATIONS
        road_only = profile in ROAD_OBSERVATIONS
        racing = profile != "legacy"
        lookahead = RACING_LOOKAHEAD_METERS if racing else LOOKAHEAD_METERS
        if planned:
            if self._plan_features is None:
                reference = self.config.racing_reference or ReferenceDrivingConfig()
                self._plan_features = PlanFeatures(
                    self._track_plan(reference), reference.follower.lookahead_seconds
                )
            waypoints = self._plan_features.observation(
                position, forward, right, float(velocity @ forward)
            )
        elif road_only:
            waypoints = ()
        else:
            future = geometry.sample(projection.distance + lookahead) - position
            scale = lookahead[:, None] if racing else WAYPOINT_SCALE
            waypoints = (np.column_stack((future @ right, future @ forward)) / scale).ravel()
        extra = []
        if racing:
            extra = [
                *(np.array(self.car.telemetry["slip_ratios"]) / 2.0),
                *self.car.telemetry["locked_wheels"],
            ]
            if not (planned or road_only):
                extra.extend(self.previous_action)
        corridor = ()
        if profile == CORRIDOR_OBSERVATION or road_only:
            if self._corridor_features is None:
                angles = np.array([tile[1] for tile in self.track])
                edges = self.track_layout.edges if self.track_layout is not None else None
                self._corridor_features = CorridorFeatures(geometry, angles, edges=edges)
            corridor = self._corridor_features.observation(
                projection.distance, position, forward, right
            )
        preview = ()
        if profile == ROAD_PREVIEW_OBSERVATION:
            turns = self._corridor_features.road_turns(projection.distance, projection.tangent)
            preview = (*self.previous_action, *turns)
        state = np.array(
            [
                np.dot(velocity, forward) / self.config.forward_speed_scale,
                np.dot(velocity, right) / 30.0,
                hull.angularVelocity / 5.0,
                np.dot(projection.tangent, right),
                np.dot(projection.tangent, forward),
                projection.lateral / TRACK_WIDTH,
                self.road_contact_fraction,
                self.car.wheels[0].joint.angle / 0.4,
                *(extra if planned else waypoints),
                *(waypoints if planned else extra),
                *corridor,
                *preview,
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
        start = (options or {}).get("corner_start")
        if start is not None and (
            not isinstance(start, (CornerStart, RoadStart))
            or self.config.observation_profile not in MAP_OBSERVATIONS
            or self.config.control_profile != "direct"
            or (
                isinstance(start, RoadStart)
                and self.config.observation_profile not in ROAD_OBSERVATIONS
            )
        ):
            raise ValueError("La salida de sector requiere el currículo de Formula directo.")
        self.geometry = None
        self._plan_features = None
        self._corridor_features = None
        self.progress = None
        self.elapsed_steps = 0
        self.off_track_steps = 0
        self.off_track_events = 0
        self._consecutive_off_track = 0
        self._episode_done = False
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._corner_start = start
        try:
            observation, _ = super().reset(seed=seed, options=options)
        finally:
            self._corner_start = None
        assert self.geometry is not None and self.car is not None
        start = self.geometry.project(np.array(self.car.hull.position)).distance
        self.progress = LapProgress(self.geometry.length, start, self.config.minimum_coverage)
        return observation, self._info("running", native_reward=0.0)

    def step(self, action: np.ndarray | None):
        if action is None:
            # reset crea el coche clásico antes de llamar a step(None). Se sustituye
            # antes del primer paso físico, sin parches globales sobre Gymnasium.
            if self.config.vehicle_model == "formula-v1":
                from f1_ai.formula_car import FormulaCar

                assert self.car is not None and self.config.formula is not None
                if not isinstance(self.car, FormulaCar):
                    angle, x, y = self.track[0][1:4]
                    speed = 0.0
                    if isinstance(self._corner_start, RoadStart):
                        angle, x, y, speed = self._corner_start.pose(self._road_geometry())
                    elif self._corner_start is not None:
                        reference = self.config.racing_reference or ReferenceDrivingConfig()
                        angle, x, y, speed = self._corner_start.pose(
                            self._road_geometry(), self._track_plan(reference)
                        )
                    self.car.destroy()
                    self.car = FormulaCar(
                        self.world, angle, x, y, self.config.formula, initial_speed=speed
                    )
            return super().step(None)
        if self.progress is None or self._episode_done:
            raise gym.error.ResetNeeded("Se debe llamar a reset antes del siguiente episodio.")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("La acción debe contener dos números finitos en [-1, 1].")
        action_delta = action - self.previous_action
        self.previous_action = action.copy()
        steering, acceleration = action
        native_action = np.array(
            [
                steering * self.config.steering_scale,
                max(float(acceleration), 0.0),
                max(float(-acceleration), 0.0),
            ],
            dtype=np.float32,
        )
        previous_native_reward = self.prev_reward
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
        outside = abs(x) > self.playfield or abs(y) > self.playfield
        if self.track_layout is not None:
            native_reward = -100.0 if outside else self.reward - previous_native_reward
        off_track_failure = self._consecutive_off_track >= self.config.off_track_grace_steps
        lap_time_profile = self.config.reward_profile in LAP_TIME_REWARDS
        reason = "running"
        if outside:
            reason = "out_of_bounds"
        elif off_track_failure:
            reason = "off_track"
        elif completed and (not off_track or not lap_time_profile):
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
        elif lap_time_profile:
            hull = self.car.hull
            velocity = np.array(hull.linearVelocity)
            forward = np.array(hull.GetWorldVector((0, 1)))
            right = np.array(hull.GetWorldVector((1, 0)))
            beta = math.atan2(float(velocity @ right), max(abs(float(velocity @ forward)), 3.0))
            reward = lap_time_step_reward(
                gain / self.geometry.length,
                on_road=not off_track,
                action_delta=action_delta,
                max_abs_slip=max(abs(value) for value in self.car.telemetry["slip_ratios"]),
                sideslip_rad=beta,
            )
        reward += 100.0 * (reason == "lap_complete")
        reward -= 50.0 * (reason == "off_track") + 100.0 * outside
        return observation, float(reward), terminated, truncated, self._info(reason, native_reward)

    def _info(self, reason: str, native_reward: float) -> dict[str, Any]:
        assert self.geometry is not None and self.car is not None
        return {
            **getattr(self.car, "telemetry", {}),
            "vehicle_model": self.config.vehicle_model,
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


def make_env(
    config: EnvironmentConfig,
    render_mode: str | None = None,
    *,
    track_layout: TrackLayout | None = None,
) -> gym.Env:
    env = NumericCarRacing(config, render_mode=render_mode, track_layout=track_layout)
    if config.control_profile == RACING_RESIDUAL:
        from f1_ai.residual_env import ResidualRacingEnv

        return ResidualRacingEnv(env)
    return env


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
