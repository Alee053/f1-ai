"""Seguimiento del plan global con estado SI privilegiado, sin red neuronal."""

import math
from dataclasses import dataclass

import numpy as np

from f1_ai.geometry import CircuitGeometry
from f1_ai.racing_line import MAX_STEERING_RAD, RacingPlan, longitudinal_limits
from f1_ai.vehicle import GRAVITY, aerodynamic_forces


@dataclass(frozen=True)
class ReferenceState:
    position: tuple[float, float]
    heading_rad: float
    velocity: tuple[float, float]
    yaw_rate_rad_s: float
    slip_ratios: tuple[float, float, float, float]
    locked_wheels: tuple[bool, bool, bool, bool]

    @classmethod
    def from_car(cls, car):
        return cls(
            position=tuple(car.hull.position),
            heading_rad=float(car.hull.angle),
            velocity=tuple(car.hull.linearVelocity),
            yaw_rate_rad_s=float(car.hull.angularVelocity),
            slip_ratios=car.telemetry["slip_ratios"],
            locked_wheels=car.telemetry["locked_wheels"],
        )

    def __post_init__(self):
        for name, size in (("position", 2), ("velocity", 2), ("slip_ratios", 4)):
            values = np.asarray(getattr(self, name))
            real_numbers = np.issubdtype(values.dtype, np.floating) or np.issubdtype(
                values.dtype, np.integer
            )
            if values.shape != (size,) or not real_numbers:
                raise ValueError(f"{name} debe contener {size} números finitos.")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} debe contener {size} números finitos.")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        for value in (self.heading_rad, self.yaw_rate_rad_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value)
            ):
                raise ValueError("Los ángulos y velocidades deben ser finitos.")
        if len(self.locked_wheels) != 4 or any(
            not isinstance(value, (bool, np.bool_)) for value in self.locked_wheels
        ):
            raise ValueError("locked_wheels debe contener cuatro indicadores booleanos.")
        object.__setattr__(
            self, "locked_wheels", tuple(bool(value) for value in self.locked_wheels)
        )


@dataclass(frozen=True)
class FollowerConfig:
    lookahead_seconds: float = 0.35
    speed_gain: float = 0.8
    steering_rate_rad_s: float = 1.5
    pedal_rate_s: float = 2.0
    brake_rate_s: float = 4.0
    drive_slip_limit: float = 0.15
    drive_slip_cut: float = 0.6

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} debe ser positivo y finito.")
        if self.steering_rate_rad_s > 3 or self.drive_slip_cut <= self.drive_slip_limit:
            raise ValueError("El límite de giro o los umbrales de slip no son válidos.")


class RacingController:
    """Pure pursuit de una trazada planificada y seguimiento de su velocidad.

    La regulación de slip y bloqueo pertenece al controlador. La física mantiene
    su ausencia de TC y ABS. Se debe reiniciar el controlador entre episodios.
    """

    def __init__(self, plan: RacingPlan, config: FollowerConfig = FollowerConfig()):
        self.plan = plan
        self.config = config
        self.geometry = CircuitGeometry(plan.points)
        self._periodic_speed = np.r_[plan.speed_m_s, plan.speed_m_s[0]]
        self.reset()

    def reset(self):
        self._last_action = np.zeros(2, dtype=np.float64)

    def action(
        self, state: ReferenceState, dt: float = 0.02, *, previous_action: np.ndarray | None = None
    ) -> np.ndarray:
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not math.isfinite(dt)
            or not 0 < dt <= 0.2
        ):
            raise ValueError("dt debe pertenecer a (0, 0.2] segundos.")
        if previous_action is not None:
            previous_action = np.asarray(previous_action, dtype=np.float64)
            if (
                previous_action.shape != (2,)
                or not np.isfinite(previous_action).all()
                or np.any(np.abs(previous_action) > 1)
            ):
                raise ValueError("previous_action debe contener dos valores finitos en [-1, 1].")
            # DAgger debe limitar cambios respecto al mando aplicado por el alumno.
            self._last_action = previous_action.copy()
        vehicle = self.plan.vehicle
        forward = np.array([-math.sin(state.heading_rad), math.cos(state.heading_rad)])
        right = np.array([forward[1], -forward[0]])
        speed = max(0.0, float(np.dot(state.velocity, forward)))
        projection = self.geometry.project(np.array(state.position))
        lookahead = float(np.clip(speed * self.config.lookahead_seconds, 6.0, 24.0))
        target = self.geometry.sample(np.array([projection.distance + lookahead]))[0]
        relative = target - state.position
        curvature_right = 2 * float(relative @ right) / max(float(relative @ relative), 1.0)
        steering_angle = math.atan(vehicle.wheelbase_m * curvature_right)
        steering_angle += 0.08 * (speed * curvature_right + state.yaw_rate_rad_s)
        steering = float(np.clip(steering_angle / MAX_STEERING_RAD, -1, 1))

        desired_speed = float(
            np.interp(projection.distance, self.geometry.cumulative, self._periodic_speed)
        )
        index = min(
            int(np.searchsorted(self.geometry.cumulative, projection.distance, side="right") - 1),
            len(self.plan.points) - 1,
        )
        next_index = (index + 1) % len(self.plan.points)
        feedforward = (self.plan.speed_m_s[next_index] ** 2 - self.plan.speed_m_s[index] ** 2) / (
            2 * self.geometry.lengths[index]
        )
        actual_curvature = abs(state.yaw_rate_rad_s) / max(speed, 3.0)
        accel_limit, brake_limit, feasible = longitudinal_limits(
            vehicle, speed, actual_curvature, self.plan.config.grip_fraction
        )
        if feasible:
            acceleration = float(
                np.clip(
                    feedforward + self.config.speed_gain * (desired_speed - speed),
                    -brake_limit,
                    accel_limit,
                )
            )
            _, drag = aerodynamic_forces(speed, vehicle)
            force = (
                vehicle.mass_kg * acceleration
                + drag
                + vehicle.rolling_resistance * vehicle.mass_kg * GRAVITY
            )
            if force >= 0:
                available = min(
                    vehicle.drive_force_limit_n,
                    vehicle.power_kw * 1000 * vehicle.drivetrain_efficiency / max(speed, 3.0),
                )
                pedal = min(force / available, 1.0)
            else:
                pedal = max(force / vehicle.brake_force_limit_n, -1.0)
        else:
            # Recuperación moderada fuera de la envolvente nominal; no se certifica estabilidad.
            pedal = -0.1

        rear_slip = max(abs(value) for value in state.slip_ratios[2:])
        if pedal > 0:
            factor = np.clip(
                (self.config.drive_slip_cut - rear_slip)
                / (self.config.drive_slip_cut - self.config.drive_slip_limit),
                0,
                1,
            )
            pedal *= factor
        elif any(state.locked_wheels):
            pedal *= 0.5
        steering_delta = self.config.steering_rate_rad_s * dt / MAX_STEERING_RAD
        pedal_delta = (
            self.config.brake_rate_s if pedal < self._last_action[1] else self.config.pedal_rate_s
        ) * dt
        action = self._last_action + np.clip(
            np.array([steering, pedal]) - self._last_action,
            [-steering_delta, -pedal_delta],
            [steering_delta, pedal_delta],
        )
        if rear_slip >= self.config.drive_slip_cut:
            action[1] = min(action[1], 0.0)
        self._last_action = np.clip(action, -1, 1)
        return self._last_action.astype(np.float32)
