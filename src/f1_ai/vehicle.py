"""Parámetros SI y fuerzas del modelo planar Formula v1.

Los coeficientes aero y de neumáticos son aproximaciones ajustables, no datos
identificados de un monoplaza. Las ruedas siguen pudiendo patinar y bloquearse.
"""

import math
from dataclasses import dataclass, fields

GRAVITY = 9.81


@dataclass(frozen=True)
class FormulaConfig:
    mass_kg: float = 770.0
    power_kw: float = 750.0
    wheelbase_m: float = 3.4
    width_m: float = 1.9
    tyre_radius_m: float = 0.36
    wheel_mass_kg: float = 12.0
    wheel_inertia_kg_m2: float = 1.6
    cg_height_m: float = 0.30
    front_weight_fraction: float = 0.46
    front_aero_fraction: float = 0.45
    brake_front_fraction: float = 0.57
    grip_coefficient: float = 1.65
    tyre_load_sensitivity: float = 0.10
    sliding_grip_loss: float = 0.20
    air_density_kg_m3: float = 1.225
    downforce_cla_m2: float = 5.0
    drag_cda_m2: float = 1.2
    drivetrain_efficiency: float = 0.90
    drive_force_limit_n: float = 14000.0
    brake_force_limit_n: float = 30000.0
    rolling_resistance: float = 0.015
    grass_grip_fraction: float = 0.45

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} debe ser un número finito.")
            if not math.isfinite(value):
                raise ValueError(f"{field.name} debe ser un número finito.")
        limits = {
            "mass_kg": (300, 2000),
            "power_kw": (1, 1500),
            "wheelbase_m": (1.5, 5),
            "width_m": (1, 3),
            "tyre_radius_m": (0.2, 0.6),
            "wheel_mass_kg": (1, 40),
            "wheel_inertia_kg_m2": (0.1, 10),
            "cg_height_m": (0.05, 0.8),
            "front_weight_fraction": (0.3, 0.7),
            "front_aero_fraction": (0.2, 0.8),
            "brake_front_fraction": (0.3, 0.8),
            "grip_coefficient": (0.2, 3),
            "tyre_load_sensitivity": (0, 0.3),
            "sliding_grip_loss": (0, 0.5),
            "air_density_kg_m3": (0.5, 1.5),
            "downforce_cla_m2": (0, 12),
            "drag_cda_m2": (0.1, 4),
            "drivetrain_efficiency": (0.5, 1),
            "drive_force_limit_n": (500, 40000),
            "brake_force_limit_n": (1000, 80000),
            "rolling_resistance": (0, 0.05),
            "grass_grip_fraction": (0.1, 1),
        }
        for name, (low, high) in limits.items():
            if not low <= getattr(self, name) <= high:
                raise ValueError(f"{name} debe pertenecer a [{low}, {high}].")
        if 4 * self.wheel_mass_kg >= self.mass_kg:
            raise ValueError("La masa de las ruedas debe ser menor que la masa total.")


def aerodynamic_forces(speed_m_s: float, config: FormulaConfig) -> tuple[float, float]:
    """Magnitudes de downforce y drag en N; la masa inercial permanece fija."""
    pressure = 0.5 * config.air_density_kg_m3 * speed_m_s**2
    return pressure * config.downforce_cla_m2, pressure * config.drag_cda_m2


def normal_loads(
    config: FormulaConfig, speed_m_s: float, longitudinal_accel: float, lateral_accel: float
) -> tuple[float, float, float, float]:
    """Carga vertical FL, FR, RL, RR; transferencia cuasiestática, sin suspensión 3D."""
    downforce, _ = aerodynamic_forces(speed_m_s, config)
    weight = config.mass_kg * GRAVITY
    total = weight + downforce
    transfer = config.mass_kg * longitudinal_accel * config.cg_height_m / config.wheelbase_m
    front = min(
        total,
        max(
            0.0,
            weight * config.front_weight_fraction
            + downforce * config.front_aero_fraction
            - transfer,
        ),
    )
    rear = total - front
    loads = []
    for axle_load, share, tyre_width in [
        (front, config.front_weight_fraction, 0.28),
        (rear, 1 - config.front_weight_fraction, 0.375),
    ]:
        delta = (
            config.mass_kg
            * lateral_accel
            * config.cg_height_m
            * share
            / (config.width_m - tyre_width)
        )
        delta = min(axle_load / 2, max(-axle_load / 2, delta))
        loads.extend((axle_load / 2 + delta, axle_load / 2 - delta))
    return tuple(loads)


def tyre_forces(
    config: FormulaConfig, slip_ratio: float, slip_angle: float, load_n: float, surface: float
) -> tuple[float, float, float]:
    """Curvas suaves de slip, caída por deslizamiento y círculo combinado de fuerzas."""
    if load_n <= 0:
        return 0.0, 0.0, 0.0
    reference_load = config.mass_kg * GRAVITY / 4
    load_factor = max(load_n / reference_load, 0.01) ** -config.tyre_load_sensitivity
    excess_slip = max(abs(slip_ratio) - 0.15, abs(slip_angle) - 0.12, 0.0)
    sliding = 1 - config.sliding_grip_loss * (1 - math.exp(-3 * excess_slip))
    limit = config.grip_coefficient * load_factor * surface * load_n * sliding
    longitudinal = limit * math.tanh(12 * slip_ratio)
    lateral = -limit * math.tanh(10 * slip_angle)
    demand = math.hypot(longitudinal, lateral)
    if demand > limit:
        longitudinal *= limit / demand
        lateral *= limit / demand
    return longitudinal, lateral, limit
