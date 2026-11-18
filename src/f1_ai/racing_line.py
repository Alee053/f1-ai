"""Referencia global aproximada de tiempo de vuelta, sin redes ni simulación.

Se comparan trazadas dentro de un corredor con un modelo puntual por ejes. La
estimación no sustituye una vuelta Box2D ni certifica un óptimo de tiempo.
"""

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from f1_ai.geometry import CircuitGeometry
from f1_ai.vehicle import GRAVITY, FormulaConfig, aerodynamic_forces, normal_loads, tyre_forces

MAX_STEERING_RAD = 0.4


@dataclass(frozen=True)
class ReferenceConfig:
    samples: int = 128
    optimization_steps: int = 12
    speed_limit_m_s: float = 60.0
    grip_fraction: float = 0.75
    road_margin_m: float = 0.75
    braking_fraction: float = 1.0
    time_search_passes: int = 0

    def __post_init__(self):
        if type(self.samples) is not int or not 16 <= self.samples <= 512:
            raise ValueError("samples debe pertenecer a [16, 512].")
        if type(self.optimization_steps) is not int or not 0 <= self.optimization_steps <= 64:
            raise ValueError("optimization_steps debe pertenecer a [0, 64].")
        if type(self.time_search_passes) is not int or not 0 <= self.time_search_passes <= 6:
            raise ValueError("time_search_passes debe pertenecer a [0, 6].")
        for name in ("speed_limit_m_s", "grip_fraction", "road_margin_m", "braking_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} debe ser un número finito.")
        if not 1 <= self.speed_limit_m_s <= 100:
            raise ValueError("speed_limit_m_s debe pertenecer a [1, 100].")
        if not 0 < self.grip_fraction <= 1 or self.road_margin_m < 0:
            raise ValueError("El margen debe ser no negativo y grip_fraction pertenecer a (0, 1].")
        if not 0 < self.braking_fraction <= 1:
            raise ValueError("braking_fraction debe pertenecer a (0, 1].")


def _path_geometry(points):
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(outgoing, axis=1)
    previous = np.roll(lengths, 1)
    chord = np.linalg.norm(incoming + outgoing, axis=1)
    if min(lengths.min(), chord.min()) <= 1e-8:
        return None
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    curvature = 2 * cross / (previous * lengths * chord)
    return lengths, curvature


def _geometric_cost(points, objective):
    geometry = _path_geometry(points)
    if geometry is None:
        return math.inf
    lengths, curvature = geometry
    if objective == "length":
        return float(lengths.sum())
    return float(np.sum(curvature**2 * (lengths + np.roll(lengths, 1)) / 2))


def _improve_offsets(center, normals, limit, steps, objective):
    offsets = np.zeros(len(center))
    epsilon = 0.02
    for _ in range(steps):
        base = _geometric_cost(center + normals * offsets[:, None], objective)
        gradient = np.zeros(len(center))
        for index in range(len(center)):
            trial = offsets.copy()
            trial[index] += epsilon
            plus = _geometric_cost(center + normals * trial[:, None], objective)
            trial[index] -= 2 * epsilon
            minus = _geometric_cost(center + normals * trial[:, None], objective)
            if math.isfinite(plus) and math.isfinite(minus):
                gradient[index] = (plus - minus) / (2 * epsilon)
            elif math.isfinite(plus):
                gradient[index] = (plus - base) / epsilon
            elif math.isfinite(minus):
                gradient[index] = (base - minus) / epsilon
        magnitude = float(np.max(np.abs(gradient)))
        if magnitude < 1e-10:
            break
        # Búsqueda acotada: como máximo 0.5 m por iteración y doce reducciones.
        for scale in (0.5 * 0.5**power for power in range(12)):
            trial = np.clip(offsets - scale * gradient / magnitude, -limit, limit)
            if _geometric_cost(center + normals * trial[:, None], objective) < base - 1e-10:
                offsets = trial
                break
        else:
            break
    return offsets


def longitudinal_limits(vehicle, speed, curvature, grip_fraction):
    """Aceleración y frenada disponibles en m/s² para la envolvente por ejes.

    Se reserva fuerza lateral antes de asignar fuerza longitudinal. Se aproximan
    las cargas sin transferencia longitudinal ni dinámica de giro de las ruedas.
    """
    lateral = speed**2 * abs(curvature)
    loads = normal_loads(vehicle, speed, 0.0, min(lateral, 6 * GRAVITY))
    limits = [tyre_forces(vehicle, 0.0, 0.0, load, 1.0)[2] * grip_fraction for load in loads]
    front, rear = sum(limits[:2]), sum(limits[2:])
    required = vehicle.mass_kg * lateral
    front_lateral = required * vehicle.front_weight_fraction
    rear_lateral = required * (1 - vehicle.front_weight_fraction)
    if front_lateral > front or rear_lateral > rear:
        return 0.0, 0.0, False
    front_available = math.sqrt(max(0.0, front**2 - front_lateral**2))
    rear_available = math.sqrt(max(0.0, rear**2 - rear_lateral**2))
    _, drag = aerodynamic_forces(speed, vehicle)
    resistance = drag + vehicle.rolling_resistance * vehicle.mass_kg * GRAVITY
    drive = min(
        rear_available,
        vehicle.drive_force_limit_n,
        vehicle.power_kw * 1000 * vehicle.drivetrain_efficiency / max(speed, 3.0),
    )
    brake = min(
        vehicle.brake_force_limit_n,
        front_available / vehicle.brake_front_fraction,
        rear_available / (1 - vehicle.brake_front_fraction),
    )
    return (
        max(0.0, (drive - resistance) / vehicle.mass_kg),
        (brake + resistance) / vehicle.mass_kg,
        drive >= resistance,
    )


def _lateral_speed_cap(turn, vehicle, config):
    low, high = 0.0, config.speed_limit_m_s
    for _ in range(24):
        middle = (low + high) / 2
        if longitudinal_limits(vehicle, middle, turn, config.grip_fraction)[2]:
            low = middle
        else:
            high = middle
    return low


def _speed_profile(lengths, curvature, vehicle, config):
    speeds = np.array([_lateral_speed_cap(turn, vehicle, config) for turn in curvature])
    # Las pasadas incluyen la unión final/inicio: se planifica una vuelta lanzada.
    for _ in range(len(speeds)):
        previous = speeds.copy()
        for index in range(len(speeds)):
            next_index = (index + 1) % len(speeds)
            accel, _, _ = longitudinal_limits(
                vehicle, speeds[index], curvature[index], config.grip_fraction
            )
            speeds[next_index] = min(
                speeds[next_index], math.sqrt(speeds[index] ** 2 + 2 * accel * lengths[index])
            )
        for index in reversed(range(len(speeds))):
            next_index = (index + 1) % len(speeds)
            _, brake, _ = longitudinal_limits(
                vehicle, speeds[next_index], curvature[next_index], config.grip_fraction
            )
            speeds[index] = min(
                speeds[index],
                math.sqrt(
                    speeds[next_index] ** 2 + 2 * config.braking_fraction * brake * lengths[index]
                ),
            )
        if np.max(np.abs(speeds - previous)) < 1e-7:
            break
    else:
        raise ValueError("El perfil de velocidad periódico no convergió.")
    if speeds.min() <= 1e-3:
        raise ValueError("No existe un perfil de avance positivo con estos parámetros.")
    lap_time = float(np.sum(2 * lengths / (speeds + np.roll(speeds, -1))))
    return speeds, lap_time


@lru_cache(maxsize=16)
def _speed_cap_table(vehicle, config):
    curvature = np.linspace(0, math.tan(MAX_STEERING_RAD) / vehicle.wheelbase_m, 97)
    limits = np.array([_lateral_speed_cap(turn, vehicle, config) for turn in curvature])
    return curvature, limits


def _time_search(center, normals, width, offsets, vehicle, config):
    """Prueba desplazamientos suaves usando un perfil rápido para ordenar candidatos."""
    turns, limits = _speed_cap_table(vehicle, config)
    accel, brake, _ = longitudinal_limits(vehicle, 20.0, 0.0, config.grip_fraction)
    accel = max(accel, 0.5)
    brake = max(brake * config.braking_fraction, 0.5)
    count = min(24, max(4, len(center) // 6))
    station = np.arange(len(center)) / len(center)
    anchors = np.arange(count) / count
    separation = np.abs((station[:, None] - anchors + 0.5) % 1.0 - 0.5) * count
    basis = 0.5 * (1 + np.cos(np.pi * np.minimum(separation, 1)))

    def estimated_time(lateral):
        shape = _path_geometry(center + normals * lateral[:, None])
        if shape is None:
            return math.inf
        lengths, curvature = shape
        if np.max(np.abs(curvature)) > turns[-1]:
            return math.inf
        distance = np.r_[0, np.cumsum(lengths[:-1])]
        travel = (distance[:, None] - distance) % float(lengths.sum())
        speed_squared = np.interp(np.abs(curvature), turns, limits) ** 2
        reachable = speed_squared + np.minimum(2 * accel * travel, 2 * brake * travel.T)
        speeds = np.sqrt(np.min(reachable, axis=1))
        return float(np.sum(2 * lengths / (speeds + np.roll(speeds, -1))))

    time = estimated_time(offsets)
    for stage in range(config.time_search_passes):
        adjustment = 1.5 * 0.5**stage
        previous_time = time
        for direction in basis.T:
            for sign in (-1, 1):
                trial = np.clip(offsets + sign * adjustment * direction, -width, width)
                trial_time = estimated_time(trial)
                if trial_time < time - 1e-9:
                    offsets, time = trial, trial_time
        if time < previous_time:
            yield f"minimum_time_pass_{stage + 1}", offsets


def _inside_corridor(points, geometry, usable_width):
    # Se comprueban también puntos interiores de cada segmento de la trazada.
    for fraction in (0.0, 0.25, 0.5, 0.75):
        samples = points * (1 - fraction) + np.roll(points, -1, axis=0) * fraction
        for point in samples:
            projection = geometry.project(point)
            nearest = geometry.sample(np.array([projection.distance]))[0]
            if np.linalg.norm(point - nearest) > usable_width + 1e-6:
                return False
    return True


@dataclass(frozen=True)
class RacingPlan:
    centerline: np.ndarray
    points: np.ndarray
    offsets_m: np.ndarray
    curvature_m_inv: np.ndarray
    speed_m_s: np.ndarray
    estimated_lap_seconds: float
    centerline_estimated_lap_seconds: float | None
    candidate: str
    half_width_m: float
    vehicle: FormulaConfig
    config: ReferenceConfig

    def to_dict(self):
        return {
            "method": "bounded_candidates_with_axle_speed_envelope_v1",
            "optimality_proven": False,
            "box2d_driving_validated": False,
            "information": "full circuit map, physical parameters and unnormalized vehicle state",
            "estimated_lap_seconds": self.estimated_lap_seconds,
            "centerline_estimated_lap_seconds": self.centerline_estimated_lap_seconds,
            "selected_candidate": self.candidate,
            "half_width_m": self.half_width_m,
            "vehicle": asdict(self.vehicle),
            "settings": asdict(self.config),
            "centerline": self.centerline.tolist(),
            "points": self.points.tolist(),
            "offsets_m": self.offsets_m.tolist(),
            "curvature_m_inv": self.curvature_m_inv.tolist(),
            "speed_m_s": self.speed_m_s.tolist(),
        }


def plan_reference(
    centerline: np.ndarray,
    half_width_m: float,
    vehicle: FormulaConfig,
    config: ReferenceConfig = ReferenceConfig(),
) -> RacingPlan:
    """Compara candidatos por tiempo estimado, sin premio por cercanía al centro.

    El corredor se reduce por semianchura del coche y margen. No se certifica la
    huella completa de la carrocería ni la factibilidad de la dinámica Box2D.
    """
    if (
        isinstance(half_width_m, bool)
        or not isinstance(half_width_m, (int, float))
        or not math.isfinite(half_width_m)
    ):
        raise ValueError("La semianchura de pista debe ser finita.")
    usable = half_width_m - vehicle.width_m / 2 - config.road_margin_m
    if usable <= 0:
        raise ValueError("El coche y el margen no caben en el corredor.")
    geometry = CircuitGeometry(centerline)
    distances = np.linspace(0, geometry.length, config.samples, endpoint=False)
    center = geometry.sample(distances)
    tangents = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    if norms.min() <= 1e-8:
        raise ValueError("La pista tiene cambios de sentido degenerados.")
    tangents /= norms[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    curved = _improve_offsets(center, normals, usable, config.optimization_steps, "curvature")
    short = _improve_offsets(center, normals, usable, config.optimization_steps, "length")
    candidates = [
        ("centerline", np.zeros(config.samples)),
        ("left_offset", np.full(config.samples, -usable)),
        ("right_offset", np.full(config.samples, usable)),
        ("minimum_curvature_candidate", curved),
        ("short_path_candidate", short),
        ("half_curvature_candidate", 0.5 * curved),
        ("half_short_path_candidate", 0.5 * short),
    ]

    def evaluate(name, offsets):
        points = center + normals * offsets[:, None]
        shape = _path_geometry(points)
        if shape is None or not _inside_corridor(points, geometry, usable):
            return None
        lengths, curvature = shape
        if np.max(np.abs(curvature)) > math.tan(MAX_STEERING_RAD) / vehicle.wheelbase_m:
            return None
        speeds, lap_time = _speed_profile(lengths, curvature, vehicle, config)
        return lap_time, name, points, offsets, curvature, speeds

    best = None
    center_time = None
    for name, offsets in candidates:
        candidate = evaluate(name, offsets)
        if candidate is None:
            continue
        if name == "centerline":
            center_time = candidate[0]
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("Ninguna trazada candidata respeta el corredor y el límite de dirección.")
    if config.time_search_passes:
        for name, offsets in _time_search(center, normals, usable, best[3], vehicle, config):
            candidate = evaluate(name, offsets)
            if candidate is not None and candidate[0] < best[0]:
                best = candidate
    lap_time, name, points, offsets, curvature, speeds = best
    arrays = [
        np.array(values, dtype=np.float64, copy=True)
        for values in (center, points, offsets, curvature, speeds)
    ]
    for array in arrays:
        array.setflags(write=False)
    return RacingPlan(*arrays, lap_time, center_time, name, half_width_m, vehicle, config)
