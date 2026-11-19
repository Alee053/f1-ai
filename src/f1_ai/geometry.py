"""Proyección sobre el eje de un circuito cerrado y seguimiento de una vuelta."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    distance: float
    lateral: float
    tangent: np.ndarray


class CircuitGeometry:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 3:
            raise ValueError("El circuito debe contener al menos tres puntos bidimensionales.")
        if not np.isfinite(self.points).all():
            raise ValueError("El circuito contiene coordenadas no finitas.")
        self.segments = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.segments, axis=1)
        if np.any(self.lengths <= 1e-8):
            raise ValueError("El circuito contiene segmentos de longitud cero.")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.length = float(self.cumulative[-1])
        self.tangents = self.segments / self.lengths[:, None]

    def project(self, position: np.ndarray) -> Projection:
        relative = np.asarray(position) - self.points
        t = np.clip(np.sum(relative * self.segments, axis=1) / self.lengths**2, 0, 1)
        candidates = self.points + t[:, None] * self.segments
        squared_distances = np.sum((np.asarray(position) - candidates) ** 2, axis=1)
        index = int(np.argmin(squared_distances))
        tangent = self.tangents[index]
        right = np.array([tangent[1], -tangent[0]])
        return Projection(
            distance=float(self.cumulative[index] + t[index] * self.lengths[index]),
            lateral=float(np.dot(np.asarray(position) - candidates[index], right)),
            tangent=tangent,
        )

    def sample(self, distances: np.ndarray) -> np.ndarray:
        distances = np.asarray(distances) % self.length
        indices = np.searchsorted(self.cumulative, distances, side="right") - 1
        t = (distances - self.cumulative[indices]) / self.lengths[indices]
        return self.points[indices] + t[..., None] * self.segments[indices]


class TurnPreview:
    """Giro acumulado del asfalto en un intervalo, incluyendo el cierre de vuelta."""

    def __init__(self, road: CircuitGeometry):
        self.length = road.length
        tangent = road.tangents
        following = np.roll(tangent, -1, axis=0)
        cross = tangent[:, 0] * following[:, 1] - tangent[:, 1] * following[:, 0]
        dot = np.sum(tangent * following, axis=1)
        accumulated = np.r_[0.0, np.cumsum(np.abs(np.arctan2(cross, dot)))]
        blend = np.minimum(2.0, road.lengths / 2)
        self.stations = np.r_[
            0.0,
            np.column_stack((road.cumulative[1:] - blend, road.cumulative[1:])).ravel(),
        ]
        self.angles = np.r_[0.0, np.column_stack((accumulated[:-1], accumulated[1:])).ravel()]

    def ahead(self, distance: float, lookahead_m: float) -> float:
        start = distance % self.length
        laps, end = divmod(start + lookahead_m, self.length)
        return float(
            laps * self.angles[-1]
            + np.interp(end, self.stations, self.angles)
            - np.interp(start, self.stations, self.angles)
        )


class LapProgress:
    """Evita premiar oscilaciones, marcha atrás o saltos entre segmentos distantes."""

    def __init__(self, length: float, start: float, minimum_coverage: float):
        self.length = length
        self.start = start
        self.minimum_coverage = minimum_coverage
        self.previous = 0.0
        self.net_distance = 0.0
        self.furthest = 0.0
        self.completed = False

    def update(self, distance: float, coverage: float) -> tuple[float, bool]:
        relative = (distance - self.start) % self.length
        delta = (relative - self.previous + self.length / 2) % self.length - self.length / 2
        crossed_start = self.previous > self.length * 0.75 and relative < self.length * 0.25
        self.previous = relative
        # Un paso físico dura 0.02 s; diez metros exceden el avance normal del modelo.
        if abs(delta) > 10.0:
            return 0.0, False
        self.net_distance += delta
        new_furthest = min(self.length, max(self.furthest, self.net_distance))
        gain = new_furthest - self.furthest
        self.furthest = new_furthest
        self.completed = self.completed or (
            crossed_start
            and delta > 0
            and self.net_distance >= self.minimum_coverage * self.length
            and coverage >= self.minimum_coverage
        )
        return gain, self.completed
