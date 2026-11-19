"""Circuito cerrado en metros, ordenado desde la línea de salida."""

import json
from pathlib import Path

import numpy as np

from f1_ai.geometry import CircuitGeometry


class TrackLayout:
    def __init__(self, name: str, points: np.ndarray, width_m: float):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("El circuito necesita un nombre.")
        if type(width_m) not in (int, float) or not np.isfinite(width_m) or width_m <= 0:
            raise ValueError("El ancho del circuito debe ser positivo y finito.")
        self.name = name
        self.width_m = float(width_m)
        self.geometry = CircuitGeometry(points)
        tangents = self.geometry.tangents + np.roll(self.geometry.tangents, 1, axis=0)
        lengths = np.linalg.norm(tangents, axis=1)
        if np.any(lengths < 1e-6):
            raise ValueError("El eje del circuito no puede invertir el sentido en un punto.")
        tangents /= lengths[:, None]
        self.angles = np.arctan2(-tangents[:, 0], tangents[:, 1])
        right = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        offset = right * self.width_m / 2
        center = self.geometry.points
        self.edges = np.stack((center - offset, center + offset), axis=1)

    @classmethod
    def load(cls, path: Path) -> "TrackLayout":
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or not {"name", "points", "width_m"}.issubset(data):
            raise ValueError("El JSON del circuito requiere name, points y width_m.")
        return cls(data["name"], np.asarray(data["points"], dtype=np.float64), data["width_m"])
