"""Datos locales de la trazada. La política decide los mandos."""

from functools import lru_cache

import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

from f1_ai.geometry import CircuitGeometry
from f1_ai.racing_line import plan_reference

PREVIEW_M = np.array([6.0, 15.0, 30.0, 60.0, 100.0])


class CorridorFeatures:
    """Bordes reales del asfalto muestreados en las estaciones del eje."""

    def __init__(self, road: CircuitGeometry, road_angles: np.ndarray):
        self.road = road
        offset = TRACK_WIDTH * np.column_stack((np.cos(road_angles), np.sin(road_angles)))
        self.edges = np.stack((road.points - offset, road.points + offset), axis=1)

    def observation(self, distance, position, forward, right):
        road = self.road
        stations = (distance + PREVIEW_M) % road.length
        segment = np.searchsorted(road.cumulative, stations, side="right") - 1
        fraction = (stations - road.cumulative[segment]) / road.lengths[segment]
        begin = self.edges[segment]
        end = self.edges[(segment + 1) % len(road.points)]
        points = begin + (end - begin) * fraction[:, None, None]
        relative = (points - position) @ np.column_stack((right, forward))
        return (relative / (PREVIEW_M + TRACK_WIDTH)[:, None, None]).ravel()


class ReferenceTimeProgress:
    """Tiempo acumulado de referencia en estaciones equiespaciadas del eje."""

    def __init__(self, plan):
        lengths = CircuitGeometry(plan.points).lengths
        speeds = plan.speed_m_s
        seconds = 2 * lengths / (speeds + np.roll(speeds, -1))
        elapsed = np.r_[0, np.cumsum(seconds)]
        self.fractions = elapsed / elapsed[-1]
        self.stations = np.linspace(0, 1, len(elapsed))

    def fraction(self, track_fraction):
        laps, phase = divmod(track_fraction, 1.0)
        return laps + float(np.interp(phase, self.stations, self.fractions))

    def adjustment(self, start, before, after, length):
        estimated = self.fraction((start + after) / length) - self.fraction(
            (start + before) / length
        )
        return estimated - (after - before) / length


@lru_cache(maxsize=96)
def track_plan(points_bytes, vehicle, settings):
    points = np.frombuffer(points_bytes, dtype=np.float64).reshape(-1, 2)
    return plan_reference(points, TRACK_WIDTH, vehicle, settings)


class PlanFeatures:
    def __init__(self, plan, lookahead_seconds):
        self.geometry = CircuitGeometry(plan.points)
        self.speeds = np.r_[plan.speed_m_s, plan.speed_m_s[0]]
        self.curvature = np.r_[plan.curvature_m_inv, plan.curvature_m_inv[0]]
        self.segment_acceleration = np.diff(self.speeds**2) / (2 * self.geometry.lengths)
        self.lookahead_seconds = lookahead_seconds

    def observation(self, position, forward, right, speed):
        geometry = self.geometry
        projection = geometry.project(position)
        preview = (projection.distance + PREVIEW_M) % geometry.length
        future = geometry.sample(preview) - position
        target_speed = np.interp(projection.distance, geometry.cumulative, self.speeds)
        future_speed = np.interp(preview, geometry.cumulative, self.speeds)
        future_curvature = np.interp(preview, geometry.cumulative, self.curvature)
        segment = min(
            len(self.segment_acceleration) - 1,
            int(np.searchsorted(geometry.cumulative, projection.distance, side="right") - 1),
        )
        distance = float(np.clip(speed * self.lookahead_seconds, 6.0, 24.0))
        target = geometry.sample(projection.distance + distance) - position
        preview_rows = np.column_stack(
            (
                future @ right / PREVIEW_M,
                future @ forward / PREVIEW_M,
                future_speed / 60,
                12 * future_curvature,
            )
        )
        return np.concatenate(
            (
                [
                    projection.lateral / TRACK_WIDTH,
                    (target_speed - speed) / 20,
                    self.segment_acceleration[segment] / 15,
                    target @ right / distance,
                    target @ forward / distance,
                ],
                preview_rows.ravel(),
            )
        )
