"""Cámara en perspectiva sobre un mundo de física planar."""

import math

import numpy as np
import pygame

CAMERA_MODES = ("top", "chase")


def validate_view(mode: str, grid_m: float):
    if mode not in CAMERA_MODES:
        raise ValueError("La vista debe ser top o chase.")
    if (
        isinstance(grid_m, bool)
        or not isinstance(grid_m, (int, float))
        or not math.isfinite(grid_m)
        or not 0.5 <= grid_m <= 20
    ):
        raise ValueError("La cuadrícula debe medir entre 0.5 y 20 metros.")


def _lattice_segments(origin, spacing, radius):
    low = np.floor((origin - radius) / spacing) * spacing
    high = np.ceil((origin + radius) / spacing) * spacing
    x = np.arange(low[0], high[0] + spacing / 2, spacing)
    y = np.arange(low[1], high[1] + spacing / 2, spacing)
    vertical = np.column_stack((x, np.full_like(x, low[1]), x, np.full_like(x, high[1])))
    horizontal = np.column_stack((np.full_like(y, low[0]), y, np.full_like(y, high[0]), y))
    return np.concatenate((vertical, horizontal)).reshape(-1, 2, 2)


def _clip_lines(lines, near, far):
    depths = lines[:, :, 1]
    lines = lines[(depths.max(axis=1) >= near) & (depths.min(axis=1) <= far)]
    start, end = lines[:, :1], lines[:, 1:]
    delta = end - start
    clipped_depth = np.clip(lines[:, :, 1], near, far)
    factor = np.divide(
        clipped_depth - start[:, :, 1],
        delta[:, :, 1],
        out=np.zeros_like(clipped_depth),
        where=delta[:, :, 1] != 0,
    )
    return start + factor[:, :, None] * delta


def _clip_polygon(points, near):
    if np.all(points[:, 1] >= near):
        return points
    result = []
    previous = points[-1]
    for point in points:
        inside = point[1] >= near
        if inside != (previous[1] >= near):
            delta = point - previous
            result.append(previous + delta * ((near - previous[1]) / delta[1]))
        if inside:
            result.append(point)
        previous = point
    return np.asarray(result)


class ChaseCamera:
    """Proyección a 27.5 grados hacia abajo, con referencias fijas al suelo."""

    height_m = 16.0
    behind_m = 17.0
    pitch_rad = math.radians(27.5)
    near_m = 1.0

    def __init__(self, grid_m: float = 1.0):
        validate_view("chase", grid_m)
        self.grid_m = grid_m
        self._road_source = None

    def draw(self, surface, env, *, guide=None):
        car = env.car
        if car is None:
            raise ValueError("Se requiere una pista inicializada para dibujar.")
        width, height = surface.get_size()
        forward = np.array(car.hull.GetWorldVector((0, 1)))
        right = np.array(car.hull.GetWorldVector((1, 0)))
        origin = np.array(car.hull.position) - self.behind_m * forward
        basis = np.column_stack((right, forward))
        sine, cosine = math.sin(self.pitch_rad), math.cos(self.pitch_rad)
        focal = 1.25 * height

        def local(points):
            return (np.asarray(points, dtype=np.float64) - origin) @ basis

        def screen(points):
            x, ahead = points[..., 0], points[..., 1]
            depth = cosine * ahead + sine * self.height_m
            up = sine * ahead - cosine * self.height_m
            return np.stack(
                (width / 2 + focal * x / depth, 0.42 * height - focal * up / depth), axis=-1
            )

        def draw_lines(world, near, far, color):
            pixels = screen(_clip_lines(local(world), near, far))
            for start, end in pixels.tolist():
                pygame.draw.line(surface, color, start, end)

        def draw_polygon(points, color):
            clipped = _clip_polygon(points, self.near_m)
            if len(clipped) < 3:
                return
            pixels = screen(clipped)
            if (pixels.max(axis=0) >= 0).all() and (pixels.min(axis=0) < (width, height)).all():
                pygame.draw.polygon(surface, color, pixels.tolist())

        surface.fill((47, 77, 58))
        fine = _lattice_segments(origin, self.grid_m, 75)
        coarse = _lattice_segments(origin, self.grid_m * 5, 140)
        draw_lines(fine, self.near_m, 38, (64, 101, 76))
        draw_lines(fine, 38, 68, (55, 88, 67))
        draw_lines(coarse, self.near_m, 145, (73, 113, 86))

        if self._road_source is not env.road_poly:
            self._road_source = env.road_poly
            self._road_points = np.array([points for points, _ in env.road_poly])
            self._road_colors = [tuple(int(value) for value in color) for _, color in env.road_poly]
        road = local(self._road_points)
        depth = road[:, :, 1]
        visible = np.flatnonzero((depth.max(axis=1) >= self.near_m) & (depth.min(axis=1) < 145))
        for index in visible:
            draw_polygon(road[index], self._road_colors[index])

        if guide is not None:
            segments = np.stack((guide, np.roll(guide, -1, axis=0)), axis=1)
            draw_lines(segments, self.near_m, 120, (218, 178, 91))
        for body in car.drawlist:
            color = tuple(int(value * 255) for value in body.color)
            for fixture in body.fixtures:
                vertices = [tuple(body.transform * v) for v in fixture.shape.vertices]
                draw_polygon(local(vertices), color)
        return surface
