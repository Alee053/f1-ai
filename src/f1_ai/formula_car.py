"""Adaptador Box2D del modelo Formula v1, con fuerzas SI y ruedas independientes.

Reutiliza la carrocería y las trazas de Car de Gymnasium, de licencia MIT.
La dinámica de esa clase se sustituye, sin modificar la biblioteca instalada.
"""

import math

import numpy as np
from Box2D.b2 import polygonShape, revoluteJointDef
from gymnasium.envs.box2d.car_dynamics import Car

from f1_ai.vehicle import GRAVITY, FormulaConfig, aerodynamic_forces, normal_loads, tyre_forces


class FormulaCar(Car):
    def __init__(
        self, world, init_angle, init_x, init_y, config: FormulaConfig, *, initial_speed=0.0
    ):
        super().__init__(world, init_angle, init_x, init_y)
        self.config = config
        # Se conservan los polígonos visuales, redimensionados a la batalla y ancho elegidos.
        for fixture in list(self.hull.fixtures):
            vertices = [
                (
                    v[0] * config.width_m / 2.4,
                    (v[1] + 1.64) * config.wheelbase_m / 3.24
                    - config.wheelbase_m * config.front_weight_fraction,
                )
                for v in fixture.shape.vertices
            ]
            self.hull.DestroyFixture(fixture)
            self.hull.CreateFixture(shape=polygonShape(vertices=vertices), density=1.0)
        self.hull.ResetMassData()
        chassis_mass = config.mass_kg - 4 * config.wheel_mass_kg
        density_scale = chassis_mass / self.hull.mass
        for fixture in self.hull.fixtures:
            fixture.density *= density_scale
        self.hull.ResetMassData()
        mass_data = self.hull.massData
        mass_data.center = (0, 0)
        self.hull.massData = mass_data
        for index, wheel in enumerate(self.wheels):
            world.DestroyJoint(wheel.joint)
            width = 0.28 if index < 2 else 0.375
            x = (-1 if index % 2 == 0 else 1) * (config.width_m - width) / 2
            y = config.wheelbase_m * (
                1 - config.front_weight_fraction if index < 2 else -config.front_weight_fraction
            )
            wheel.position = self.hull.GetWorldPoint((x, y))
            wheel.angle = init_angle
            for fixture in list(wheel.fixtures):
                wheel.DestroyFixture(fixture)
            vertices = [
                (-width / 2, -config.tyre_radius_m),
                (width / 2, -config.tyre_radius_m),
                (width / 2, config.tyre_radius_m),
                (-width / 2, config.tyre_radius_m),
            ]
            wheel.CreateFixture(
                shape=polygonShape(vertices=vertices),
                density=config.wheel_mass_kg / (2 * config.tyre_radius_m * width),
                categoryBits=0x0020,
                maskBits=0x001,
                restitution=0,
            )
            wheel.wheel_rad = config.tyre_radius_m
            wheel.tyre_width = width
            wheel.joint = world.CreateJoint(
                revoluteJointDef(
                    bodyA=self.hull,
                    bodyB=wheel,
                    localAnchorA=(x, y),
                    localAnchorB=(0, 0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=20000,
                    motorSpeed=0,
                    lowerAngle=-0.4,
                    upperAngle=0.4,
                )
            )
        velocity = self.hull.GetWorldVector((0, initial_speed))
        if initial_speed:
            self.hull.linearVelocity = velocity
            for wheel in self.wheels:
                wheel.linearVelocity = velocity
                wheel.omega = initial_speed / wheel.wheel_rad
        self._previous_velocity = np.array(velocity)
        self._acceleration = np.zeros(2)
        self.telemetry = {}
        self.step(0.0)

    def gas(self, value):
        # Sin TC: el par no se recorta al límite de adherencia.
        for wheel in self.wheels[2:]:
            wheel.gas = float(np.clip(value, 0, 1))

    def step(self, dt):
        config = self.config
        velocity = np.array(self.hull.linearVelocity)
        speed = float(np.linalg.norm(velocity))
        forward = np.array(self.hull.GetWorldVector((0, 1)))
        right = np.array(self.hull.GetWorldVector((1, 0)))
        if dt:
            measured = (velocity - self._previous_velocity) / dt
            self._acceleration = 0.8 * self._acceleration + 0.2 * measured
        self._previous_velocity = velocity.copy()
        ax = float(np.clip(self._acceleration @ forward, -4 * GRAVITY, 3 * GRAVITY))
        ay = float(np.clip(self._acceleration @ right, -6 * GRAVITY, 6 * GRAVITY))
        loads = normal_loads(config, speed, ax, ay)
        downforce, drag = aerodynamic_forces(speed, config)
        slips, locks, limits = [], [], []
        power_w = 0.0
        for index, wheel in enumerate(self.wheels):
            difference = wheel.steer - wheel.joint.angle
            wheel.joint.motorSpeed = float(np.clip(50 * difference, -3, 3))
            wheel_forward = np.array(wheel.GetWorldVector((0, 1)))
            wheel_right = np.array(wheel.GetWorldVector((1, 0)))
            wheel_velocity = np.array(wheel.linearVelocity)
            vf = float(wheel_velocity @ wheel_forward)
            vs = float(wheel_velocity @ wheel_right)
            surface = max(
                (float(tile.road_friction) for tile in wheel.tiles),
                default=config.grass_grip_fraction,
            )
            angle = math.atan2(vs, max(abs(vf), 3.0))
            fx_sum = fy_sum = limit = slip = 0.0
            steps = 10 if dt else 1
            interval = dt / steps
            axle_fraction = (
                config.brake_front_fraction if index < 2 else 1 - config.brake_front_fraction
            )
            brake_torque = config.brake_force(wheel.brake) * axle_fraction * wheel.wheel_rad / 2
            for _ in range(steps):
                slip = (wheel.omega * wheel.wheel_rad - vf) / max(abs(vf), 3.0)
                fx, fy, limit = tyre_forces(config, slip, angle, loads[index], surface)
                if dt:
                    # Evita invertir la velocidad por una fuerza disipativa en un solo paso.
                    fy = float(
                        np.clip(
                            fy,
                            -config.mass_kg / 4 * abs(vs) / dt,
                            config.mass_kg / 4 * abs(vs) / dt,
                        )
                    )
                    if fx * vf < 0:
                        fx = math.copysign(min(abs(fx), config.mass_kg / 4 * abs(vf) / dt), fx)
                drive_torque = 0.0
                if index >= 2:
                    drive_torque = wheel.gas * min(
                        config.drive_force_limit_n * wheel.wheel_rad / 2,
                        config.power_kw
                        * 1000
                        * config.drivetrain_efficiency
                        / 2
                        / max(abs(wheel.omega), 1.0),
                    )
                power_w += drive_torque * abs(wheel.omega) / steps
                free_omega = (
                    wheel.omega
                    + interval * (drive_torque - fx * wheel.wheel_rad) / config.wheel_inertia_kg_m2
                )
                brake_delta = interval * brake_torque / config.wheel_inertia_kg_m2
                wheel.omega = math.copysign(max(0.0, abs(free_omega) - brake_delta), free_omega)
                fx_sum += fx / steps
                fy_sum += fy / steps
            wheel.phase += wheel.omega * dt
            if dt:
                wheel.ApplyForceToCenter(tuple(fx_sum * wheel_forward + fy_sum * wheel_right), True)
            slips.append(slip)
            locks.append(
                bool(
                    wheel.brake > 0.05
                    and abs(vf) > 3
                    and abs(wheel.omega * wheel.wheel_rad) < 0.15 * abs(vf)
                )
            )
            limits.append(limit)
            if dt and (abs(slip) > 0.3 or abs(angle) > 0.15) and speed > 2:
                if wheel.skid_start is None:
                    wheel.skid_start = tuple(wheel.position)
                elif wheel.skid_particle is not None and len(wheel.skid_particle.poly) < 30:
                    wheel.skid_particle.poly.append(tuple(wheel.position))
                else:
                    wheel.skid_particle = self._create_particle(
                        wheel.skid_start, tuple(wheel.position), not bool(wheel.tiles)
                    )
                    wheel.skid_start = tuple(wheel.position)
            else:
                wheel.skid_start = wheel.skid_particle = None
        if dt and speed > 0.01:
            resistance = min(
                drag + config.rolling_resistance * config.mass_kg * GRAVITY,
                config.mass_kg * speed / dt,
            )
            self.hull.ApplyForceToCenter(tuple(-resistance * velocity / speed), True)
        self.telemetry = {
            "speed_kmh": speed * 3.6,
            "downforce_n": downforce,
            "drag_n": drag,
            "normal_loads_n": loads,
            "slip_ratios": tuple(slips),
            "locked_wheels": tuple(locks),
            "grip_limits_n": tuple(limits),
            "wheel_power_kw": power_w / 1000,
            "longitudinal_g": ax / GRAVITY,
            "lateral_g": ay / GRAVITY,
        }

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame

        def transform(point):
            vector = pygame.math.Vector2(point).rotate_rad(angle)
            return (vector.x * zoom + translation[0], vector.y * zoom + translation[1])

        if draw_particles:
            for particle in self.particles:
                if len(particle.poly) > 1:
                    pygame.draw.lines(
                        surface,
                        particle.color,
                        False,
                        [transform(p) for p in particle.poly],
                        max(1, int(zoom * 0.07)),
                    )
        for body in self.drawlist:
            color = tuple(int(c * 255) for c in body.color)
            for fixture in body.fixtures:
                points = [transform(body.transform * v) for v in fixture.shape.vertices]
                pygame.draw.polygon(surface, color, points)
            if body is not self.hull:
                offset = math.sin(body.phase) * body.wheel_rad * 0.65
                points = [
                    transform(body.GetWorldPoint((x, offset)))
                    for x in (-body.tyre_width * 0.45, body.tyre_width * 0.45)
                ]
                pygame.draw.line(
                    surface, (95, 95, 100), points[0], points[1], max(1, int(zoom * 0.06))
                )
