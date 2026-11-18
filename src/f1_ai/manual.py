"""Práctica manual con Xbox, telemetría y ajustes físicos reproducibles."""

import csv
import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pygame

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.camera import ChaseCamera, validate_view
from f1_ai.config import RACING_RESIDUAL, ExperimentConfig
from f1_ai.controls import DrivingInput, XboxInput
from f1_ai.environment import make_env
from f1_ai.vehicle import GRAVITY

# nombre, campo del TOML, incremento, mínimo y máximo del editor.
TUNING = (
    ("Carga aero ClA", "downforce_cla_m2", 0.5, 0, 12),
    ("Potencia pico kW", "power_kw", 25, 100, 1500),
    ("Masa kg", "mass_kg", 10, 300, 2000),
    ("Agarre neumático", "grip_coefficient", 0.05, 0.2, 3),
    ("Resistencia CdA", "drag_cda_m2", 0.1, 0.1, 4),
    ("Reparto freno delantero", "brake_front_fraction", 0.01, 0.3, 0.8),
    ("Densidad aire kg/m3", "air_density_kg_m3", 0.05, 0.5, 1.5),
)
BUTTON_ACTIONS = {
    pygame.CONTROLLER_BUTTON_A: "reset",
    pygame.CONTROLLER_BUTTON_X: "next_track",
    pygame.CONTROLLER_BUTTON_Y: "tuning",
    pygame.CONTROLLER_BUTTON_START: "pause",
    pygame.CONTROLLER_BUTTON_B: "pause",
    pygame.CONTROLLER_BUTTON_BACK: "exit",
    pygame.CONTROLLER_BUTTON_DPAD_UP: "select_up",
    pygame.CONTROLLER_BUTTON_DPAD_DOWN: "select_down",
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: "decrease",
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "increase",
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "aero",
    pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "save",
    pygame.CONTROLLER_BUTTON_RIGHTSTICK: "camera",
}
KEY_ACTIONS = {
    pygame.K_ESCAPE: "exit",
    pygame.K_SPACE: "pause",
    pygame.K_p: "pause",
    pygame.K_r: "reset",
    pygame.K_n: "next_track",
    pygame.K_TAB: "tuning",
    pygame.K_w: "select_up",
    pygame.K_s: "select_down",
    pygame.K_a: "decrease",
    pygame.K_d: "increase",
    pygame.K_F5: "save",
    pygame.K_g: "guide",
    pygame.K_b: "aero",
    pygame.K_c: "camera",
}


def export_config(config: ExperimentConfig, path: Path):
    """Exporta un TOML completo, con escritura exclusiva y sin dependencias nuevas."""
    values = asdict(config)
    sections = [("environment", values["environment"])]
    formula = values["environment"].get("formula")
    if formula is not None:
        sections.append(("environment.formula", formula))
    reference = values["environment"]["racing_reference"]
    if reference is not None:
        sections.extend(
            (f"environment.racing_reference.{name}", value) for name, value in reference.items()
        )
    sections.extend((name, values[name]) for name in ("circuits", "ppo"))
    lines = ["# Ajuste manual Formula v1. Coeficientes aproximados; sin homologación F1."]
    for name, values in sections:
        lines.extend(("", f"[{name}]"))
        for key, value in values.items():
            if value is not None and not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


class ManualSession:
    def __init__(self, config: ExperimentConfig, seed: int, *, camera="top", grid_m=1.0):
        validate_view(camera, grid_m)
        if config.environment.control_profile == RACING_RESIDUAL:
            raise ValueError("La conducción manual requiere mandos directos.")
        if seed not in config.circuits.train:
            raise ValueError(
                "La práctica utiliza circuitos de entrenamiento; elija una semilla de train."
            )
        # Los ajustes exportados conservan los límites del experimento original.
        self.config = config
        self.seed = seed
        self.paused = True
        self.finished = False
        self.tuning = False
        self.selected = 0
        self.guide = False
        self.camera_mode = camera
        self.chase_camera = ChaseCamera(grid_m)
        self.revision = 0
        self.episode = 0
        self.env = None
        self.info = {}
        self.previous_aero = 5.0
        self.message = "START o espacio para conducir"
        self.reset()

    def reset(self):
        if self.env is not None:
            self.env.close()
        # La práctica permite recuperar el coche en césped durante la sesión.
        self.env = make_env(
            replace(self.config.environment, max_steps=180000, off_track_grace_steps=180000)
        )
        _, self.info = self.env.reset(seed=self.seed)
        self.episode += 1
        self.finished = False

    def command(self, action: str):
        if action == "pause":
            if not self.finished:
                self.paused = not self.paused
                self.tuning = False
        elif action == "reset":
            self.reset()
            self.paused = False
            self.tuning = False
        elif action == "next_track":
            seeds = self.config.circuits.train
            self.seed = seeds[(seeds.index(self.seed) + 1) % len(seeds)]
            self.reset()
            self.paused = True
        elif action == "tuning":
            self.tuning = not self.tuning
            self.paused = True
        elif action == "guide":
            self.guide = not self.guide
        elif action == "camera":
            self.camera_mode = "chase" if self.camera_mode == "top" else "top"
        elif action in {"select_up", "select_down"}:
            self.selected = (self.selected + (-1 if action == "select_up" else 1)) % len(TUNING)
        elif action in {"increase", "decrease", "aero"}:
            parameters = self.config.environment.formula
            if parameters is None:
                self.message = "Los ajustes físicos requieren configs/formula.toml"
                return
            if action == "aero":
                current = parameters.downforce_cla_m2
                if current > 0:
                    self.previous_aero = current
                changes = {"downforce_cla_m2": 0.0 if current else self.previous_aero}
            else:
                if not self.tuning:
                    return
                _, name, step, low, high = TUNING[self.selected]
                delta = step if action == "increase" else -step
                changes = {name: round(min(high, max(low, getattr(parameters, name) + delta)), 4)}
            parameters = replace(parameters, **changes)
            self.config = replace(
                self.config, environment=replace(self.config.environment, formula=parameters)
            )
            self.revision += 1
            self.reset()
            self.paused = True
            self.message = "Ajuste aplicado. START o espacio para reanudar"

    def step(self, driving: DrivingInput):
        if self.paused or self.finished:
            return
        _, _, terminated, truncated, self.info = self.env.step(driving.action)
        if terminated or truncated:
            self.finished = True
            self.paused = True
            self.message = self.info["termination_reason"] + ". A o R para reiniciar"

    def close(self):
        if self.env is not None:
            self.env.close()


def draw_track(surface, session: ManualSession):
    env = session.env
    if session.camera_mode == "chase":
        guide = env.geometry.points if session.guide else None
        return session.chase_camera.draw(surface, env, guide=guide)
    surface.fill((39, 70, 57))
    width, height = surface.get_size()
    angle = -env.car.hull.angle
    speed = float(np.linalg.norm(env.car.hull.linearVelocity))
    zoom = max(3.8, 10.5 - 0.06 * speed)
    rotated_position = pygame.math.Vector2(env.car.hull.position).rotate_rad(angle)
    translation = (width / 2 - rotated_position.x * zoom, height * 0.23 - rotated_position.y * zoom)

    def transform(point):
        # Pygame 2.6 no acepta directamente filas ndarray de NumPy 2.5.
        p = pygame.math.Vector2(float(point[0]), float(point[1])).rotate_rad(angle)
        return (p.x * zoom + translation[0], p.y * zoom + translation[1])

    for polygon, color in env.road_poly:
        points = [transform(p) for p in polygon]
        pygame.draw.polygon(surface, tuple(int(c) for c in color), points)
    if session.guide:
        points = [transform(p) for p in env.geometry.points]
        pygame.draw.lines(surface, (150, 170, 183), True, points, 1)
    env.car.draw(surface, zoom, translation, angle, True)
    return pygame.transform.flip(surface, False, True)


def draw_dashboard(window, session: ManualSession, pad: XboxInput, driving: DrivingInput):
    width, height = window.get_size()
    panel_width = 350
    viewport = pygame.Surface((width - panel_width, height))
    window.blit(draw_track(viewport, session), (0, 0))
    x = width - panel_width
    pygame.draw.rect(window, (14, 20, 30), (x, 0, panel_width, height))
    small = pygame.font.Font(None, 22)
    normal = pygame.font.Font(None, 26)
    bold = pygame.font.Font(None, 32)
    big = pygame.font.Font(None, 80)

    def text(value, y, color=(213, 225, 236), font=normal, offset=20):
        window.blit(font.render(str(value), True, color), (x + offset, y))

    text(
        "FORMULA v1" if session.config.environment.formula else "CLASSIC", 20, (91, 219, 209), bold
    )
    text("Modelo 2D experimental", 53, font=small)
    speed = float(np.linalg.norm(session.env.car.hull.linearVelocity))
    if session.config.environment.formula is not None:
        speed *= 3.6
    text(f"{speed:03.0f}", 84, font=big)
    text("km/h" if session.config.environment.formula else "unidades/s", 145, font=small)
    parameters = session.config.environment.formula
    info = session.info
    if parameters is not None:
        df = info.get("downforce_n", 0)
        text(f"{parameters.mass_kg:.0f} kg   {parameters.power_kw:.0f} kW pico", 177)
        text(f"Downforce   {df / 1000:.2f} kN", 216)
        text(f"Carga aero   {df / (parameters.mass_kg * GRAVITY):.2f} x peso", 244)
        text(
            f"Agarre base {parameters.grip_coefficient:.2f} ClA {parameters.downforce_cla_m2:.1f}",
            272,
        )
        rear_slip = max(abs(v) for v in info.get("slip_ratios", (0, 0, 0, 0))[2:])
        locked = any(info.get("locked_wheels", (False,) * 4))
        warning = (
            "RUEDAS BLOQUEADAS"
            if locked
            else "PATINAMIENTO"
            if rear_slip > 0.3
            else "SIN ABS / SIN TC"
        )
        text(warning, 307, (255, 135, 90) if locked or rear_slip > 0.3 else (133, 157, 175), small)
    else:
        text("Física CarRacing original", 190)
        text("Velocidad sin calibración SI", 221, font=small)
    for label, value, y, color in [
        ("RT  Acelerador", max(0, float(driving.action[1])), 345, (91, 219, 160)),
        ("LT  Freno", driving.brake, 391, (249, 112, 103)),
    ]:
        text(label, y, font=small)
        pygame.draw.rect(window, (35, 45, 60), (x + 20, y + 24, 305, 7), border_radius=3)
        pygame.draw.rect(window, color, (x + 20, y + 24, int(305 * value), 7), border_radius=3)
    text(f"Pista {session.seed}   Vuelta {session.info['simulated_seconds']:.2f} s", 446)
    text(f"Cobertura {100 * session.info['coverage']:.1f}%", 475, font=small)
    text(pad.name[:35], 514, (133, 157, 175), small)
    for value, y in [
        ("Stick izq: dirección   RT/LT: pedales", 544),
        ("A/R reiniciar   X/N siguiente pista", 570),
        ("START/espacio pausa   Y/TAB ajustes", 596),
        ("LB: aero A/B   RB/F5: guardar", 622),
        ("BACK/ESC salir   Flechas conducir", 648),
        ("G mostrar eje de referencia", 674),
        ("C / pulsar stick der: cambiar vista", 700),
    ]:
        text(value, y, font=small)
    if session.paused:
        shade = pygame.Surface((width - panel_width, height), pygame.SRCALPHA)
        shade.fill((4, 10, 17, 165))
        window.blit(shade, (0, 0))
        left = 32
        window.blit(
            bold.render("AJUSTES" if session.tuning else "PAUSA", True, (235, 241, 247)), (left, 32)
        )
        if session.tuning and parameters is not None:
            for index, (name, field, _, _, _) in enumerate(TUNING):
                selected = session.selected == index
                color = (91, 219, 209) if selected else (208, 220, 230)
                label = f"{'>' if selected else ' '} {name}: {getattr(parameters, field):.3g}"
                window.blit(normal.render(label, True, color), (left, 95 + index * 40))
            message = "Cruceta o W/S/A/D: ajustar. Cada cambio reinicia la pista."
            window.blit(small.render(message, True, (210, 220, 230)), (left, 410))
        window.blit(small.render(session.message, True, (210, 220, 230)), (left, height - 87))
        window.blit(
            small.render("START o espacio: reanudar. A o R: reiniciar.", True, (210, 220, 230)),
            (left, height - 55),
        )
    return window


def run_manual(
    config: ExperimentConfig,
    *,
    seed: int,
    output: Path | None = None,
    controller_index: int | None = None,
    deadzone: float = 0.08,
    exponent: float = 1.6,
    max_frames: int | None = None,
    headless: bool = False,
    camera: str = "top",
    grid_m: float = 1.0,
) -> dict:
    validate_view(camera, grid_m)
    if max_frames is not None and max_frames < 1:
        raise ValueError("El límite de cuadros debe ser positivo.")
    if headless:
        if max_frames is None:
            raise ValueError("El modo headless requiere --frames.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        output = Path("runs") / f"manual-{stamp}"
    output = create_directory(output)
    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode((1280, 800))
    pygame.display.set_caption("Formula v1 - Xbox / teclado")
    pad = session = None
    frames = 0
    saves = 0
    try:
        pad = XboxInput(controller_index, deadzone=deadzone, exponent=exponent)
        session = ManualSession(config, seed, camera=camera, grid_m=grid_m)
        write_json(
            output / "metadata.json",
            {
                **provenance(session.config, kind="manual"),
                "controller": pad.name,
                "practice_environment": asdict(session.env.config),
                "deadzone": deadzone,
                "stick_exponent": exponent,
                "initial_camera": camera,
                "grid_m": grid_m,
            },
        )
        clock = pygame.time.Clock()
        accumulator = 0.0
        running = True
        with (output / "telemetry.csv").open("x", newline="", buffering=1) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "episode",
                    "seed",
                    "config_revision",
                    "step",
                    "seconds",
                    "speed_display",
                    "steering",
                    "throttle",
                    "brake",
                    "downforce_n",
                    "rear_slip",
                    "locked_wheels",
                    "coverage",
                ]
            )
            while running and (max_frames is None or frames < max_frames):
                elapsed = min(clock.tick(60) / 1000, 0.1)
                if pad.refresh():
                    session.paused = True
                    session.message = pad.status
                for event in pygame.event.get():
                    command = None
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        session.paused = True
                        session.message = "Ventana sin foco; reanudar explícitamente"
                    elif event.type == pygame.CONTROLLERDEVICEREMOVED:
                        if event.instance_id == pad.instance_id:
                            session.paused = True
                            session.message = "Mando desconectado; START o espacio para reanudar"
                    elif event.type == pygame.KEYDOWN:
                        command = KEY_ACTIONS.get(event.key)
                    elif (
                        event.type == pygame.CONTROLLERBUTTONDOWN
                        and event.instance_id == pad.instance_id
                    ):
                        command = BUTTON_ACTIONS.get(event.button)
                    if command == "exit":
                        running = False
                    elif command == "save":
                        saves += 1
                        export_config(session.config, output / f"saved-{saves:03d}.toml")
                        session.message = f"Guardado en {output}/saved-{saves:03d}.toml"
                    elif command is not None:
                        before = session.revision
                        session.command(command)
                        if session.revision != before:
                            export_config(
                                session.config, output / f"revision-{session.revision:03d}.toml"
                            )
                before_read = pad.instance_id
                driving = pad.read()
                if before_read is not None and pad.device is None:
                    session.paused = True
                    session.message = pad.status
                keys = pygame.key.get_pressed()
                if any(
                    keys[k] for k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
                ):
                    driving = DrivingInput(
                        float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]),
                        float(keys[pygame.K_UP]),
                        float(keys[pygame.K_DOWN]),
                    )
                if not headless and not pygame.key.get_focused():
                    session.paused = True
                    session.message = "Ventana sin foco; reanudar explícitamente"
                if session.paused:
                    accumulator = 0.0
                else:
                    accumulator += elapsed
                    while accumulator >= 0.02 and not session.paused:
                        session.step(driving)
                        accumulator -= 0.02
                        info = session.info
                        if session.env.elapsed_steps % 5 == 0:
                            slip = info.get("slip_ratios", (0, 0, 0, 0))
                            writer.writerow(
                                [
                                    session.episode,
                                    session.seed,
                                    session.revision,
                                    session.env.elapsed_steps,
                                    info["simulated_seconds"],
                                    info["speed"]
                                    * (
                                        3.6 if session.config.environment.formula is not None else 1
                                    ),
                                    driving.steering,
                                    max(0, float(driving.action[1])),
                                    driving.brake,
                                    info.get("downforce_n", 0),
                                    max(abs(slip[2]), abs(slip[3])),
                                    sum(info.get("locked_wheels", (False,) * 4)),
                                    info["coverage"],
                                ]
                            )
                draw_dashboard(window, session, pad, driving)
                pygame.display.flip()
                frames += 1
        pygame.image.save(window, output / "last-frame.png")
        export_config(session.config, output / "last-config.toml")
        result = {
            "frames": frames,
            "output": str(output),
            "controller": pad.name,
            "vehicle_model": session.config.environment.vehicle_model,
            "speed_units": "km/h"
            if session.config.environment.formula is not None
            else "simulator_units/s",
            "config_revisions": session.revision,
            "episodes": session.episode,
            "last_seed": session.seed,
            "simulated_seconds": session.info["simulated_seconds"],
        }
        write_json(output / "results.json", result)
        return result
    finally:
        if session is not None:
            session.close()
        if pad is not None:
            pad.close()
        pygame.quit()
