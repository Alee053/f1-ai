"""Vídeo de una política cargada bajo su propio contrato de control."""

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np
import pygame
from gymnasium.envs.box2d.car_racing import FPS

from f1_ai.agents import PPOAgent
from f1_ai.artifacts import write_json
from f1_ai.camera import ChaseCamera, validate_view
from f1_ai.config import RACING_RESIDUAL, ExperimentConfig
from f1_ai.environment import NumericCarRacing
from f1_ai.residual_env import ResidualRacingEnv
from f1_ai.track_layout import TrackLayout
from f1_ai.training import load_model, read_model_metadata


class ReplayPhysics(NumericCarRacing):
    on_frame_step = None

    def step(self, action):
        result = super().step(action)
        if action is not None and self.on_frame_step is not None:
            _, _, terminated, truncated, info = result
            self.on_frame_step(action, info, terminated or truncated)
        return result


def _chase_frame(view, surface, env, info, action, method, fonts):
    width, height = surface.get_size()
    footer_height = 52
    view.draw(surface.subsurface((0, 0, width, height - footer_height)), env)
    pygame.draw.rect(surface, (14, 19, 27), (0, height - footer_height, width, footer_height))
    if env.config.formula is not None:
        speed = f"{info['speed'] * 3.6:.0f} km/h"
    else:
        speed = f"{info['speed']:.1f} unidades/s"
    title, details = fonts
    surface.blit(title.render(method, True, (229, 240, 249)), (12, height - 48))
    label = title.render(speed, True, (149, 213, 251))
    surface.blit(label, (width - 12 - label.get_width(), height - 48))
    pedals = f"Gas {max(0.0, float(action[1])):.0%}   Freno {max(0.0, -float(action[1])):.0%}"
    slip = max((abs(value) for value in info.get("slip_ratios", ())), default=0.0)
    status = f"{info['simulated_seconds']:.2f} s    {pedals}    Deslizamiento {slip:.2f}"
    surface.blit(details.render(status, True, (192, 212, 228)), (12, height - 23))
    if info["lap_complete"]:
        text = "Vuelta limpia" if not info["off_track_steps"] else "Vuelta completa"
        label = details.render(text, True, (122, 228, 176))
        surface.blit(label, (width - 12 - label.get_width(), height - 23))
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)


def record_replay(
    config: ExperimentConfig,
    model_path: Path,
    output: Path,
    *,
    seed: int = 100,
    stride: int | None = None,
    camera: str = "top",
    grid_m: float = 1.0,
    device: str | None = None,
    track: Path | None = None,
) -> dict:
    validate_view(camera, grid_m)
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise FileNotFoundError("La exportación de vídeo requiere ffmpeg.")
    residual = config.environment.control_profile == RACING_RESIDUAL
    if stride is None:
        stride = 2
    if output.suffix.lower() != ".mp4" or type(stride) is not int or stride <= 0 or FPS % stride:
        raise ValueError(f"Use una salida MP4 y un salto de cuadros que divida {FPS}.")
    metadata_path = output.with_suffix(".json")
    snapshot_path = output.with_suffix(".png")
    telemetry_path = output.with_suffix(".csv") if track is not None else None
    paths = (output, metadata_path, snapshot_path, telemetry_path)
    if any(path.exists() for path in paths if path is not None):
        raise FileExistsError("Ya existe un archivo de esta exportación.")
    layout = TrackLayout.load(track) if track is not None else None
    environment = config.environment
    if layout is not None:
        # Dimensiona el horizonte a una vuelta larga sin cambiar el contrato del modelo.
        max_steps = max(environment.max_steps, ceil(layout.geometry.length / 12.0 * FPS))
        environment = replace(environment, max_steps=max_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    model = load_model(model_path, config, device=device)
    agent = PPOAgent(model)
    physics = ReplayPhysics(environment, render_mode="rgb_array", track_layout=layout)
    env = ResidualRacingEnv(physics) if residual else physics
    view = ChaseCamera(grid_m) if camera == "chase" else None
    surface = pygame.Surface((960, 540)) if view is not None else None
    source = read_model_metadata(model_path)
    root_kind = source.get("initial_policy", {}).get("root_kind")
    method = "imitation+ppo" if root_kind in {"behavior_cloning", "dagger"} else source["kind"]
    if residual:
        method = "residual-ppo"
    label = "PPO directo" if source["kind"] == "ppo" else "Política neuronal"
    if residual:
        label = "PPO residual"
    if config.environment.formula is not None:
        label = f"Formula   /   {label}"
    track_name = layout.name if layout is not None else f"Pista {seed}"
    label = f"{label}   /   {track_name}"
    process = None
    frames = 0
    max_speed = max_slip = 0.0
    lock_steps = 0
    snapshot = None
    telemetry = []
    try:
        fps = FPS // stride
        observation, info = env.reset(seed=seed)
        fonts = None
        if view is not None:
            pygame.font.init()
            fonts = (pygame.font.Font(None, 27), pygame.font.Font(None, 22))

        def frame_for(state, action):
            if view is not None:
                return _chase_frame(view, surface, physics, state, action, label, fonts)
            return env.render()

        frame = frame_for(info, np.zeros(2))
        height, width, channels = frame.shape
        if channels != 3:
            raise ValueError("El renderizado no produjo cuadros RGB.")
        with (
            tempfile.TemporaryDirectory(dir=output.parent, prefix=".replay-") as work,
            tempfile.TemporaryFile() as errors,
        ):
            staging = Path(work) / "video.mp4"
            process = subprocess.Popen(
                [
                    encoder,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(staging),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            assert process.stdin is not None

            def emit(rgb):
                nonlocal frames
                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
                frames += 1

            def capture(action, state, finished):
                nonlocal frame, snapshot, max_speed, max_slip, lock_steps
                max_speed = max(max_speed, state["speed"])
                slip = max((abs(value) for value in state.get("slip_ratios", ())), default=0.0)
                max_slip = max(max_slip, slip)
                lock_steps += int(any(state.get("locked_wheels", ())))
                if telemetry_path is not None:
                    telemetry.append(
                        (
                            physics.elapsed_steps,
                            state["simulated_seconds"],
                            *state["position"],
                            state["speed"],
                            *map(float, action),
                            state["forward_progress"],
                            physics.road_contact_fraction,
                        )
                    )
                if physics.elapsed_steps % stride == 0 or finished:
                    frame = frame_for(state, action)
                    emit(frame)
                    if snapshot is None and physics.elapsed_steps >= 300:
                        snapshot = frame.copy()

            emit(frame)
            physics.on_frame_step = capture
            terminated = truncated = False
            while not (terminated or truncated):
                observation, _, terminated, truncated, info = env.step(agent.action(observation))
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                errors.seek(0)
                raise RuntimeError(
                    f"ffmpeg no pudo exportar el vídeo: {errors.read().decode(errors='replace')}"
                )
            os.link(staging, output)
        image = frame if snapshot is None else snapshot
        pygame.image.save(pygame.surfarray.make_surface(image.swapaxes(0, 1)), snapshot_path)
        if telemetry_path is not None:
            with telemetry_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "step",
                        "simulated_seconds",
                        "x",
                        "y",
                        "speed",
                        "steering",
                        "acceleration",
                        "forward_progress",
                        "road_contact",
                    )
                )
                writer.writerows(telemetry)
        result = {
            "model": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "inference_device": str(model.device),
            "method": method,
            "seed": seed,
            "frames": frames,
            "fps": fps,
            "camera": camera,
            "grid_m": grid_m if view is not None else None,
            "peak_speed_kmh": max_speed * 3.6 if config.environment.formula is not None else None,
            "max_abs_slip_ratio": max_slip,
            "wheel_lock_steps": lock_steps,
            "simulated_seconds": info["simulated_seconds"],
            "coverage": info["coverage"],
            "lap_complete": info["lap_complete"],
            "termination_reason": info["termination_reason"],
            "off_track_steps": info["off_track_steps"],
            "neural_policy_only": not residual,
            "hud_reward": "native CarRacing score" if view is None else None,
        }
        if layout is not None:
            result.update(
                track_name=layout.name,
                track_length_m=layout.geometry.length,
                track_width_m=layout.width_m,
                forward_progress=info["forward_progress"],
                max_seconds=environment.max_steps / FPS,
            )
        write_json(metadata_path, result)
        return result
    finally:
        env.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
