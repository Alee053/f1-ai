"""Planificación offline y ensayos del experto global, separados de PPO."""

import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from gymnasium.envs.box2d.car_racing import FPS, TRACK_WIDTH

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import ExperimentConfig
from f1_ai.environment import make_env
from f1_ai.racing_controller import FollowerConfig, RacingController, ReferenceState
from f1_ai.racing_line import ReferenceConfig, plan_reference


def _metadata(config, kind):
    if config.environment.vehicle_model != "formula-v1":
        raise ValueError("La referencia global requiere el modelo formula-v1.")
    return {
        **provenance(config, kind=kind),
        "observation_version": None,
        "observation_names": None,
        "information_regime": "full_map_and_physical_state",
        "neural_policy": False,
        "training_performed": False,
    }


def plan_saved_track(
    config: ExperimentConfig,
    track: Path,
    output: Path,
    settings: ReferenceConfig | None = None,
    *,
    half_width_m: float = TRACK_WIDTH,
) -> dict:
    """Lee un mapa local; no crea un entorno ni ejecuta pasos físicos."""
    if settings is None:
        settings = ReferenceConfig()
    metadata = _metadata(config, "offline_racing_plan")
    source = track.read_bytes()
    points = np.asarray(json.loads(source), dtype=np.float64)
    plan = plan_reference(points, half_width_m, config.environment.formula, settings)
    output = create_directory(output)
    write_json(
        output / "metadata.json",
        {
            **metadata,
            "source_track": str(track),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "physical_steps": 0,
        },
    )
    write_json(output / "reference-plan.json", plan.to_dict())
    write_json(output / "track.json", points.tolist())
    return {
        "selected_candidate": plan.candidate,
        "estimated_flying_lap_seconds": plan.estimated_lap_seconds,
        "centerline_estimated_flying_lap_seconds": plan.centerline_estimated_lap_seconds,
        "box2d_driving_validated": False,
        "output": str(output),
    }


def run_reference(
    config: ExperimentConfig,
    output: Path,
    *,
    seed: int = 0,
    steps: int | None = None,
    settings: ReferenceConfig | None = None,
    follower: FollowerConfig | None = None,
) -> dict:
    """Ensayo instrumental en train. El experto nunca usa la recompensa."""
    if settings is None:
        settings = ReferenceConfig()
    if follower is None:
        follower = FollowerConfig()
    metadata = _metadata(config, "global_reference_driving")
    if type(seed) is not int or seed not in config.circuits.train:
        raise ValueError("El desarrollo de esta referencia usa solo semillas de entrenamiento.")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("steps debe ser un entero positivo.")
    environment = config.environment
    if steps is not None:
        environment = replace(environment, max_steps=min(steps, environment.max_steps))
    output = create_directory(output)
    env = make_env(environment)
    started = time.perf_counter()
    reward_sum = 0.0
    max_rear_slip = 0.0
    lock_steps = 0
    try:
        _, info = env.reset(seed=seed)
        plan = plan_reference(env.geometry.points, TRACK_WIDTH, environment.formula, settings)
        controller = RacingController(plan, follower)
        write_json(output / "track.json", env.geometry.points.tolist())
        write_json(output / "reference-plan.json", plan.to_dict())
        write_json(
            output / "metadata.json",
            {
                **metadata,
                "circuit_seed": seed,
                "split": "train",
                "effective_environment": asdict(environment),
                "follower": asdict(follower),
                "dt_seconds": 1 / FPS,
                "reward_used_for_control": False,
            },
        )
        with (output / "trajectory.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "step",
                    "simulated_seconds",
                    "x",
                    "y",
                    "speed_m_s",
                    "steering",
                    "pedal",
                    "max_rear_slip",
                    "wheel_lock",
                    "road_contact",
                    "coverage",
                ]
            )
            terminated = truncated = False
            while not (terminated or truncated):
                car = env.car
                state = ReferenceState(
                    position=tuple(car.hull.position),
                    heading_rad=float(car.hull.angle),
                    velocity=tuple(car.hull.linearVelocity),
                    yaw_rate_rad_s=float(car.hull.angularVelocity),
                    slip_ratios=car.telemetry["slip_ratios"],
                    locked_wheels=car.telemetry["locked_wheels"],
                )
                action = controller.action(state, 1 / FPS)
                _, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                rear_slip = max(abs(value) for value in car.telemetry["slip_ratios"][2:])
                locked = any(car.telemetry["locked_wheels"])
                max_rear_slip = max(max_rear_slip, rear_slip)
                lock_steps += int(locked)
                writer.writerow(
                    [
                        env.elapsed_steps,
                        info["simulated_seconds"],
                        *info["position"],
                        info["speed"],
                        *action,
                        rear_slip,
                        int(locked),
                        env.road_contact_fraction,
                        info["coverage"],
                    ]
                )
        result = {
            "agent": "global_racing_reference",
            "circuit_seed": seed,
            "lap_complete": info["lap_complete"],
            "lap_time_seconds": info["simulated_seconds"] if info["lap_complete"] else None,
            "coverage": info["coverage"],
            "steps": env.elapsed_steps,
            "simulated_seconds": info["simulated_seconds"],
            "termination_reason": info["termination_reason"],
            "off_track_steps": info["off_track_steps"],
            "max_abs_rear_slip_ratio": max_rear_slip,
            "wheel_lock_steps": lock_steps,
            "return": reward_sum,
            "wall_seconds": time.perf_counter() - started,
            "neural_policy": False,
            "optimality_proven": False,
            "output": str(output),
        }
        write_json(output / "results.json", result)
        return result
    finally:
        env.close()
