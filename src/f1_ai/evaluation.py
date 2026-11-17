"""Evaluación por circuito; los fallos nunca se convierten en tiempos de vuelta."""

import csv
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from f1_ai.agents import Agent
from f1_ai.artifacts import write_json
from f1_ai.config import RACING_RESIDUAL, EnvironmentConfig
from f1_ai.environment import make_env


def evaluate_episode(
    config: EnvironmentConfig,
    agent: Agent,
    circuit_seed: int,
    *,
    agent_name: str,
    output: Path | None = None,
) -> dict:
    env = make_env(config)
    physics = env.unwrapped
    residual = config.control_profile == RACING_RESIDUAL
    trajectory = None
    started = time.perf_counter()
    reward_sum = 0.0
    native_sum = 0.0
    speed_sum = 0.0
    max_slip = steering_variation = last_steering = 0.0
    lock_steps = 0
    try:
        observation, info = env.reset(seed=circuit_seed)
        if output is not None:
            track_path = output / f"track-{circuit_seed}.json"
            if not track_path.exists():
                write_json(track_path, physics.geometry.points.tolist())
            if residual:
                plan_path = output / f"reference-plan-{circuit_seed}.json"
                if not plan_path.exists():
                    write_json(plan_path, env.controller.plan.to_dict())
            trajectory = (output / f"trajectory-{agent_name}-{circuit_seed}.csv").open(
                "x",
                newline="",
                encoding="utf-8",
            )
            writer = csv.writer(trajectory)
            writer.writerow(
                [
                    "step",
                    "simulated_seconds",
                    "x",
                    "y",
                    "speed",
                    "steering",
                    "acceleration",
                    "reward",
                    "coverage",
                    "forward_progress",
                    "road_contact",
                    *(["offset_command", "speed_adjustment_command"] if residual else []),
                ]
            )
        done = False
        while not done:
            action = agent.action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            native_sum += info["native_reward"]
            speed_sum += info.get("speed_sum", info["speed"])
            applied = info.get("applied_action", action)
            if residual:
                max_slip = max(max_slip, info["max_abs_slip_ratio"])
                lock_steps += info["wheel_lock_steps"]
                steering_variation += info["steering_variation"]
            else:
                max_slip = max(max_slip, *(abs(value) for value in info.get("slip_ratios", (0.0,))))
                lock_steps += int(any(info.get("locked_wheels", ())))
                steering_variation += abs(float(action[0]) - last_steering)
                last_steering = float(action[0])
            if trajectory is not None:
                writer.writerow(
                    [
                        physics.elapsed_steps,
                        info["simulated_seconds"],
                        *info["position"],
                        info["speed"],
                        *applied,
                        reward,
                        info["coverage"],
                        info["forward_progress"],
                        physics.road_contact_fraction,
                        *(action if residual else ()),
                    ]
                )
            done = terminated or truncated
        return {
            "agent": agent_name,
            "circuit_seed": circuit_seed,
            "return": reward_sum,
            "native_return": native_sum,
            "lap_complete": info["lap_complete"],
            "lap_time_seconds": info["simulated_seconds"] if info["lap_complete"] else None,
            "coverage": info["coverage"],
            "forward_progress": info["forward_progress"],
            "steps": physics.elapsed_steps,
            "simulated_seconds": info["simulated_seconds"],
            "termination_reason": info["termination_reason"],
            "terminated": terminated,
            "truncated": truncated,
            "off_track_steps": info["off_track_steps"],
            "off_track_fraction": info["off_track_steps"] / physics.elapsed_steps,
            "off_track_events": info["off_track_events"],
            "mean_speed": speed_sum / physics.elapsed_steps,
            "max_abs_slip_ratio": max_slip,
            "wheel_lock_steps": lock_steps,
            "steering_variation_per_second": steering_variation / info["simulated_seconds"],
            "track_length": info["track_length"],
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        if trajectory is not None:
            trajectory.close()
        env.close()


def summarize(episodes: Sequence[dict]) -> dict:
    if not episodes:
        raise ValueError("Se necesita al menos un episodio para resumir resultados.")
    returns = np.array([episode["return"] for episode in episodes])
    times = [episode["lap_time_seconds"] for episode in episodes if episode["lap_complete"]]
    clean_times = [
        episode["lap_time_seconds"]
        for episode in episodes
        if episode["lap_complete"] and episode["off_track_fraction"] == 0
    ]
    return {
        "episodes": len(episodes),
        "completed_laps": len(times),
        "clean_completed_laps": len(clean_times),
        "completion_rate": len(times) / len(episodes),
        "mean_return": float(returns.mean()),
        "sample_std_return": float(returns.std(ddof=1)) if len(returns) > 1 else None,
        "mean_coverage": float(np.mean([episode["coverage"] for episode in episodes])),
        "mean_lap_time_seconds": float(np.mean(times)) if times else None,
        "mean_clean_lap_time_seconds": float(np.mean(clean_times)) if clean_times else None,
        "mean_off_track_fraction": float(
            np.mean([episode["off_track_fraction"] for episode in episodes])
        ),
    }


def driving_score(summary: dict, *, prioritize_clean_time: bool) -> tuple:
    if prioritize_clean_time:
        return (
            summary["clean_completed_laps"],
            summary["completed_laps"],
            -(summary["mean_clean_lap_time_seconds"] or float("inf")),
            summary["mean_coverage"],
        )
    return (
        summary["completed_laps"],
        summary["mean_coverage"],
        -(summary["mean_lap_time_seconds"] or float("inf")),
    )


def evaluate_agents(
    config: EnvironmentConfig,
    factories: dict[str, Callable[[int], Agent]],
    seeds: tuple[int, ...],
    output: Path,
) -> dict:
    episodes = []
    summaries = {}
    for name, factory in factories.items():
        agent_episodes = []
        for seed in seeds:
            episode = evaluate_episode(config, factory(seed), seed, agent_name=name, output=output)
            agent_episodes.append(episode)
            print(
                f"{name}: circuito={seed} retorno={episode['return']:.1f} "
                f"cobertura={episode['coverage']:.1%} fin={episode['termination_reason']}",
                flush=True,
            )
        summaries[name] = summarize(agent_episodes)
        episodes.extend(agent_episodes)
    with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    results = {"summaries": summaries, "episodes": episodes}
    write_json(output / "results.json", results)
    return results
