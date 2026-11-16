"""Figuras exportables a partir de las mediciones, sin datos inventados."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.box2d.car_racing import FPS
from matplotlib.ticker import FuncFormatter, MaxNLocator

COLORS = {
    "random": "#9b4a40",
    "pursuit": "#236b61",
    "ppo": "#304f8d",
    "bc": "#8c579a",
    "bc-ppo": "#c06b23",
}
LABELS = {
    "random": "Aleatorio",
    "pursuit": "Pure pursuit",
    "ppo": "PPO",
    "bc": "Imitación",
    "bc-ppo": "Imitación + PPO",
}


def plot_evaluation(directory: Path) -> list[Path]:
    with (directory / "results.json").open(encoding="utf-8") as stream:
        results = json.load(stream)
    names = list(results["summaries"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    for index, name in enumerate(names):
        episodes = [episode for episode in results["episodes"] if episode["agent"] == name]
        coverage = [episode["coverage"] * 100 for episode in episodes]
        axes[0].scatter([index] * len(coverage), coverage, color=COLORS[name], s=32)
        axes[0].plot(
            [index - 0.15, index + 0.15], [np.mean(coverage)] * 2, color=COLORS[name], linewidth=2
        )
        axes[1].bar(index, 100 * results["summaries"][name]["completion_rate"], color=COLORS[name])
    for axis, title in zip(axes, ["Cobertura por circuito", "Vueltas completadas"], strict=True):
        axis.set_xticks(range(len(names)), [LABELS[name] for name in names])
        axis.set_ylim(-2, 105)
        axis.set_ylabel("Porcentaje")
        axis.set_title(title, fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(directory / "comparison.png", dpi=200)
    plt.close(fig)

    first_seed = results["episodes"][0]["circuit_seed"]
    with (directory / f"track-{first_seed}.json").open() as stream:
        points = np.array(json.load(stream))
    points = np.vstack((points, points[0]))
    fig, axis = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    axis.plot(points[:, 0], points[:, 1], color="#c6c6c6", linewidth=9, label="Circuito")
    axis.plot(points[:, 0], points[:, 1], color="white", linewidth=0.6, linestyle="--")
    for name in names:
        with (directory / f"trajectory-{name}-{first_seed}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        axis.plot(
            [float(row["x"]) for row in rows],
            [float(row["y"]) for row in rows],
            color=COLORS[name],
            linewidth=1.3,
            label=LABELS[name],
        )
    axis.scatter(*points[0], marker="s", s=30, color="black", label="Inicio")
    axis.set_aspect("equal")
    axis.set_xlabel("Posición x en unidades del simulador")
    axis.set_ylabel("Posición y en unidades del simulador")
    axis.set_title(f"Trayectorias medidas en el circuito {first_seed}", fontsize=11)
    axis.legend(fontsize=8, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "trajectories.png", dpi=200)
    plt.close(fig)
    return [directory / "comparison.png", directory / "trajectories.png"]


def plot_training(directory: Path) -> Path:
    with np.load(directory / "validation" / "evaluations.npz", allow_pickle=False) as data:
        timesteps = data["timesteps"]
        returns = data["results"]
        driving_keys = {"successes", "coverage", "ep_lengths"}
        driving = (
            {key: data[key] for key in driving_keys} if driving_keys <= set(data.files) else None
        )
    means = returns.mean(axis=1)
    deviations = returns.std(axis=1, ddof=int(returns.shape[1] > 1))
    if driving is None:
        # Los experimentos iniciales de SB3 solo registraban retorno y duración.
        fig, axis = plt.subplots(figsize=(8, 3.2), layout="constrained")
        axes = np.array([axis])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), layout="constrained")
        axes = axes.ravel()
        axis = axes[0]
        successes = driving["successes"]
        counts = successes.sum(axis=1)
        lap_times = np.divide(
            np.where(successes, driving["ep_lengths"] / FPS, 0).sum(axis=1),
            counts,
            out=np.full(len(counts), np.nan),
            where=counts > 0,
        )
        axes[1].plot(timesteps, 100 * successes.mean(axis=1), color=COLORS["bc-ppo"], marker=".")
        axes[1].set_ylabel("Vueltas completas (%)")
        axes[1].set_ylim(-3, 103)
        axes[2].plot(timesteps, 100 * driving["coverage"].mean(axis=1), color=COLORS["pursuit"])
        axes[2].set_ylabel("Cobertura media (%)")
        axes[2].set_ylim(-3, 103)
        axes[3].plot(timesteps, lap_times, color=COLORS["bc"], marker=".")
        axes[3].set_ylabel("Tiempo medio en éxitos (s)")
        axes[3].set_title("Sin punto cuando no hay vueltas completas", fontsize=9)
    axis.plot(timesteps, means, color=COLORS["ppo"], label="Retorno medio en validación")
    axis.fill_between(
        timesteps,
        means - deviations,
        means + deviations,
        color=COLORS["ppo"],
        alpha=0.15,
        label="Desviación estándar entre circuitos",
    )
    axis.set_ylabel("Retorno acumulado")
    fig.suptitle(f"PPO en {returns.shape[1]} circuitos de validación", fontsize=11)
    axis.legend(fontsize=7)
    for item in axes:
        item.set_xlabel("Interacciones PPO (miles)")
        item.xaxis.set_major_locator(MaxNLocator(nbins=4))
        item.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
        item.spines[["top", "right"]].set_visible(False)
        item.grid(alpha=0.15)
    path = directory / "learning_curve.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
