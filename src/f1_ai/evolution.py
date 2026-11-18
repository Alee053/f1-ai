"""NEAT sobre las mismas observaciones, recompensa y circuitos de PPO."""

import json
import random
import time
from dataclasses import asdict
from importlib.metadata import version
from math import isfinite
from pathlib import Path

import numpy as np

from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import ROAD_OBSERVATION, ExperimentConfig
from f1_ai.environment import ROAD_OBSERVATION_NAMES
from f1_ai.evaluation import driving_score, evaluate_episode, summarize
from f1_ai.training import make_training_environment

try:
    import neat
    from neat.activations import tanh_activation
    from neat.aggregations import sum_aggregation
except ModuleNotFoundError as error:
    if error.name != "neat":
        raise
    raise ValueError("NEAT requiere instalar la opción neat con uv sync --extra neat.") from error


class NEATAgent:
    def __init__(self, network: neat.nn.FeedForwardNetwork):
        self.network = network

    def action(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.network.activate(observation.tolist()), dtype=np.float32)


def _policy_data(agent: NEATAgent, config: ExperimentConfig) -> dict:
    return {
        "kind": "neat_feedforward_v1",
        "environment": asdict(config.environment),
        "observation_names": ROAD_OBSERVATION_NAMES,
        "nodes": [
            (node, bias, response, links)
            for node, _, _, bias, response, links in agent.network.node_evals
        ],
    }


def load_neat_agent(path: Path, config: ExperimentConfig) -> NEATAgent:
    """Restaura una red feedforward desde datos JSON, sin deserializar código."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if (
        not isinstance(data, dict)
        or data.get("kind") != "neat_feedforward_v1"
        or data.get("observation_names") != list(ROAD_OBSERVATION_NAMES)
        or not isinstance(data.get("nodes"), list)
    ):
        raise ValueError("El archivo no contiene una política NEAT road-v5.")
    if config.environment.observation_profile != ROAD_OBSERVATION or data.get(
        "environment"
    ) != asdict(config.environment):
        raise ValueError("La configuración de evaluación no coincide con la política NEAT.")
    inputs = list(range(-1, -1 - len(ROAD_OBSERVATION_NAMES), -1))
    available = set(inputs)
    node_evals = []
    try:
        for node, bias, response, links in data["nodes"]:
            if type(node) is not int or node < 0 or node in available:
                raise ValueError("La política NEAT contiene un nodo duplicado o inválido.")
            if not isfinite(bias) or not isfinite(response) or not isinstance(links, list):
                raise ValueError("La política NEAT contiene parámetros inválidos.")
            for source, weight in links:
                if type(source) is not int or source not in available or not isfinite(weight):
                    raise ValueError("La política NEAT contiene una conexión inválida.")
            node_evals.append((node, tanh_activation, sum_aggregation, bias, response, links))
            available.add(node)
    except (TypeError, OverflowError) as error:
        raise ValueError("La estructura de la política NEAT no es válida.") from error
    return NEATAgent(neat.nn.FeedForwardNetwork(inputs, [0, 1], node_evals))


def _discounted_return(env, agent, seed: int, steps: int, gamma: float) -> float:
    obs, _ = env.reset(seed=seed)
    discount, total = 1.0, 0.0
    for index in range(steps):
        obs, reward, terminated, truncated, _ = env.step(agent.action(obs))
        total += discount * reward
        discount *= gamma
        if index + 1 < steps and (terminated or truncated):
            obs, _ = env.reset()
            discount = 1.0
    return total


def train_neat(
    config: ExperimentConfig, output: Path, neat_config: Path, *, rollout_steps: int
) -> dict:
    settings = config.ppo
    if (
        config.environment.observation_profile != ROAD_OBSERVATION
        or config.environment.control_profile != "direct"
        or config.environment.reward_profile != "lap-time-v1"
        or settings.corner_curriculum
        or settings.time_progress_mix
    ):
        raise ValueError(
            "NEAT requiere road-v5, mandos directos y recompensa lap-time-v1 sin guía."
        )
    evolution = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config),
    )
    genome_config = evolution.genome_config
    if (
        not genome_config.feed_forward
        or genome_config.num_inputs != len(ROAD_OBSERVATION_NAMES)
        or genome_config.num_outputs != 2
        or genome_config.activation_options != ["tanh"]
        or genome_config.activation_default != "tanh"
        or genome_config.aggregation_options != ["sum"]
        or genome_config.aggregation_default != "sum"
        or not evolution.no_fitness_termination
    ):
        raise ValueError(
            "La evolución requiere redes feedforward tanh/sum de 36 entradas y 2 salidas."
        )
    if type(rollout_steps) is not int or rollout_steps <= 0 or evolution.pop_size < 2:
        raise ValueError("Se requieren pasos por genoma positivos y al menos dos genomas.")
    steps_per_generation = evolution.pop_size * rollout_steps
    if settings.total_timesteps % steps_per_generation:
        raise ValueError(f"--timesteps debe ser múltiplo de {steps_per_generation} para NEAT.")

    output = create_directory(output)
    evolution.save(str(output / "neat-config.ini"))
    metadata = provenance(config, kind="neat")
    metadata["versions"]["neat-python"] = version("neat-python")
    metadata["execution"] = {
        "device": "cpu",
        "population_size": evolution.pop_size,
        "steps_per_genome": rollout_steps,
        "fitness": "discounted_training_reward",
    }
    write_json(output / "metadata.json", metadata)
    rng = np.random.default_rng(settings.seed)
    env = make_training_environment(config)
    history = []
    best_score = None

    def evaluate_population(genomes, neat_settings):
        nonlocal best_score
        if len(genomes) != evolution.pop_size:
            raise RuntimeError("El tamaño de población cambió durante la evolución.")
        # Cada genoma de esta generación empieza con la misma secuencia de circuitos.
        rollout_seed = int(rng.integers(2**32))
        champion, best_fitness = None, -float("inf")
        for _, genome in genomes:
            agent = NEATAgent(neat.nn.FeedForwardNetwork.create(genome, neat_settings))
            genome.fitness = _discounted_return(
                env, agent, rollout_seed, rollout_steps, settings.gamma
            )
            if genome.fitness > best_fitness:
                champion, best_fitness = agent, genome.fitness

        generation = len(history) + 1
        episodes = [
            evaluate_episode(config.environment, champion, seed, agent_name="neat")
            for seed in config.circuits.validation
        ]
        summary = summarize(episodes)
        score = (*driving_score(summary, prioritize_clean_time=True), summary["mean_return"])
        record = {
            "generation": generation,
            "timesteps": generation * steps_per_generation,
            "rollout_seed": rollout_seed,
            "fitness": best_fitness,
            "nodes": len(champion.network.node_evals),
            "summary": summary,
            "episodes": episodes,
        }
        history.append(record)
        write_json(output / f"generation-{generation}.json", record)
        if best_score is None or score > best_score:
            best_score = score
            for name, payload in (
                ("best_model", _policy_data(champion, config)),
                ("best_selection", record),
            ):
                pending = output / f"{name}.tmp"
                write_json(pending, payload)
                pending.replace(output / f"{name}.json")
        print(
            f"[neat] generation={generation} steps={record['timesteps']} "
            f"laps={summary['completed_laps']}/{summary['episodes']} "
            f"coverage={summary['mean_coverage']:.3f}",
            flush=True,
        )

    started = time.perf_counter()
    random_state = random.getstate()
    try:
        population = neat.Population(evolution, seed=settings.seed)
        population.run(evaluate_population, settings.total_timesteps // steps_per_generation)
    finally:
        random.setstate(random_state)
        env.close()
    result = {
        "kind": "neat",
        "actual_timesteps": len(history) * steps_per_generation,
        "generations": len(history),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "training_results.json", result)
    return result
