"""Comandos reproducibles del proyecto, sin servicios externos."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from f1_ai.agents import PPOAgent, PursuitAgent, RandomAgent
from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import load_config
from f1_ai.environment import make_env
from f1_ai.evaluation import evaluate_agents


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Agente PPO de carreras 2D, SIS-252")
    root.add_argument("--config", type=Path, default=Path("configs/ppo.toml"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Comprobar el contrato de Gymnasium y las particiones")
    benchmark = commands.add_parser("benchmark", help="Medir pasos físicos sin renderizado")
    benchmark.add_argument("--steps", type=int, default=2000)
    benchmark.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluar referencias o un modelo PPO")
    evaluate.add_argument(
        "--agents", nargs="+", choices=["random", "pursuit", "ppo"], default=["random", "pursuit"]
    )
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference-speed", type=float, default=30.0)
    evaluate.add_argument("--reference-lateral-acceleration", type=float, default=8.0)
    evaluate.add_argument(
        "--model", type=Path, help="Checkpoint local de confianza con metadata.json"
    )
    training = commands.add_parser("train", help="Entrenar PPO o ejecutar una prueba de 512 pasos")
    training.add_argument("--smoke", action="store_true", help="Prueba de integración de 512 pasos")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int)
    training.add_argument("--timesteps", type=int)
    training.add_argument(
        "--initial-model", type=Path, help="Inicializar pesos desde un modelo compatible"
    )
    plot = commands.add_parser("plot", help="Exportar figuras de una evaluación existente")
    plot.add_argument("--evaluation", type=Path, required=True)
    training_plot = commands.add_parser(
        "plot-training", help="Exportar la curva de validación de PPO"
    )
    training_plot.add_argument("--run", type=Path, required=True)
    demonstrations = commands.add_parser(
        "demonstrations", help="Registrar demostraciones solo en train"
    )
    demonstrations.add_argument("--output", type=Path, required=True)
    demonstrations.add_argument("--target-speed", type=float, default=30.0)
    demonstrations.add_argument("--lateral-acceleration", type=float, default=8.0)
    imitation = commands.add_parser("clone", help="Aprender el actor por imitación, antes de PPO")
    imitation.add_argument("--dataset", type=Path, required=True)
    imitation.add_argument("--output", type=Path, required=True)
    imitation.add_argument("--epochs", type=int, default=50)
    imitation.add_argument("--learning-rate", type=float, default=1e-3)
    imitation.add_argument("--dagger-rounds", type=int, default=0)
    imitation.add_argument("--seed", type=int)
    return root


def run_check(config):
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as check_sb3

    env = make_env(config.environment)
    try:
        check_env(env, skip_render_check=True)
        check_sb3(env)
        if env.surf is not None or env.screen is not None:
            raise RuntimeError("Se generó una superficie gráfica durante la comprobación.")
        print("Gymnasium y Stable-Baselines3: OK. Estado numérico sin renderizado RGB.")
        print(
            f"Circuitos: {len(config.circuits.train)} entrenamiento, "
            f"{len(config.circuits.validation)} validación, {len(config.circuits.test)} prueba."
        )
    finally:
        env.close()


def run_benchmark(config, args):
    if args.steps <= 0:
        raise ValueError("El número de pasos debe ser positivo.")
    output = create_directory(args.output)
    env = make_env(replace(config.environment, max_steps=max(args.steps + 1, 3000)))
    agent = PursuitAgent()
    resets = 0
    started = time.perf_counter()
    try:
        observation, _ = env.reset(seed=config.circuits.train[0])
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = env.step(agent.action(observation))
            if terminated or truncated:
                resets += 1
                observation, _ = env.reset(
                    seed=config.circuits.train[resets % len(config.circuits.train)]
                )
        elapsed = time.perf_counter() - started
        results = {
            "physical_steps": args.steps,
            "wall_seconds": elapsed,
            "steps_per_second": args.steps / elapsed,
            "resets": resets,
            "rgb_surface_created": env.surf is not None,
        }
        write_json(output / "metadata.json", provenance(config, kind="benchmark"))
        write_json(output / "results.json", results)
        print(json.dumps(results, indent=2))
    finally:
        env.close()


def run_evaluate(config, args):
    if "ppo" in args.agents and args.model is None:
        raise ValueError("La evaluación PPO requiere --model.")
    if args.model is not None and "ppo" not in args.agents:
        raise ValueError("Se indicó un modelo sin incluir ppo en --agents.")
    model = None
    if args.model is not None:
        from f1_ai.training import load_model

        model = load_model(args.model, config)
    output = create_directory(args.output)
    factories = {}
    for name in args.agents:
        if name == "random":
            factories[name] = lambda seed: RandomAgent(config.ppo.seed, seed)
        elif name == "pursuit":
            factories[name] = lambda seed: PursuitAgent(
                args.reference_speed,
                args.reference_lateral_acceleration,
            )
        elif name == "ppo":
            factories[name] = lambda seed: PPOAgent(model)
    write_json(
        output / "metadata.json",
        {
            **provenance(config, kind="evaluation"),
            "split": args.split,
            "agents": args.agents,
            "reference": {
                "target_speed": args.reference_speed,
                "lateral_acceleration": args.reference_lateral_acceleration,
            },
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()
            if args.model is not None
            else None,
        },
    )
    results = evaluate_agents(
        config.environment,
        factories,
        getattr(config.circuits, args.split),
        output,
    )
    print(json.dumps(results["summaries"], indent=2, allow_nan=False))


def main():
    args = parser().parse_args()
    try:
        config = load_config(args.config)
        if getattr(args, "seed", None) is not None:
            config = replace(config, ppo=replace(config.ppo, seed=args.seed))
        if getattr(args, "timesteps", None) is not None:
            if args.smoke:
                raise ValueError("La prueba smoke fija su propio número de pasos.")
            config = replace(config, ppo=replace(config.ppo, total_timesteps=args.timesteps))
        match args.command:
            case "check":
                run_check(config)
            case "benchmark":
                run_benchmark(config, args)
            case "evaluate":
                run_evaluate(config, args)
            case "train":
                from f1_ai.training import train

                print(
                    json.dumps(
                        train(
                            config,
                            args.output,
                            smoke=args.smoke,
                            initial_model=args.initial_model,
                        ),
                        indent=2,
                    )
                )
            case "plot":
                from f1_ai.plotting import plot_evaluation

                for path in plot_evaluation(args.evaluation):
                    print(path)
            case "plot-training":
                from f1_ai.plotting import plot_training

                print(plot_training(args.run))
            case "demonstrations":
                from f1_ai.imitation import record_demonstrations

                result = record_demonstrations(
                    config,
                    args.output,
                    target_speed=args.target_speed,
                    lateral_acceleration=args.lateral_acceleration,
                )
                print(json.dumps(result, indent=2))
            case "clone":
                from f1_ai.imitation import clone

                result = clone(
                    config,
                    args.dataset,
                    args.output,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    dagger_rounds=args.dagger_rounds,
                )
                print(json.dumps(result, indent=2))
    except (ValueError, TypeError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
