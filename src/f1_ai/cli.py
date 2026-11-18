"""Comandos reproducibles del proyecto, sin servicios externos."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

from f1_ai.agents import PPOAgent, PursuitAgent, RandomAgent, ZeroCorrectionAgent
from f1_ai.artifacts import create_directory, provenance, write_json
from f1_ai.config import RACING_RESIDUAL, load_config
from f1_ai.environment import make_env
from f1_ai.evaluation import evaluate_agents


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Agente PPO de carreras 2D, SIS-252")
    root.add_argument("--config", type=Path, default=None)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Comprobar el contrato de Gymnasium y las particiones")
    benchmark = commands.add_parser("benchmark", help="Medir pasos físicos sin renderizado")
    benchmark.add_argument("--steps", type=int, default=2000)
    benchmark.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluar referencias o una política guardada")
    evaluate.add_argument(
        "--agents",
        nargs="+",
        choices=["random", "pursuit", "reference", "ppo", "bc", "bc-ppo", "neat"],
        default=["random", "pursuit"],
    )
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference-speed", type=float, default=30.0)
    evaluate.add_argument("--reference-lateral-acceleration", type=float, default=8.0)
    evaluate.add_argument(
        "--min-completion", type=float, help="Exigir una tasa mínima a cada agente"
    )
    evaluate.add_argument(
        "--model", type=Path, help="Checkpoint PPO local de confianza o política NEAT JSON"
    )
    training = commands.add_parser("train", help="Entrenar PPO o ejecutar una prueba de 512 pasos")
    training.add_argument("--smoke", action="store_true", help="Prueba de integración de 512 pasos")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int)
    training.add_argument("--timesteps", type=int)
    training.add_argument(
        "--rehearsal-dataset", type=Path, help="Alternar PPO con repaso BC de un dataset train"
    )
    training.add_argument(
        "--rehearsal-fade-steps", type=int, help="Pasos hasta reducir a cero el repaso BC"
    )
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initial-model", type=Path, help="Inicializar pesos desde un modelo compatible"
    )
    initialization.add_argument(
        "--resume-model",
        type=Path,
        help="Conservar pesos, optimizador y contador de un PPO guardado",
    )
    training.add_argument(
        "--transfer-vehicle",
        action="store_true",
        help="Permitir cambios en la física Formula v1 al usar --initial-model",
    )
    evolution = commands.add_parser(
        "train-neat", help="Evolucionar una política de mandos directos"
    )
    evolution.add_argument("--output", type=Path, required=True)
    evolution.add_argument("--seed", type=int)
    evolution.add_argument("--timesteps", type=int)
    evolution.add_argument("--rollout-steps", type=int, default=512)
    evolution.add_argument("--neat-config", type=Path, default=Path("configs/neat-road.ini"))
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
    demonstrations.add_argument(
        "--teacher", choices=["pursuit", "racing", "ppo"], default="pursuit"
    )
    demonstrations.add_argument("--teacher-model", type=Path, help="Checkpoint local del maestro")
    imitation = commands.add_parser("clone", help="Aprender el actor por imitación, antes de PPO")
    imitation.add_argument("--dataset", type=Path, required=True)
    imitation.add_argument("--output", type=Path, required=True)
    imitation.add_argument("--epochs", type=int, default=50)
    imitation.add_argument("--learning-rate", type=float, default=1e-3)
    imitation.add_argument("--dagger-rounds", type=int, default=0)
    imitation.add_argument(
        "--teacher-model", type=Path, help="Checkpoint del maestro PPO para DAgger"
    )
    imitation.add_argument("--seed", type=int)
    replay = commands.add_parser("replay", help="Exportar una vuelta de la red neuronal en MP4")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--seed", type=int, default=100)
    replay.add_argument("--device", choices=["cpu", "cuda"])
    play = commands.add_parser("play", help="Conducir con mando Xbox o teclado y ajustar la física")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--output", type=Path)
    play.add_argument("--controller", type=int)
    play.add_argument("--deadzone", type=float, default=0.08)
    play.add_argument("--steering-curve", type=float, default=1.6)
    play.add_argument(
        "--frames", type=int, help="Limitar cuadros para comprobaciones de la interfaz"
    )
    play.add_argument("--headless", action="store_true")
    for visual in (play, replay):
        visual.add_argument("--camera", choices=["top", "chase"], default="top")
        visual.add_argument("--grid-meters", type=float, default=1.0)
    commands.add_parser("controllers", help="Listar mandos detectados por SDL2")
    plan = commands.add_parser(
        "plan-reference", help="Planificar desde un mapa JSON guardado, sin simular ni entrenar"
    )
    plan.add_argument("--track", type=Path, required=True)
    plan.add_argument("--half-width", type=float, default=TRACK_WIDTH, help="Semianchura en metros")
    reference = commands.add_parser(
        "reference", help="Probar el experto con mapa global en una pista de entrenamiento"
    )
    reference.add_argument("--seed", type=int, default=0)
    reference.add_argument("--steps", type=int, help="Acotar pasos físicos de la comprobación")
    reference.add_argument("--lookahead-seconds", type=float, default=0.35)
    reference.add_argument("--speed-gain", type=float, default=0.8)
    for command in (plan, reference):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--samples", type=int, default=128)
        command.add_argument("--iterations", type=int, default=12)
        command.add_argument("--time-search-passes", type=int, default=0)
        command.add_argument("--max-speed", type=float, default=60, help="Límite en m/s")
        command.add_argument("--grip-fraction", type=float, default=0.75)
        command.add_argument("--braking-fraction", type=float, default=1.0)
        command.add_argument("--road-margin", type=float, default=0.75, help="Margen en metros")
    return root


def run_check(config):
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as check_sb3

    env = make_env(config.environment)
    try:
        check_env(env, skip_render_check=True)
        check_sb3(env)
        if env.unwrapped.surf is not None or env.unwrapped.screen is not None:
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
    agent = PursuitAgent(environment=config.environment)
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
    residual = config.environment.control_profile == RACING_RESIDUAL
    if "reference" in args.agents and not residual:
        raise ValueError("El agente reference requiere el perfil de control racing-residual-v1.")
    if args.min_completion is not None and not 0 <= args.min_completion <= 1:
        raise ValueError("La finalización mínima debe pertenecer a [0, 1].")
    learned = set(args.agents) & {"ppo", "bc", "bc-ppo", "neat"}
    if len(learned) > 1:
        raise ValueError("Un checkpoint solo puede evaluarse bajo una etiqueta de método.")
    if learned and args.model is None:
        raise ValueError("La evaluación de la red requiere --model.")
    if args.model is not None and not learned:
        raise ValueError("Se indicó un modelo sin incluir una red en --agents.")
    agent = None
    device = None
    if "neat" in learned:
        from f1_ai.evolution import load_neat_agent

        agent = load_neat_agent(args.model, config)
        device = "cpu"
    elif args.model is not None:
        from f1_ai.training import load_model, read_model_metadata

        model_metadata = read_model_metadata(args.model)
        root = model_metadata.get("initial_policy", {}).get("root_kind")
        imitated = model_metadata["kind"] in {"behavior_cloning", "dagger"}
        if "bc" in learned and not imitated:
            raise ValueError("La etiqueta bc requiere un modelo de imitación.")
        if "bc-ppo" in learned and root not in {"behavior_cloning", "dagger"}:
            raise ValueError("La etiqueta bc-ppo requiere PPO inicializado por imitación.")
        if "ppo" in learned and (imitated or root in {"behavior_cloning", "dagger"}):
            raise ValueError("El checkpoint incluye imitación: use bc o bc-ppo para identificarlo.")
        model = load_model(args.model, config)
        agent = PPOAgent(model)
        device = str(model.device)
    output = create_directory(args.output)
    factories = {}
    for name in args.agents:
        if name == "random":
            factories[name] = lambda seed: RandomAgent(config.ppo.seed, seed)
        elif name == "reference":
            factories[name] = lambda seed: ZeroCorrectionAgent()
        elif name == "pursuit":
            factories[name] = lambda seed: PursuitAgent(
                args.reference_speed,
                args.reference_lateral_acceleration,
                environment=config.environment,
            )
        elif agent is not None and name in learned:
            factories[name] = lambda seed: agent
    write_json(
        output / "metadata.json",
        {
            **provenance(config, kind="evaluation"),
            "split": args.split,
            "agents": args.agents,
            "uses_global_reference": residual,
            "reference": {
                "target_speed": args.reference_speed,
                "lateral_acceleration": args.reference_lateral_acceleration,
            },
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()
            if args.model is not None
            else None,
            "inference_device": device,
        },
    )
    results = evaluate_agents(
        config.environment,
        factories,
        getattr(config.circuits, args.split),
        output,
    )
    print(json.dumps(results["summaries"], indent=2, allow_nan=False))
    if args.min_completion is not None:
        failed = [
            name
            for name, summary in results["summaries"].items()
            if summary["completion_rate"] < args.min_completion
        ]
        if failed:
            raise SystemExit(f"No se alcanzó la finalización exigida: {', '.join(failed)}")


def main():
    args = parser().parse_args()
    try:
        if args.command == "controllers":
            from f1_ai.controls import connected_controllers

            print(json.dumps(connected_controllers(), indent=2))
            return
        default_config = "configs/ppo.toml"
        if args.command in {"play", "plan-reference", "reference"}:
            default_config = "configs/formula.toml"
        elif args.command == "train-neat":
            default_config = "configs/formula-road-cuda.toml"
        config_path = args.config or Path(default_config)
        config = load_config(config_path)
        if args.command in {"train", "train-neat", "clone"} and args.seed is not None:
            config = replace(config, ppo=replace(config.ppo, seed=args.seed))
        if getattr(args, "timesteps", None) is not None:
            if getattr(args, "smoke", False):
                raise ValueError("La prueba smoke fija su propio número de pasos.")
            config = replace(config, ppo=replace(config.ppo, total_timesteps=args.timesteps))
        if getattr(args, "rehearsal_fade_steps", None) is not None:
            config = replace(
                config, ppo=replace(config.ppo, rehearsal_fade_steps=args.rehearsal_fade_steps)
            )
        match args.command:
            case "plan-reference" | "reference":
                from f1_ai.racing_controller import FollowerConfig
                from f1_ai.racing_experiment import plan_saved_track, run_reference
                from f1_ai.racing_line import ReferenceConfig

                settings = ReferenceConfig(
                    samples=args.samples,
                    optimization_steps=args.iterations,
                    time_search_passes=args.time_search_passes,
                    speed_limit_m_s=args.max_speed,
                    grip_fraction=args.grip_fraction,
                    braking_fraction=args.braking_fraction,
                    road_margin_m=args.road_margin,
                )
                if args.command == "plan-reference":
                    result = plan_saved_track(
                        config, args.track, args.output, settings, half_width_m=args.half_width
                    )
                else:
                    result = run_reference(
                        config,
                        args.output,
                        seed=args.seed,
                        steps=args.steps,
                        settings=settings,
                        follower=FollowerConfig(
                            lookahead_seconds=args.lookahead_seconds, speed_gain=args.speed_gain
                        ),
                    )
                print(json.dumps(result, indent=2, allow_nan=False))
            case "play":
                from f1_ai.manual import run_manual

                print(
                    json.dumps(
                        run_manual(
                            config,
                            seed=args.seed,
                            output=args.output,
                            controller_index=args.controller,
                            deadzone=args.deadzone,
                            exponent=args.steering_curve,
                            max_frames=args.frames,
                            headless=args.headless,
                            camera=args.camera,
                            grid_m=args.grid_meters,
                        ),
                        indent=2,
                    )
                )
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
                            resume_model=args.resume_model,
                            rehearsal_dataset=args.rehearsal_dataset,
                            transfer_vehicle=args.transfer_vehicle,
                        ),
                        indent=2,
                    )
                )
            case "train-neat":
                from f1_ai.evolution import train_neat

                result = train_neat(
                    config, args.output, args.neat_config, rollout_steps=args.rollout_steps
                )
                print(json.dumps(result, indent=2, allow_nan=False))
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
                    teacher_kind=args.teacher,
                    teacher_model=args.teacher_model,
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
                    teacher_model=args.teacher_model,
                )
                print(json.dumps(result, indent=2))
            case "replay":
                from f1_ai.replay import record_replay

                print(
                    json.dumps(
                        record_replay(
                            config,
                            args.model,
                            args.output,
                            seed=args.seed,
                            camera=args.camera,
                            grid_m=args.grid_meters,
                            device=args.device,
                        ),
                        indent=2,
                    )
                )
    except (ValueError, TypeError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
