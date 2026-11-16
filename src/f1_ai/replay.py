"""Vídeo de una política cargada, sin controlador de referencia en la inferencia."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.image import imsave

from f1_ai.agents import PPOAgent
from f1_ai.artifacts import write_json
from f1_ai.config import ExperimentConfig
from f1_ai.environment import make_env
from f1_ai.training import load_model, read_model_metadata


def record_replay(
    config: ExperimentConfig,
    model_path: Path,
    output: Path,
    *,
    seed: int = 100,
    stride: int = 2,
) -> dict:
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise FileNotFoundError("La exportación de vídeo requiere ffmpeg.")
    if output.suffix.lower() != ".mp4" or stride <= 0 or 50 % stride:
        raise ValueError("Use una salida MP4 y un salto de cuadros que divida 50.")
    metadata_path = output.with_suffix(".json")
    snapshot_path = output.with_suffix(".png")
    if any(path.exists() for path in (output, metadata_path, snapshot_path)):
        raise FileExistsError("Ya existe un archivo de esta exportación.")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    model = load_model(model_path, config)
    agent = PPOAgent(model)
    env = make_env(config.environment, render_mode="rgb_array")
    process = None
    frames = 0
    snapshot_saved = False
    try:
        observation, _ = env.reset(seed=seed)
        frame = env.render()
        height, width, channels = frame.shape
        if channels != 3:
            raise ValueError("El renderizado no produjo cuadros RGB.")
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
                str(50 // stride),
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
                str(output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
        frames += 1
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, info = env.step(agent.action(observation))
            if env.elapsed_steps % stride == 0 or terminated or truncated:
                frame = env.render()
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                frames += 1
                if not snapshot_saved and env.elapsed_steps >= 300:
                    imsave(snapshot_path, frame)
                    snapshot_saved = True
        process.stdin.close()
        assert process.stderr is not None
        errors = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg no pudo exportar el vídeo: {errors}")
        if not snapshot_saved:
            imsave(snapshot_path, frame)
        source = read_model_metadata(model_path)
        root_kind = source.get("initial_policy", {}).get("root_kind")
        method = "imitation+ppo" if root_kind in {"behavior_cloning", "dagger"} else source["kind"]
        result = {
            "model": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "method": method,
            "seed": seed,
            "frames": frames,
            "fps": 50 // stride,
            "simulated_seconds": info["simulated_seconds"],
            "coverage": info["coverage"],
            "lap_complete": info["lap_complete"],
            "termination_reason": info["termination_reason"],
            "off_track_steps": info["off_track_steps"],
            "neural_policy_only": True,
            "hud_reward": "native CarRacing score; evaluation uses its documented reward profile",
        }
        write_json(metadata_path, result)
        return result
    finally:
        env.close()
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait()
