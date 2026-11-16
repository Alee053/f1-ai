"""Resultados locales con metadatos y sin sobrescritura de experimentos anteriores."""

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from f1_ai.config import ExperimentConfig
from f1_ai.environment import OBSERVATION_NAMES, OBSERVATION_VERSION


def create_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def provenance(config: ExperimentConfig, *, kind: str) -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "kind": kind,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "python_version": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("gymnasium", "box2d", "stable-baselines3", "torch", "numpy")
        },
        "observation_version": OBSERVATION_VERSION,
        "observation_names": OBSERVATION_NAMES,
        "action_names": ["steering", "signed_acceleration"],
        "configuration": asdict(config),
    }
