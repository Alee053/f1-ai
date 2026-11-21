"""Selección por vueltas completas antes de optimizar cobertura o velocidad."""

import json
from pathlib import Path
from statistics import fmean

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from f1_ai.agents import SB3Agent
from f1_ai.artifacts import write_json
from f1_ai.config import LAP_TIME_REWARDS, ExperimentConfig
from f1_ai.curriculum import CURRICULUM_FINAL_STAGE_STEPS
from f1_ai.evaluation import driving_score, evaluate_episode, summarize

PLATEAU_MIN_LAP_GAIN_SECONDS = 0.15


class ValidationPlateau:
    """Compara fiabilidad y tiempos sobre las mismas vueltas sin incidentes."""

    def __init__(self, patience: int, warmup_steps: int):
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.best_reliability = (-1, -1)
        self.best_times: dict[int, float] = {}
        self.stale_evaluations = 0

    def observe(self, timesteps: int, summary: dict, episodes: list[dict]) -> dict:
        times = {
            episode["circuit_seed"]: episode["lap_time_seconds"]
            for episode in episodes
            if episode["lap_complete"]
            and episode["off_track_steps"] == 0
            and episode["wheel_lock_steps"] == 0
        }
        reliability = (summary["completed_laps"], len(times))
        common = self.best_times.keys() & times.keys()
        gain = fmean(self.best_times[seed] - times[seed] for seed in common) if common else None
        improved = reliability > self.best_reliability or (
            reliability == self.best_reliability
            and gain is not None
            and gain >= PLATEAU_MIN_LAP_GAIN_SECONDS
        )
        if improved:
            self.best_reliability = reliability
            self.best_times = times
            self.stale_evaluations = 0
        elif timesteps >= self.warmup_steps:
            self.stale_evaluations += 1
        return {
            "improved": improved,
            "incident_free_laps": len(times),
            "mean_incident_free_lap_seconds": fmean(times.values()) if times else None,
            "comparable_laps": len(common),
            "mean_gain_seconds": gain,
            "stale_evaluations": self.stale_evaluations,
        }

    @property
    def reached(self) -> bool:
        return self.stale_evaluations >= self.patience


class DrivingEvaluation(BaseCallback):
    def __init__(
        self,
        config: ExperimentConfig,
        output: Path,
        *,
        agent_name: str = "ppo",
        plateau_patience: int = 0,
    ):
        super().__init__()
        self.config = config
        self.output = output
        self.agent_name = agent_name
        self.directory = output / "validation"
        self.directory.mkdir()
        self.checkpoints = output / "checkpoints"
        self.checkpoints.mkdir()
        self.next_evaluation = config.ppo.eval_every
        self.history: list[dict] = []
        self.best_score: tuple | None = None
        warmup = (
            CURRICULUM_FINAL_STAGE_STEPS * config.ppo.road_curriculum_scale
            if config.ppo.corner_curriculum
            else 0
        )
        self.plateau = ValidationPlateau(plateau_patience, warmup) if plateau_patience else None

    def _evaluate(self):
        episodes = [
            evaluate_episode(
                self.config.environment,
                SB3Agent(self.model),
                seed,
                agent_name=self.agent_name,
            )
            for seed in self.config.circuits.validation
        ]
        summary = summarize(episodes)
        record = {"timesteps": self.num_timesteps, "summary": summary, "episodes": episodes}
        if self.plateau is None:
            ranking = driving_score(
                summary,
                prioritize_clean_time=self.config.environment.reward_profile in LAP_TIME_REWARDS,
            )
            score = (*ranking, summary["mean_return"])
            improved = self.best_score is None or score > self.best_score
            if improved:
                self.best_score = score
        else:
            plateau = self.plateau.observe(self.num_timesteps, summary, episodes)
            record["plateau"] = plateau
            improved = plateau["improved"]
            self.logger.record("eval/incident_free_laps", plateau["incident_free_laps"])
            self.logger.record("eval/stale_evaluations", plateau["stale_evaluations"])
        self.history.append(record)
        np.savez_compressed(
            self.directory / "evaluations.npz",
            timesteps=np.array([row["timesteps"] for row in self.history]),
            results=np.array([[ep["return"] for ep in row["episodes"]] for row in self.history]),
            ep_lengths=np.array([[ep["steps"] for ep in row["episodes"]] for row in self.history]),
            successes=np.array(
                [[ep["lap_complete"] for ep in row["episodes"]] for row in self.history]
            ),
            coverage=np.array([[ep["coverage"] for ep in row["episodes"]] for row in self.history]),
        )
        write_json(self.directory / f"step-{self.num_timesteps}.json", record)
        if self.num_timesteps > 0:
            self.model.save(self.checkpoints / f"{self.agent_name}_{self.num_timesteps}_steps.zip")
        if improved:
            self.model.save(self.output / "best_model.zip")
            # Es un artefacto generado que se actualiza dentro de esta corrida.
            (self.output / "best_selection.json").write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
            )
        self.logger.record("eval/completion_rate", summary["completion_rate"])
        self.logger.record("eval/mean_coverage", summary["mean_coverage"])
        self.logger.record("eval/mean_reward", summary["mean_return"])
        if self.config.ppo.no_progress_steps and self.model.ep_info_buffer:
            failures = sum(
                episode["termination_reason"] == "no_progress"
                for episode in self.model.ep_info_buffer
            )
            self.logger.record("train/recent_no_progress_episodes", failures)
        if self.config.ppo.corner_curriculum and self.model.ep_info_buffer:
            recent = self.model.ep_info_buffer
            self.logger.record(
                "curriculum/recent_sector_episodes",
                sum(episode["training_sector"] for episode in recent),
            )
            self.logger.record("curriculum/recent_episodes", len(recent))
            self.logger.record("curriculum/latest_episode_stage", recent[-1]["curriculum_stage"])
            if self.config.ppo.road_start_curriculum and self.config.ppo.learning_pace_m_s:
                sectors = [episode for episode in recent if episode["training_sector"]]
                if sectors:
                    clean = sum(
                        episode["termination_reason"]
                        in {"road_launch_complete", "road_sector_complete"}
                        for episode in sectors
                    )
                    self.logger.record("curriculum/recent_clean_sector_rate", clean / len(sectors))
            if self.config.ppo.recovery_curriculum:
                self.logger.record(
                    "curriculum/recent_recovery_episodes",
                    sum(episode["training_recovery"] for episode in recent),
                )
            if self.config.ppo.launch_curriculum:
                self.logger.record(
                    "curriculum/recent_launch_episodes",
                    sum(episode["training_launch"] for episode in recent),
                )
            if self.config.ppo.wide_entry_curriculum:
                wide = [episode for episode in recent if episode["training_wide_entry"]]
                clean_seconds = [
                    episode["training_wide_sector_seconds"]
                    for episode in wide
                    if episode["training_wide_sector_seconds"] > 0
                ]
                self.logger.record("curriculum/recent_wide_entry_episodes", len(wide))
                if wide:
                    self.logger.record(
                        "curriculum/wide_sector_clean_rate", len(clean_seconds) / len(wide)
                    )
                if clean_seconds:
                    self.logger.record(
                        "curriculum/mean_clean_wide_sector_seconds", np.mean(clean_seconds)
                    )
        self.logger.dump(self.num_timesteps)
        print(
            f"[evaluation] steps={self.num_timesteps} "
            f"laps={summary['completed_laps']}/{summary['episodes']} "
            f"clean={summary['clean_completed_laps']} "
            f"coverage={summary['mean_coverage']:.3f} return={summary['mean_return']:.2f}",
            flush=True,
        )

    def _on_training_start(self):
        period = self.config.ppo.eval_every
        self.next_evaluation = (self.num_timesteps // period + 1) * period
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
            self.next_evaluation += self.config.ppo.eval_every
        return self.plateau is None or not self.plateau.reached
