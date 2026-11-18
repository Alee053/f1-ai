"""Selección por vueltas completas antes de optimizar cobertura o velocidad."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from f1_ai.agents import PPOAgent
from f1_ai.artifacts import write_json
from f1_ai.config import LAP_TIME_REWARDS, ExperimentConfig
from f1_ai.evaluation import driving_score, evaluate_episode, summarize


class DrivingEvaluation(BaseCallback):
    def __init__(self, config: ExperimentConfig, output: Path):
        super().__init__()
        self.config = config
        self.output = output
        self.directory = output / "validation"
        self.directory.mkdir()
        self.checkpoints = output / "checkpoints"
        self.checkpoints.mkdir()
        self.next_evaluation = config.ppo.eval_every
        self.history: list[dict] = []
        self.best_score: tuple | None = None

    def _evaluate(self):
        episodes = [
            evaluate_episode(self.config.environment, PPOAgent(self.model), seed, agent_name="ppo")
            for seed in self.config.circuits.validation
        ]
        summary = summarize(episodes)
        ranking = driving_score(
            summary,
            prioritize_clean_time=self.config.environment.reward_profile in LAP_TIME_REWARDS,
        )
        score = (*ranking, summary["mean_return"])
        record = {"timesteps": self.num_timesteps, "summary": summary, "episodes": episodes}
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
            self.model.save(self.checkpoints / f"ppo_{self.num_timesteps}_steps.zip")
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.model.save(self.output / "best_model.zip")
            # Es un artefacto generado que se actualiza dentro de esta corrida.
            (self.output / "best_selection.json").write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
            )
        self.logger.record("eval/completion_rate", summary["completion_rate"])
        self.logger.record("eval/mean_coverage", summary["mean_coverage"])
        self.logger.record("eval/mean_reward", summary["mean_return"])
        if self.config.ppo.corner_curriculum and self.model.ep_info_buffer:
            recent = self.model.ep_info_buffer
            self.logger.record(
                "curriculum/recent_sector_episodes",
                sum(episode["training_sector"] for episode in recent),
            )
            self.logger.record("curriculum/recent_episodes", len(recent))
            self.logger.record("curriculum/latest_episode_stage", recent[-1]["curriculum_stage"])
            if self.config.ppo.recovery_curriculum:
                self.logger.record(
                    "curriculum/recent_recovery_episodes",
                    sum(episode["training_recovery"] for episode in recent),
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
        return True
