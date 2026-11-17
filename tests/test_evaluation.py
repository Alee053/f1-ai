import pytest

from f1_ai.agents import PursuitAgent
from f1_ai.config import EnvironmentConfig
from f1_ai.evaluation import evaluate_episode, summarize


def test_failed_episodes_do_not_produce_a_lap_time():
    summary = summarize(
        [
            {
                "return": -10,
                "lap_complete": False,
                "lap_time_seconds": None,
                "coverage": 0.2,
                "off_track_fraction": 0.4,
            }
        ]
    )
    assert summary["completion_rate"] == 0
    assert summary["mean_lap_time_seconds"] is None
    assert summary["sample_std_return"] is None


def test_lap_time_averages_only_successes():
    common = {"return": 1, "coverage": 0.9, "off_track_fraction": 0}
    episodes = [
        {**common, "lap_complete": True, "lap_time_seconds": 30},
        {**common, "lap_complete": False, "lap_time_seconds": None},
    ]
    summary = summarize(episodes)
    assert summary["mean_lap_time_seconds"] == 30
    assert summary["completion_rate"] == pytest.approx(0.5)


def test_clean_laps_exclude_successes_with_off_track_steps():
    common = {"return": 1, "coverage": 1.0, "lap_time_seconds": 40}
    summary = summarize(
        [
            {**common, "lap_complete": True, "off_track_fraction": 0},
            {**common, "lap_complete": True, "off_track_fraction": 0.01},
            {**common, "lap_complete": False, "off_track_fraction": 0},
        ]
    )
    assert summary["completed_laps"] == 2
    assert summary["clean_completed_laps"] == 1


def test_reference_controller_completes_a_forward_lap():
    result = evaluate_episode(EnvironmentConfig(), PursuitAgent(), 0, agent_name="pursuit")
    assert result["lap_complete"]
    assert result["coverage"] >= 0.95
    assert result["forward_progress"] == pytest.approx(1.0)
    assert result["termination_reason"] == "lap_complete"
    assert result["lap_time_seconds"] > 0
