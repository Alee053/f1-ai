import numpy as np
import pytest

from f1_ai.geometry import CircuitGeometry, LapProgress


@pytest.fixture
def square():
    return CircuitGeometry(np.array([[0, 0], [0, 20], [20, 20], [20, 0]]))


def test_projection_and_lookahead_wrap_around_finish(square):
    projection = square.project(np.array([2, 10]))
    assert projection.distance == pytest.approx(10)
    assert projection.lateral == pytest.approx(2)
    np.testing.assert_allclose(square.sample(np.array([85, -5])), [[0, 5], [5, 0]])


def test_forward_lap_needs_finish_crossing_and_coverage():
    progress = LapProgress(80, 0, 0.95)
    for distance in range(5, 80, 5):
        gain, complete = progress.update(distance, 1.0)
        assert gain == pytest.approx(5)
        assert not complete
    assert progress.update(0, 1.0) == (5, True)


def test_backwards_lap_is_not_success_and_does_not_earn_progress():
    progress = LapProgress(80, 0, 0.95)
    for distance in range(75, -1, -5):
        assert progress.update(distance, 1.0) == (0, False)


def test_oscillation_cannot_farm_progress_reward():
    progress = LapProgress(80, 0, 0.95)
    assert progress.update(5, 0.1)[0] == 5
    assert progress.update(0, 0.1)[0] == 0
    assert progress.update(5, 0.1)[0] == 0


def test_incomplete_coverage_and_teleport_cannot_complete_lap():
    progress = LapProgress(80, 0, 0.95)
    for distance in range(5, 81, 5):
        assert not progress.update(distance % 80, 0.8)[1]
    progress = LapProgress(80, 0, 0.95)
    assert progress.update(70, 1.0) == (0, False)
    assert progress.update(0, 1.0)[1] is False


def test_degenerate_track_is_rejected():
    with pytest.raises(ValueError, match="longitud cero"):
        CircuitGeometry(np.zeros((3, 2)))
