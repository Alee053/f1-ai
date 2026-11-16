from pathlib import Path

import numpy as np
import pygame
import pytest

from f1_ai.config import load_config
from f1_ai.controls import DrivingInput, XboxInput, stick_value, trigger_value
from f1_ai.manual import ManualSession, export_config, run_manual


def test_xbox_stick_deadzone_curve_and_endpoints():
    assert stick_value(-32768) == -1
    assert stick_value(32767) == 1
    assert stick_value(-845) == 0
    assert 0 < stick_value(16000) < 0.5
    assert stick_value(-16000) == pytest.approx(-stick_value(16000), abs=1e-4)


def test_triggers_are_released_at_zero_and_clipped_at_full_scale():
    assert trigger_value(0) == trigger_value(500) == 0
    assert trigger_value(32767) == trigger_value(32768) == 1
    assert 0 < trigger_value(16000) < 1


def test_brake_has_priority_when_both_triggers_are_pressed():
    np.testing.assert_array_equal(DrivingInput(0.5, 1, 0.4).action, [0.5, -np.float32(0.4)])
    np.testing.assert_array_equal(DrivingInput(-1, 1, 0).action, [-1, 1])


def test_controller_disconnect_returns_neutral_and_requires_resume(monkeypatch):
    from pygame._sdl2 import controller

    class Pad:
        id = 7
        name = "Test Xbox"
        connected = True

        def attached(self):
            return self.connected

        def get_axis(self, axis):
            return {
                pygame.CONTROLLER_AXIS_LEFTX: 10000,
                pygame.CONTROLLER_AXIS_TRIGGERRIGHT: 32767,
                pygame.CONTROLLER_AXIS_TRIGGERLEFT: 0,
            }[axis]

        def quit(self):
            pass

    pad = Pad()
    monkeypatch.setattr(controller, "init", lambda: None)
    monkeypatch.setattr(controller, "get_count", lambda: 1 if pad.connected else 0)
    monkeypatch.setattr(controller, "is_controller", lambda index: True)
    monkeypatch.setattr(controller, "Controller", lambda index: pad)
    device = XboxInput()
    assert device.read().throttle == 1
    pad.connected = False
    assert device.refresh()
    np.testing.assert_array_equal(device.read().action, [0, 0])
    assert device.instance_id is None
    pad.connected = True
    assert device.refresh()
    assert device.instance_id == 7
    device.close()


def test_manual_tuning_and_export_round_trip_preserve_physics(tmp_path):
    config = load_config("configs/formula.toml")
    session = ManualSession(config, 0)
    try:
        session.command("tuning")
        session.command("increase")
        assert session.paused
        assert session.config.environment.formula.downforce_cla_m2 == 5.5
        session.command("aero")
        assert session.config.environment.formula.downforce_cla_m2 == 0
        session.command("aero")
        assert session.config.environment.formula.downforce_cla_m2 == 5.5
        path = tmp_path / "adjustment.toml"
        export_config(session.config, path)
        assert load_config(path) == session.config
        assert load_config(path).environment.max_steps == config.environment.max_steps
        assert session.env.config.max_steps == 180000
        with pytest.raises(FileExistsError):
            export_config(session.config, path)
        session.command("next_track")
        assert session.seed == 1 and session.paused
    finally:
        session.close()


def test_manual_practice_does_not_use_final_test_circuits():
    with pytest.raises(ValueError, match="entrenamiento"):
        ManualSession(load_config("configs/formula.toml"), 200)


def test_manual_headless_renders_and_saves_complete_configuration(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    # La comprobación no depende de tener un mando real conectado al ejecutar tests.
    from pygame._sdl2 import controller

    monkeypatch.setattr(controller, "get_count", lambda: 0)
    output = tmp_path / "manual"
    result = run_manual(
        load_config("configs/formula.toml"), seed=0, output=output, max_frames=2, headless=True
    )
    assert result["frames"] == 2
    assert (output / "last-frame.png").stat().st_size > 1000
    assert load_config(output / "last-config.toml").environment.vehicle_model == "formula-v1"
