"""Objetivo de tiempo con costes menores por controles bruscos y slip excesivo."""

import numpy as np


def lap_time_step_reward(
    progress_fraction: float,
    *,
    on_road: bool,
    action_delta: np.ndarray,
    max_abs_slip: float,
    sideslip_rad: float,
) -> float:
    """Cada vuelta válida suma el mismo avance; menos pasos conserva más retorno.

    Los bonos terminales se aplican en el entorno. No se pondera por distancia al
    eje ni por alineación con su tangente. Se permite el slip útil del neumático.
    """
    if not on_road:
        return -1.0
    smoothness = 0.02 * float(np.dot(action_delta, action_delta))
    excessive_slip = 0.1 * min(max(0.0, max_abs_slip - 0.2), 5.0) ** 2
    excessive_drift = 0.1 * max(0.0, abs(sideslip_rad) - 0.15) ** 2
    return 1000.0 * progress_fraction - 0.1 - smoothness - excessive_slip - excessive_drift
