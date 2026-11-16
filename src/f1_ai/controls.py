"""Entrada Xbox estandarizada por SDL2; sin números de eje específicos del driver."""

import math
from dataclasses import dataclass

import numpy as np
import pygame
from pygame._sdl2 import controller


@dataclass(frozen=True)
class DrivingInput:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    @property
    def action(self) -> np.ndarray:
        # Dos controles del entorno: el freno tiene prioridad si ambos gatillos se pulsan.
        acceleration = -self.brake if self.brake > 0 else self.throttle
        return np.array([self.steering, acceleration], dtype=np.float32)


def stick_value(raw: int, deadzone: float = 0.08, exponent: float = 1.6) -> float:
    if not math.isfinite(deadzone) or not 0 <= deadzone < 0.5:
        raise ValueError("La zona muerta debe pertenecer a [0, 0.5).")
    if not math.isfinite(exponent) or not 1 <= exponent <= 3:
        raise ValueError("La curva del stick debe pertenecer a [1, 3].")
    value = max(-1.0, min(1.0, raw / (32768 if raw < 0 else 32767)))
    magnitude = max(0.0, (abs(value) - deadzone) / (1 - deadzone)) ** exponent
    return math.copysign(magnitude, value)


def trigger_value(raw: int) -> float:
    value = max(0.0, min(1.0, raw / 32767))
    return max(0.0, (value - 0.03) / 0.97)


class XboxInput:
    """Detecta, abre y recupera mandos SDL; el llamador pausa si cambia la conexión."""

    def __init__(self, index: int | None = None, *, deadzone=0.08, exponent=1.6):
        stick_value(0, deadzone, exponent)
        if index is not None and index < 0:
            raise ValueError("El índice del mando debe ser no negativo.")
        controller.init()
        if index is not None and (
            index >= controller.get_count() or not controller.is_controller(index)
        ):
            raise ValueError("El mando solicitado no está disponible; ejecute f1-ai controllers.")
        self.index = index
        self.deadzone = deadzone
        self.exponent = exponent
        self.device = None
        self.status = "Sin mando; se puede usar el teclado"
        self.refresh()

    @property
    def instance_id(self):
        return self.device.id if self.device is not None else None

    @property
    def name(self):
        return self.device.name if self.device is not None else "Teclado"

    def refresh(self) -> bool:
        """Devuelve True si se pierde o recupera una conexión, para exigir reanudación."""
        changed = False
        if self.device is not None and not self.device.attached():
            self.device.quit()
            self.device = None
            self.status = "Mando desconectado; START o espacio para reanudar"
            changed = True
        if self.device is None:
            indices = range(controller.get_count()) if self.index is None else [self.index]
            for index in indices:
                if index < controller.get_count() and controller.is_controller(index):
                    self.device = controller.Controller(index)
                    self.status = f"Conectado: {self.device.name}"
                    changed = True
                    break
        return changed

    def read(self) -> DrivingInput:
        if self.device is None:
            return DrivingInput()
        try:
            if not self.device.attached():
                self.close()
                self.status = "Mando desconectado; START o espacio para reanudar"
                return DrivingInput()
            return DrivingInput(
                stick_value(
                    self.device.get_axis(pygame.CONTROLLER_AXIS_LEFTX), self.deadzone, self.exponent
                ),
                trigger_value(self.device.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)),
                trigger_value(self.device.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)),
            )
        except pygame.error as error:
            self.status = f"Error de mando: {error}"
            self.device.quit()
            self.device = None
            return DrivingInput()

    def close(self):
        if self.device is not None:
            self.device.quit()
            self.device = None


def connected_controllers() -> list[dict]:
    pygame.display.init()
    controller.init()
    pygame.event.pump()
    devices = []
    for index in range(controller.get_count()):
        if controller.is_controller(index):
            device = controller.Controller(index)
            try:
                devices.append(
                    {
                        "index": index,
                        "name": device.name,
                        "instance_id": device.id,
                        "mapping": device.get_mapping(),
                    }
                )
            finally:
                device.quit()
    return devices
