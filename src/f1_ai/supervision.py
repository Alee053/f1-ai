"""Ajuste del actor MLP y repaso supervisado entre actualizaciones PPO."""

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.nn import functional as F


def actor_parameters(policy):
    return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()]


class ActorRehearsal:
    """Repasa minibatches aleatorios sin actualizar crítico ni exploración."""

    def __init__(self, model, data, *, seed=42, batch_size=256, updates=4, learning_rate=1e-4):
        if batch_size <= 0 or updates <= 0 or not 0 < learning_rate < 1:
            raise ValueError("Los parámetros de repaso supervisado no son válidos.")
        if not len(data["observations"]):
            raise ValueError("Se necesitan demostraciones para el repaso.")
        self.observations = torch.as_tensor(
            data["observations"], dtype=torch.float32, device=model.device
        )
        self.actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=model.device)
        self.parameters = actor_parameters(model.policy)
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.learning_rate = learning_rate
        self.generator = torch.Generator().manual_seed(seed)
        self.batch_size, self.updates = batch_size, updates

    def update(self, policy, *, weight=1.0):
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * weight
        policy.set_training_mode(True)
        total = 0.0
        for _ in range(self.updates):
            indices = torch.randint(
                len(self.observations), (self.batch_size,), generator=self.generator
            ).to(self.observations.device)
            mean = policy.get_distribution(self.observations[indices]).distribution.mean
            if policy.squash_output:
                mean = torch.tanh(mean)
            loss = F.mse_loss(mean, self.actions[indices])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 0.5)
            self.optimizer.step()
            total += float(loss.detach())
        return total / self.updates


class RehearsalPPO(PPO):
    """Alterna PPO estándar y repaso BC; no modifica la función de pérdida de PPO."""

    rehearsal_fade_steps = 0

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "actor_rehearsal"]

    def train(self):
        weight = 1.0
        if self.rehearsal_fade_steps:
            weight = max(0.0, 1.0 - self.num_timesteps / self.rehearsal_fade_steps)
        rehearsal = getattr(self, "actor_rehearsal", None)
        if weight and rehearsal is None:
            raise RuntimeError(
                "PPO con repaso requiere cargar sus demostraciones antes de aprender."
            )
        super().train()
        self.logger.record("train/rehearsal_weight", weight)
        if weight:
            loss = rehearsal.update(self.policy, weight=weight)
            self.logger.record("train/rehearsal_mse", loss)


def fit_actor(
    model: PPO,
    data: dict[str, np.ndarray],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    batch_size: int = 256,
) -> list[float]:
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate < 1:
        raise ValueError("La configuración de ajuste supervisado no es válida.")
    observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=model.device)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=model.device)
    parameters = actor_parameters(model.policy)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        order = torch.randperm(len(observations), generator=generator).to(model.device)
        total = 0.0
        for indices in order.split(batch_size):
            mean = model.policy.get_distribution(observations[indices]).distribution.mean
            if model.policy.squash_output:
                mean = torch.tanh(mean)
            loss = F.mse_loss(mean, actions[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(observations))
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"[imitation] epoch={epoch + 1} mse={losses[-1]:.6f}", flush=True)
    return losses
