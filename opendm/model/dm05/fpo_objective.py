"""Vanilla Flow Policy Optimization (FPO) actor objective for DM05."""

from dataclasses import dataclass

import torch


@dataclass
class FPOActorOutput:
    """Diagnostics returned with the scalar actor objective."""

    loss: torch.Tensor
    ratio: torch.Tensor
    log_ratio: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


def compute_semimarkov_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discounts: torch.Tensor,
    gae_discounts: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE when each policy transition spans multiple env steps."""

    tensors = (values, next_values, discounts, gae_discounts)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [B, T].")
    if any(tensor.shape != rewards.shape for tensor in tensors):
        raise ValueError("All GAE tensors must have the same shape [B, T].")
    if valid_mask is None:
        valid = torch.ones_like(rewards)
    else:
        if valid_mask.shape != rewards.shape:
            raise ValueError("valid_mask must match rewards.")
        valid = valid_mask.to(device=rewards.device, dtype=rewards.dtype)

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.shape[1])):
        delta = (
            rewards[:, step]
            + discounts[:, step] * next_values[:, step]
            - values[:, step]
        )
        last_advantage = (
            delta + gae_discounts[:, step] * last_advantage
        ) * valid[:, step]
        advantages[:, step] = last_advantage
    return advantages, advantages + values


def cfm_log_ratio(
    old_per_action_fm: torch.Tensor,
    current_per_action_fm: torch.Tensor,
    action_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert CFM loss improvement into one log-ratio proxy per transition.

    Args:
        old_per_action_fm: Frozen rollout losses with shape [B, H, N].
        current_per_action_fm: Current actor losses with shape [B, H, N].
        action_valid_mask: Executed action positions with shape [B, H].
    """

    if old_per_action_fm.ndim != 3:
        raise ValueError("old_per_action_fm must have shape [B, H, N].")
    if current_per_action_fm.shape != old_per_action_fm.shape:
        raise ValueError("current_per_action_fm must match old_per_action_fm.")
    if action_valid_mask.shape != old_per_action_fm.shape[:2]:
        raise ValueError("action_valid_mask must have shape [B, H].")

    valid = action_valid_mask.to(
        device=current_per_action_fm.device,
        dtype=current_per_action_fm.dtype,
    ).unsqueeze(-1)
    return ((old_per_action_fm.detach() - current_per_action_fm) * valid).sum(
        dim=1
    ).mean(dim=-1)


def clipped_fpo_actor_loss(
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    valid_mask: torch.Tensor | None = None,
) -> FPOActorOutput:
    """Apply PPO clipping to a differentiable CFM-derived log-ratio."""

    if log_ratio.ndim != 1:
        raise ValueError("log_ratio must be a 1D tensor.")
    if advantage.shape != log_ratio.shape:
        raise ValueError("advantage must match log_ratio.")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1).")

    actor_advantage = advantage.detach().to(
        device=log_ratio.device, dtype=log_ratio.dtype
    )
    if valid_mask is None:
        valid = torch.ones_like(log_ratio)
    else:
        if valid_mask.shape != log_ratio.shape:
            raise ValueError("valid_mask must match log_ratio.")
        valid = valid_mask.to(device=log_ratio.device, dtype=log_ratio.dtype)

    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)

    unclipped_objective = ratio * actor_advantage
    clipped_objective = clipped_ratio * actor_advantage
    surrogate = torch.minimum(unclipped_objective, clipped_objective)
    valid_count = valid.sum().clamp_min(1)
    loss = -(surrogate * valid).sum() / valid_count

    with torch.no_grad():
        clip_fraction = (
            ((ratio - 1).abs() > clip_epsilon).to(valid.dtype) * valid
        ).sum() / valid_count
        approx_kl = (((ratio - 1) - log_ratio) * valid).sum() / valid_count

    return FPOActorOutput(
        loss=loss,
        ratio=ratio,
        log_ratio=log_ratio,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
    )
