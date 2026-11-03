import torch

from opendm.model.dm05.fpo_objective import (
    cfm_log_ratio,
    clipped_fpo_actor_loss,
    compute_semimarkov_gae,
)
from opendm.model.dm05.fpo_value import DM05FPOValueHead


def test_cfm_log_ratio_uses_same_samples_and_only_executed_actions():
    old_loss = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0], [100.0, 100.0]]]
    )
    current_loss = torch.tensor(
        [[[0.5, 2.5], [1.0, 3.0], [0.0, 0.0]]], requires_grad=True
    )
    action_valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

    log_ratio = cfm_log_ratio(old_loss, current_loss, action_valid_mask)

    torch.testing.assert_close(log_ratio, torch.tensor([1.5]))


def test_clipped_fpo_actor_loss_matches_ppo_clipping():
    log_ratio = torch.log(torch.tensor([1.2, 0.8, 4.0]))
    advantage = torch.tensor([1.0, -1.0, 100.0])
    valid_mask = torch.tensor([1.0, 1.0, 0.0])

    output = clipped_fpo_actor_loss(
        log_ratio,
        advantage,
        clip_epsilon=0.1,
        valid_mask=valid_mask,
    )

    torch.testing.assert_close(output.loss, torch.tensor((-1.1 + 0.9) / 2))
    torch.testing.assert_close(output.clip_fraction, torch.tensor(1.0))


def test_positive_advantage_reduces_current_cfm_loss():
    old_loss = torch.ones(2, 1, 1)
    current_loss = torch.ones(2, 1, 1, requires_grad=True)
    action_valid_mask = torch.ones(2, 1)
    advantage = torch.tensor([1.0, -1.0])

    log_ratio = cfm_log_ratio(old_loss, current_loss, action_valid_mask)
    output = clipped_fpo_actor_loss(
        log_ratio,
        advantage,
        clip_epsilon=0.2,
    )
    output.loss.backward()

    assert current_loss.grad is not None
    assert current_loss.grad[0].item() > 0
    assert current_loss.grad[1].item() < 0


def test_semimarkov_gae_stops_at_episode_boundary_and_padding():
    rewards = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    gae_discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    valid_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    advantages, returns = compute_semimarkov_gae(
        rewards,
        values,
        next_values,
        discounts,
        gae_discounts,
        valid_mask,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[7.0, 2.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(returns, advantages)


def test_value_head_ignores_padded_prefix_tokens():
    torch.manual_seed(0)
    value_head = DM05FPOValueHead(
        prefix_dim=4,
        state_dim=2,
        hidden_dims=(8,),
    )
    prefix = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0, 6.0],
                [100.0, 100.0, 100.0, 100.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])
    states = torch.tensor([[0.5, -0.5]])

    pooled = value_head.pool_prefix(prefix, mask)
    value = value_head(prefix, mask, states)

    torch.testing.assert_close(
        pooled,
        torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
    )
    assert value.shape == (1,)
    value.sum().backward()
    assert any(parameter.grad is not None for parameter in value_head.parameters())


def test_streaming_cfm_backward_matches_monolithic_gradient():
    torch.manual_seed(3)
    sample_count, batch_size, horizon = 4, 3, 5
    old = torch.rand(sample_count, batch_size, horizon)
    current_monolithic = torch.rand(
        sample_count,
        batch_size,
        horizon,
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    advantage = torch.tensor([1.0, -0.5, 0.25])

    monolithic_log_ratio = cfm_log_ratio(
        old.permute(1, 2, 0),
        current_monolithic.permute(1, 2, 0),
        valid,
    )
    monolithic_loss = clipped_fpo_actor_loss(
        monolithic_log_ratio,
        advantage,
        clip_epsilon=0.2,
    ).loss
    monolithic_loss.backward()
    expected_gradient = current_monolithic.grad.detach().clone()

    current_streaming = current_monolithic.detach().clone().requires_grad_(True)
    with torch.no_grad():
        streaming_log_ratio = cfm_log_ratio(
            old.permute(1, 2, 0),
            current_streaming.permute(1, 2, 0),
            valid,
        )
    proxy_log_ratio = streaming_log_ratio.detach().requires_grad_(True)
    proxy_loss = clipped_fpo_actor_loss(
        proxy_log_ratio,
        advantage,
        clip_epsilon=0.2,
    ).loss
    log_ratio_gradient = torch.autograd.grad(
        proxy_loss,
        proxy_log_ratio,
    )[0]
    for sample_index in range(sample_count):
        log_ratio_contribution = -(
            current_streaming[sample_index] * valid
        ).sum(dim=1) / sample_count
        (log_ratio_contribution * log_ratio_gradient).sum().backward()

    torch.testing.assert_close(
        current_streaming.grad,
        expected_gradient,
    )
