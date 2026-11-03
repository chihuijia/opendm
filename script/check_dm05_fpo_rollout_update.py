#!/usr/bin/env python3
"""Run one value+actor FPO update from a saved LIBERO chunk rollout."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor

from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.transforms import (
    ChatTokenization,
    Normalize,
    PadAction,
    PixelTransform,
)
from opendm.model.dm05.dm05_arch import (
    DM05Config,
    DM05ForConditionalGeneration,
)
from opendm.model.dm05.dm05_lora import (
    _apply_attention_config_for_inference,
    unwrap_dm05_model,
)
from opendm.model.dm05.fpo_objective import (
    cfm_log_ratio,
    clipped_fpo_actor_loss,
    compute_semimarkov_gae,
)
from opendm.model.dm05.fpo_value import DM05FPOValueHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--sim-root", default="/workspace/opendm-sim")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actor-learning-rate", type=float, default=3e-7)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-actor-chunks", type=int, default=16)
    parser.add_argument("--actor-micro-batch-size", type=int, default=4)
    parser.add_argument("--n-cfm-samples", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_rollout_file(path: Path, sim_root: str) -> dict:
    sys.path.insert(0, sim_root)
    from evaluation.utils.libero_fpo_rollout import aggregate_chunk_transitions
    from evaluation.utils.libero_rl_reward import (
        LiberoRewardConfig,
        compute_potential_shaped_rewards,
    )

    with np.load(path) as data:
        rollout = {key: data[key] for key in data.files}
    metadata = json.loads(str(rollout["metadata_json"].item()))
    shaped = compute_potential_shaped_rewards(
        rollout["goal_predicates_before"],
        rollout["goal_predicates_after"],
        rollout["dones"],
        rollout["truncated"],
        config=LiberoRewardConfig(gamma=0.995),
    )
    chunks = aggregate_chunk_transitions(
        shaped["reward"],
        rollout["dones"],
        rollout["truncated"],
        rollout["chunk_index"],
        rollout["sampled_chunks"],
        gamma=0.995,
        gae_lambda=0.99,
    )
    rollout["metadata"] = metadata
    rollout["chunk_transitions"] = chunks
    rollout["path"] = str(path)
    return rollout


def load_rollouts(args: argparse.Namespace) -> list[dict]:
    rollout_path = Path(args.rollout)
    if rollout_path.is_dir():
        paths = sorted(rollout_path.glob("*.npz"))
    else:
        paths = [rollout_path]
    if not paths:
        raise ValueError(f"No rollout NPZ files found at {rollout_path}.")
    return [load_rollout_file(path, args.sim_root) for path in paths]


def build_model(args: argparse.Namespace):
    with open(
        f"{args.checkpoint}/adapter_config.json", encoding="utf-8"
    ) as adapter_file:
        base_model_path = json.load(adapter_file)["base_model_name_or_path"]

    config = DM05Config.from_pretrained(base_model_path)
    _apply_attention_config_for_inference(
        config,
        llm_attn_implementation="sdpa",
        vision_attn_implementation="sdpa",
        action_attn_implementation="sdpa",
    )
    base_model = DM05ForConditionalGeneration.from_pretrained(
        base_model_path,
        config=config,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint,
        is_trainable=True,
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    return model, processor


def build_rollout_batch(
    rollout: dict,
    processor,
    norm_stats_path: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    metadata = rollout["metadata"]
    goal = metadata.get("model_goal", metadata["task_description"])
    prompt = metadata.get(
        "model_prompt",
        f"What action should the robot take to {goal}?",
    )
    chunks = rollout["sampled_chunks"]
    chunk_valid = rollout["chunk_action_valid_mask"]
    agent_images = rollout["chunk_observation_agent_images"]
    wrist_images = rollout["chunk_observation_wrist_images"]
    states = rollout["chunk_observation_states"]

    pixel_transform = PixelTransform(NoAugmentationPipeline())
    normalizer = Normalize(
        norm_stats_path=norm_stats_path,
        norm_keys=["state", "action"],
        use_quantiles=True,
    )
    tokenizer = ChatTokenization(
        processor=processor,
        max_length=768,
        image_prompts=["Head", "Left wrist"],
        add_state=False,
    )
    pad_action = PadAction(32)
    collator = TrainingCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=768,
    )

    samples = []
    normalized_states = []
    chunk_count = chunks.shape[0]
    for observation_index in range(chunk_count + 1):
        action_index = min(observation_index, max(chunk_count - 1, 0))
        action = (
            chunks[action_index].copy()
            if chunk_count
            else np.zeros((10, 7), dtype=np.float32)
        )
        if observation_index < chunk_count:
            valid = chunk_valid[observation_index]
        else:
            valid = np.zeros(action.shape[0], dtype=np.bool_)
        action_mask = np.broadcast_to(valid[:, None], action.shape).copy()
        sample = {
            "images": [
                Image.fromarray(agent_images[observation_index]).convert("RGB"),
                Image.fromarray(wrist_images[observation_index]).convert("RGB"),
            ],
            "prompt": prompt,
            "state": states[observation_index].copy(),
            # TrainingCollator concatenates the leading singleton dimension.
            "action": action[None, ...],
            "action_mask": action_mask[None, ...],
            "meta_data": {"robot_type": "Franka", "speed": "0.5"},
        }
        sample = pixel_transform(sample)
        sample = normalizer(sample)
        normalized_states.append(torch.from_numpy(sample["state"].copy()))
        sample = tokenizer(sample)
        sample = pad_action(sample)
        samples.append(sample)

    return collator(samples), torch.stack(normalized_states)


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def concatenate_batches(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = batches[0].keys()
    if any(batch.keys() != keys for batch in batches[1:]):
        raise ValueError("All rollout batches must contain the same tensor keys.")
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in keys
    }


def select_observations(
    batch: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    batch_size = batch["input_ids"].shape[0]
    images_per_observation = batch["pixel_values"].shape[0] // batch_size
    return {
        "input_ids": batch["input_ids"][start:end],
        "attention_mask": batch["attention_mask"][start:end],
        "token_type_ids": batch["token_type_ids"][start:end],
        "pixel_values": batch["pixel_values"][
            start * images_per_observation : end * images_per_observation
        ],
    }


def index_observations(
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = batch["input_ids"].shape[0]
    images_per_observation = batch["pixel_values"].shape[0] // batch_size
    pixel_shape = batch["pixel_values"].shape[1:]
    grouped_pixels = batch["pixel_values"].reshape(
        batch_size,
        images_per_observation,
        *pixel_shape,
    )
    selected_pixels = grouped_pixels.index_select(0, indices).reshape(
        -1,
        *pixel_shape,
    )
    return {
        "input_ids": batch["input_ids"].index_select(0, indices),
        "attention_mask": batch["attention_mask"].index_select(0, indices),
        "token_type_ids": batch["token_type_ids"].index_select(0, indices),
        "pixel_values": selected_pixels,
    }


def extract_all_prefix_features(
    dm05_model,
    batch: dict[str, torch.Tensor],
    micro_batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_batches = []
    mask_batches = []
    for start in range(0, batch["input_ids"].shape[0], micro_batch_size):
        inputs = select_observations(
            batch,
            start,
            min(start + micro_batch_size, batch["input_ids"].shape[0]),
        )
        features, mask = dm05_model.fpo_prefix_features(**inputs)
        feature_batches.append(features)
        mask_batches.append(mask)
    return torch.cat(feature_batches), torch.cat(mask_batches)


def evaluate_cfm_samples(
    model,
    actor_batch: dict[str, torch.Tensor],
    flow_noises: torch.Tensor,
    flow_times: torch.Tensor,
    *,
    requires_grad: bool,
) -> torch.Tensor:
    """Return per-action CFM losses with shape [B, H, N]."""

    sample_losses = []
    context = torch.enable_grad if requires_grad else torch.no_grad
    with context():
        for sample_index in range(flow_times.shape[0]):
            output = model(
                **actor_batch,
                flow_noise=flow_noises[sample_index],
                flow_time=flow_times[sample_index],
            )
            sample_losses.append(output.per_action_fm)
    return torch.stack(sample_losses, dim=-1)


def backward_fpo_actor_streaming(
    model,
    actor_batch: dict[str, torch.Tensor],
    flow_noises: torch.Tensor,
    flow_times: torch.Tensor,
    action_valid: torch.Tensor,
    log_ratio: torch.Tensor,
    actor_advantage: torch.Tensor,
    *,
    clip_epsilon: float,
    micro_batch_size: int,
) -> None:
    """Backpropagate the exact FPO gradient while holding one CFM graph at a time."""

    proxy_log_ratio = log_ratio.detach().requires_grad_(True)
    proxy_output = clipped_fpo_actor_loss(
        proxy_log_ratio,
        actor_advantage,
        clip_epsilon=clip_epsilon,
    )
    log_ratio_gradient = torch.autograd.grad(
        proxy_output.loss,
        proxy_log_ratio,
    )[0].detach()
    valid = action_valid.to(
        device=actor_batch["action"].device,
        dtype=actor_batch["action"].dtype,
    )
    sample_count = flow_times.shape[0]

    batch_size = actor_batch["input_ids"].shape[0]
    for sample_index in range(sample_count):
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_batch = select_observations(
                actor_batch,
                start,
                end,
            )
            micro_batch["action"] = actor_batch["action"][start:end]
            micro_batch["action_mask"] = actor_batch["action_mask"][start:end]
            output = model(
                **micro_batch,
                flow_noise=flow_noises[sample_index, start:end],
                flow_time=flow_times[sample_index, start:end],
            )
            # log_ratio = old - mean_N(sum_H(current)); old is frozen.
            current_log_ratio_contribution = -(
                output.per_action_fm * valid[start:end]
            ).sum(dim=1) / sample_count
            (
                current_log_ratio_contribution
                * log_ratio_gradient[start:end]
            ).sum().backward()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rollouts = load_rollouts(args)
    model, processor = build_model(args)
    rollout_batches = []
    rollout_states = []
    current_observation_indices = []
    trajectory_chunk_ranges = []
    chunk_success_labels = []
    observation_offset = 0
    chunk_offset = 0
    for rollout in rollouts:
        rollout_batch, normalized_state = build_rollout_batch(
            rollout,
            processor,
            args.norm_stats,
        )
        chunk_count = int(rollout["sampled_chunks"].shape[0])
        if chunk_count == 0:
            raise ValueError(f"Rollout {rollout['path']} has no action chunks.")
        rollout_batches.append(rollout_batch)
        rollout_states.append(normalized_state)
        current_observation_indices.extend(
            range(observation_offset, observation_offset + chunk_count)
        )
        trajectory_chunk_ranges.append(
            (chunk_offset, chunk_offset + chunk_count)
        )
        chunk_success_labels.extend(
            [bool(rollout["metadata"].get("success", False))] * chunk_count
        )
        observation_offset += chunk_count + 1
        chunk_offset += chunk_count

    batch = concatenate_batches(rollout_batches)
    normalized_states = torch.cat(rollout_states)
    batch = batch_to_device(batch, args.device)
    normalized_states = normalized_states.to(args.device)
    chunk_count = len(current_observation_indices)
    if args.n_cfm_samples <= 0:
        raise ValueError("n_cfm_samples must be positive.")
    if args.actor_micro_batch_size <= 0:
        raise ValueError("actor_micro_batch_size must be positive.")
    if args.target_kl <= 0:
        raise ValueError("target_kl must be positive.")

    dm05_model = unwrap_dm05_model(model)
    prefix_features, prefix_masks = extract_all_prefix_features(dm05_model, batch)
    value_head = DM05FPOValueHead(
        prefix_dim=prefix_features.shape[-1],
        state_dim=normalized_states.shape[-1],
    ).to(device=args.device, dtype=prefix_features.dtype)
    resume_path = Path(args.checkpoint) / "fpo_training_state.pt"
    resume_state = None
    if resume_path.is_file():
        resume_state = torch.load(
            resume_path,
            map_location=args.device,
            weights_only=False,
        )
        value_head.load_state_dict(resume_state["value_head"])
    current_indices = torch.tensor(
        current_observation_indices,
        device=args.device,
        dtype=torch.long,
    )
    next_indices = current_indices + 1
    values = value_head(
        prefix_features.index_select(0, current_indices),
        prefix_masks.index_select(0, current_indices),
        normalized_states.index_select(0, current_indices),
    )
    with torch.no_grad():
        next_values = value_head(
            prefix_features.index_select(0, next_indices),
            prefix_masks.index_select(0, next_indices),
            normalized_states.index_select(0, next_indices),
        )

    advantage_parts = []
    return_parts = []
    for rollout, (start, end) in zip(
        rollouts,
        trajectory_chunk_ranges,
        strict=True,
    ):
        transitions = rollout["chunk_transitions"]
        rewards = (
            torch.from_numpy(transitions["rewards"])
            .to(args.device)
            .unsqueeze(0)
        )
        discounts = (
            torch.from_numpy(transitions["discounts"])
            .to(args.device)
            .unsqueeze(0)
        )
        gae_discounts = (
            torch.from_numpy(transitions["gae_discounts"])
            .to(args.device)
            .unsqueeze(0)
        )
        trajectory_advantages, trajectory_returns = compute_semimarkov_gae(
            rewards,
            values.detach()[start:end].unsqueeze(0),
            next_values.detach()[start:end].unsqueeze(0),
            discounts,
            gae_discounts,
        )
        advantage_parts.append(trajectory_advantages.squeeze(0))
        return_parts.append(trajectory_returns.squeeze(0))
    advantages = torch.cat(advantage_parts)
    returns = torch.cat(return_parts).detach()

    value_optimizer = torch.optim.AdamW(
        value_head.parameters(),
        lr=args.value_learning_rate,
        weight_decay=0.0,
    )
    if resume_state is not None:
        value_optimizer.load_state_dict(resume_state["value_optimizer"])
    for parameter_group in value_optimizer.param_groups:
        parameter_group["lr"] = args.value_learning_rate
    value_loss_before = torch.nn.functional.mse_loss(values, returns)
    value_optimizer.zero_grad(set_to_none=True)
    value_loss_before.backward()
    torch.nn.utils.clip_grad_norm_(value_head.parameters(), 25.0)
    value_optimizer.step()
    with torch.no_grad():
        value_loss_after = torch.nn.functional.mse_loss(
            value_head(
                prefix_features.index_select(0, current_indices),
                prefix_masks.index_select(0, current_indices),
                normalized_states.index_select(0, current_indices),
            ),
            returns,
        )

    actor_count = min(args.max_actor_chunks, chunk_count)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    success_labels = torch.tensor(
        chunk_success_labels,
        device=args.device,
        dtype=torch.bool,
    )
    successful_indices = torch.nonzero(
        success_labels,
        as_tuple=False,
    ).flatten()
    failed_indices = torch.nonzero(
        ~success_labels,
        as_tuple=False,
    ).flatten()
    if successful_indices.numel() and failed_indices.numel():
        successful_count = min(
            actor_count // 2,
            successful_indices.numel(),
        )
        failed_count = min(
            actor_count - successful_count,
            failed_indices.numel(),
        )
        successful_count = min(
            actor_count - failed_count,
            successful_indices.numel(),
        )
        actor_chunk_indices = torch.cat(
            [
                successful_indices[
                    torch.randperm(
                        successful_indices.numel(),
                        generator=generator,
                        device=args.device,
                    )[:successful_count]
                ],
                failed_indices[
                    torch.randperm(
                        failed_indices.numel(),
                        generator=generator,
                        device=args.device,
                    )[:failed_count]
                ],
            ]
        )
    else:
        actor_chunk_indices = torch.randperm(
            chunk_count,
            generator=generator,
            device=args.device,
        )[:actor_count]
    actor_count = actor_chunk_indices.numel()
    actor_observation_indices = current_indices.index_select(
        0,
        actor_chunk_indices,
    )
    actor_batch = index_observations(batch, actor_observation_indices)
    actor_batch["action"] = batch["action"].index_select(
        0,
        actor_observation_indices,
    )
    actor_batch["action_mask"] = batch["action_mask"].index_select(
        0,
        actor_observation_indices,
    )
    actor_advantage = advantages.index_select(0, actor_chunk_indices)
    actor_advantage = (
        actor_advantage - actor_advantage.mean()
    ) / actor_advantage.std(unbiased=False).clamp_min(1e-6)

    flow_noises = torch.randn(
        (args.n_cfm_samples, *actor_batch["action"].shape),
        generator=generator,
        device=args.device,
        dtype=actor_batch["action"].dtype,
    )
    flow_times = torch.rand(
        (args.n_cfm_samples, actor_count),
        generator=generator,
        device=args.device,
        dtype=actor_batch["action"].dtype,
    )
    old_per_action_fm = evaluate_cfm_samples(
        model,
        actor_batch,
        flow_noises,
        flow_times,
        requires_grad=False,
    )
    current_per_action_fm = evaluate_cfm_samples(
        model,
        actor_batch,
        flow_noises,
        flow_times,
        requires_grad=False,
    )
    action_valid = actor_batch["action_mask"].any(dim=-1)
    log_ratio = cfm_log_ratio(
        old_per_action_fm,
        current_per_action_fm,
        action_valid,
    )
    actor_output = clipped_fpo_actor_loss(
        log_ratio,
        actor_advantage,
        clip_epsilon=0.01,
    )

    actor_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.actor_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    if resume_state is not None:
        actor_optimizer.load_state_dict(resume_state["actor_optimizer"])
    for parameter_group in actor_optimizer.param_groups:
        parameter_group["lr"] = args.actor_learning_rate
    actor_optimizer.zero_grad(set_to_none=True)
    backward_fpo_actor_streaming(
        model,
        actor_batch,
        flow_noises,
        flow_times,
        action_valid,
        log_ratio,
        actor_advantage,
        clip_epsilon=0.01,
        micro_batch_size=args.actor_micro_batch_size,
    )
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        25.0,
    )
    actor_optimizer.step()

    updated_per_action_fm = evaluate_cfm_samples(
        model,
        actor_batch,
        flow_noises,
        flow_times,
        requires_grad=False,
    )
    updated_log_ratio = cfm_log_ratio(
        old_per_action_fm,
        updated_per_action_fm,
        action_valid,
    )
    updated_actor = clipped_fpo_actor_loss(
        updated_log_ratio,
        actor_advantage,
        clip_epsilon=0.01,
    )

    result = {
        "source_checkpoint": str(Path(args.checkpoint)),
        "rollout_source": str(Path(args.rollout)),
        "seed": args.seed,
        "actor_learning_rate": args.actor_learning_rate,
        "value_learning_rate": args.value_learning_rate,
        "rollout_count": len(rollouts),
        "successful_rollouts": sum(
            bool(rollout["metadata"].get("success", False))
            for rollout in rollouts
        ),
        "chunk_count": chunk_count,
        "actor_chunk_count": actor_count,
        "actor_micro_batch_size": args.actor_micro_batch_size,
        "actor_success_chunk_count": int(
            success_labels.index_select(0, actor_chunk_indices).sum().item()
        ),
        "n_cfm_samples": args.n_cfm_samples,
        "return_mean": float(returns.mean()),
        "advantage_mean_before_normalization": float(advantages.mean()),
        "value_loss_before": float(value_loss_before.detach()),
        "value_loss_after": float(value_loss_after.detach()),
        "actor_loss_before": float(actor_output.loss.detach()),
        "actor_loss_after": float(updated_actor.loss.detach()),
        "actor_grad_norm": float(actor_grad_norm),
        "ratio_before_mean": float(actor_output.ratio.mean().detach()),
        "ratio_after_mean": float(updated_actor.ratio.mean().detach()),
        "clip_fraction_after": float(updated_actor.clip_fraction.detach()),
        "approx_kl_after": float(updated_actor.approx_kl.detach()),
        "target_kl": args.target_kl,
        "target_kl_exceeded": bool(
            updated_actor.approx_kl.detach().item() > args.target_kl
        ),
        "actor_surrogate_improved": bool(
            updated_actor.loss.detach().item()
            < actor_output.loss.detach().item()
        ),
    }
    print(json.dumps(result, indent=2))

    if not result["value_loss_after"] < result["value_loss_before"]:
        raise RuntimeError("Value update did not reduce value loss.")
    if not np.isfinite(result["actor_grad_norm"]):
        raise RuntimeError("Actor gradient norm is not finite.")
    if result["target_kl_exceeded"]:
        raise RuntimeError("Actor update exceeded the target KL.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        shutil.copy2(args.norm_stats, output_dir / "norm_stats.json")
        torch.save(
            {
                "value_head": value_head.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "prefix_dim": int(prefix_features.shape[-1]),
                "state_dim": int(normalized_states.shape[-1]),
                "source_rollout": str(Path(args.rollout)),
                "source_rollout_files": [
                    rollout["path"] for rollout in rollouts
                ],
            },
            output_dir / "fpo_training_state.pt",
        )
        with open(output_dir / "fpo_metrics.json", "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"Saved FPO checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
