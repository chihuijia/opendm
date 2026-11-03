#!/usr/bin/env python3
"""Run one real DM05 LoRA FPO gradient step on a LIBERO sample."""

import argparse
import json

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor

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
)
from opendm.model.dm05.fpo_value import DM05FPOValueHead
from playground.dm05_libero_lora import DM05DataConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--norm-stats-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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
    data_config = DM05DataConfig(
        dataset_name="libero_pi0_all",
        jsonl_dir=args.jsonl_dir,
        image_dir=args.image_dir,
        norm_stats_root=args.norm_stats_root,
    )
    dataset, collator = data_config.build_dataset(
        processor,
        action_horizon=10,
        tokenizer_max_length=768,
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=collator,
                num_workers=0,
            )
        )
    )
    batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    dm05_model = unwrap_dm05_model(model)
    prefix_features, prefix_valid_mask = dm05_model.fpo_prefix_features(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        token_type_ids=batch["token_type_ids"],
    )
    value_head = DM05FPOValueHead(
        prefix_dim=prefix_features.shape[-1],
        state_dim=8,
    ).to(device=args.device, dtype=prefix_features.dtype)
    value = value_head(
        prefix_features,
        prefix_valid_mask,
        torch.zeros(
            prefix_features.shape[0],
            8,
            device=args.device,
            dtype=prefix_features.dtype,
        ),
    )
    value.square().mean().backward()
    value_grad_finite = all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        for parameter in value_head.parameters()
    )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    flow_noise = torch.randn(
        batch["action"].shape,
        generator=generator,
        device=args.device,
        dtype=batch["action"].dtype,
    )
    flow_time = torch.full(
        (batch["action"].shape[0],),
        0.5,
        device=args.device,
        dtype=batch["action"].dtype,
    )

    forward_args = {
        **batch,
        "flow_noise": flow_noise,
        "flow_time": flow_time,
        "return_flow_details": True,
    }
    with torch.no_grad():
        old_output = model(**forward_args)
    current_output = model(**forward_args)

    action_valid_mask = batch["action_mask"].any(dim=-1)
    log_ratio = cfm_log_ratio(
        old_output.per_action_fm.unsqueeze(-1),
        current_output.per_action_fm.unsqueeze(-1),
        action_valid_mask,
    )
    actor_output = clipped_fpo_actor_loss(
        log_ratio,
        torch.ones_like(log_ratio),
        clip_epsilon=0.2,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    actor_output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()

    with torch.no_grad():
        updated_output = model(**forward_args)

    result = {
        "old_per_sample_fm": float(old_output.per_sample_fm.item()),
        "current_before_step": float(current_output.per_sample_fm.item()),
        "current_after_step": float(updated_output.per_sample_fm.item()),
        "initial_ratio": float(actor_output.ratio.item()),
        "actor_loss": float(actor_output.loss.item()),
        "grad_norm_before_clip": float(grad_norm),
        "flow_noise_reused": bool(
            torch.equal(old_output.flow_noise, current_output.flow_noise)
        ),
        "flow_time_reused": bool(
            torch.equal(old_output.flow_time, current_output.flow_time)
        ),
        "prefix_feature_shape": list(prefix_features.shape),
        "initial_value": float(value.item()),
        "value_grad_finite": value_grad_finite,
    }
    print(json.dumps(result, indent=2))

    if not result["current_after_step"] < result["current_before_step"]:
        raise RuntimeError("Positive-advantage FPO step did not reduce CFM loss.")


if __name__ == "__main__":
    main()
