#!/usr/bin/env python3
"""Summarize paired SFT-versus-FPO LIBERO evaluations across seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/workspace/opendm-zxy/p7_3seed",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    return parser.parse_args()


def load_episode_results(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_results = payload["task_results"]
    if len(task_results) != 1:
        raise ValueError(f"Expected one task in {path}.")
    return task_results[0]["episode_results"]


def mean_success_steps(episodes: list[dict]) -> float | None:
    steps = [int(item["steps"]) for item in episodes if item["success"]]
    return statistics.mean(steps) if steps else None


def exact_mcnemar_p(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    lower = min(gained, lost)
    one_tail = sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * one_tail)


def summarize_seed(root: Path, seed: int) -> dict:
    seed_root = root / f"task8_seed_{seed}"
    baseline = load_episode_results(seed_root / "baseline_eval/results.json")
    final = load_episode_results(seed_root / "final_eval/results.json")
    if len(baseline) != len(final):
        raise ValueError(f"Paired episode count mismatch for seed {seed}.")
    if [item["episode"] for item in baseline] != [
        item["episode"] for item in final
    ]:
        raise ValueError(f"Paired episode ids mismatch for seed {seed}.")

    gained = 0
    lost = 0
    retained = 0
    both_failed = 0
    paired_step_deltas = []
    for before, after in zip(baseline, final, strict=True):
        before_success = bool(before["success"])
        after_success = bool(after["success"])
        if before_success and after_success:
            retained += 1
            paired_step_deltas.append(int(after["steps"]) - int(before["steps"]))
        elif before_success:
            lost += 1
        elif after_success:
            gained += 1
        else:
            both_failed += 1

    baseline_successes = retained + lost
    final_successes = retained + gained
    round_metrics = []
    for round_index in range(1, 4):
        metrics_path = (
            seed_root
            / f"checkpoint_round{round_index}"
            / "fpo_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        round_metrics.append(
            {
                "round": round_index,
                "rollout_successes": metrics["successful_rollouts"],
                "rollout_count": metrics["rollout_count"],
                "value_loss_before": metrics["value_loss_before"],
                "value_loss_after": metrics["value_loss_after"],
                "actor_grad_norm": metrics["actor_grad_norm"],
                "clip_fraction_after": metrics["clip_fraction_after"],
                "approx_kl_after": metrics["approx_kl_after"],
            }
        )

    episode_count = len(baseline)
    return {
        "seed": seed,
        "episodes": episode_count,
        "baseline_successes": baseline_successes,
        "final_successes": final_successes,
        "baseline_success_rate": baseline_successes / episode_count,
        "final_success_rate": final_successes / episode_count,
        "success_rate_delta": (
            final_successes - baseline_successes
        ) / episode_count,
        "gained": gained,
        "lost": lost,
        "retained": retained,
        "both_failed": both_failed,
        "baseline_success_steps_mean": mean_success_steps(baseline),
        "final_success_steps_mean": mean_success_steps(final),
        "both_success_step_delta_mean": (
            statistics.mean(paired_step_deltas)
            if paired_step_deltas
            else None
        ),
        "round_metrics": round_metrics,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    per_seed = [summarize_seed(root, seed) for seed in args.seeds]
    total_episodes = sum(item["episodes"] for item in per_seed)
    baseline_successes = sum(item["baseline_successes"] for item in per_seed)
    final_successes = sum(item["final_successes"] for item in per_seed)
    gained = sum(item["gained"] for item in per_seed)
    lost = sum(item["lost"] for item in per_seed)
    seed_deltas = [item["success_rate_delta"] for item in per_seed]
    summary = {
        "task_id": 8,
        "seeds": list(args.seeds),
        "per_seed": per_seed,
        "aggregate": {
            "episodes": total_episodes,
            "baseline_successes": baseline_successes,
            "final_successes": final_successes,
            "baseline_success_rate": baseline_successes / total_episodes,
            "final_success_rate": final_successes / total_episodes,
            "success_rate_delta": (
                final_successes - baseline_successes
            ) / total_episodes,
            "gained": gained,
            "lost": lost,
            "retained": sum(item["retained"] for item in per_seed),
            "both_failed": sum(item["both_failed"] for item in per_seed),
            "seed_delta_mean": statistics.mean(seed_deltas),
            "seed_delta_std": (
                statistics.stdev(seed_deltas) if len(seed_deltas) > 1 else 0.0
            ),
            "paired_mcnemar_exact_p": exact_mcnemar_p(gained, lost),
        },
    }
    output_path = root / "aggregate_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
