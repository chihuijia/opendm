#!/usr/bin/env python3
"""Run one reproducible three-round DM05-FPO seed on LIBERO task 8."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_SFT_CHECKPOINT = (
    "/workspace/opendm-zxy/checkpoints/"
    "dm05_libero_lora_stage_1k_20261016/checkpoint-10000"
)
DEFAULT_NORM_STATS = (
    "/workspace/opendm-zxy/data/norm_stats/"
    "libero_pi0_all_3577e2559f079b3e.json"
)
DEFAULT_OUTPUT_ROOT = (
    "/workspace/opendm-zxy/p7_3seed"
)
ROUND_CONFIGS = (
    {
        "actor_learning_rate": 3e-7,
        "value_learning_rate": 1e-5,
        "max_actor_chunks": 4,
    },
    {
        "actor_learning_rate": 1e-7,
        "value_learning_rate": 1e-5,
        "max_actor_chunks": 4,
    },
    {
        "actor_learning_rate": 3e-8,
        "value_learning_rate": 1e-5,
        "max_actor_chunks": 8,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sft-checkpoint", default=DEFAULT_SFT_CHECKPOINT)
    parser.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--opendm-root", default="/workspace/opendm")
    parser.add_argument("--sim-root", default="/workspace/opendm-sim")
    parser.add_argument(
        "--opendm-python", default="/workspace/opendm_env/bin/python"
    )
    parser.add_argument(
        "--libero-python", default="/workspace/envs/libero_env/bin/python"
    )
    parser.add_argument("--eval-trials", type=int, default=50)
    parser.add_argument("--collect-trials", type=int, default=5)
    parser.add_argument("--health-timeout-s", type=int, default=180)
    return parser.parse_args()


def checkpoint_ready(path: Path) -> bool:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "norm_stats.json",
    )
    return all((path / name).is_file() for name in required)


def evaluation_ready(path: Path, expected_episodes: int) -> bool:
    result_path = path / "results.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return int(result.get("total_episodes", -1)) == expected_episodes


def collection_ready(path: Path, expected_episodes: int) -> bool:
    if not evaluation_ready(path, expected_episodes):
        return False
    rollout_files = sorted((path / "rollouts").glob("*.npz"))
    return len(rollout_files) == expected_episodes


def wait_for_health(port: int, timeout_s: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Inference server exited early with code {process.returncode}."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Inference server did not become healthy: {url}")


def start_server(
    args: argparse.Namespace,
    checkpoint: Path,
    log_path: Path,
) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        "bash",
        "script/dm05_launcher.sh",
        "--exp",
        "playground/dm05_libero_lora.py",
        "--task",
        "inference",
        "--model-config.model-name-or-path",
        str(checkpoint),
        "--model-config.chunk-size",
        "10",
        "--model-config.llm-attn-implementation",
        "sdpa",
        "--model-config.vision-attn-implementation",
        "sdpa",
        "--model-config.action-attn-implementation",
        "sdpa",
        "--inference-config.output-action-dim",
        "7",
        "--inference-config.image-prompts",
        "Head",
        "Left wrist",
        "--inference-config.port",
        str(args.port),
    ]
    env = os.environ.copy()
    env["PATH"] = f"{Path(args.opendm_python).parent}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    process = subprocess.Popen(
        command,
        cwd=args.opendm_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_health(args.port, args.health_timeout_s, process)
    except Exception:
        stop_server(process, log_handle)
        raise
    return process, log_handle


def stop_server(process: subprocess.Popen, log_handle: object) -> None:
    try:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    finally:
        log_handle.close()


def run_evaluation(
    args: argparse.Namespace,
    output_dir: Path,
    trials: int,
    sampling_seed_base: int,
    save_rollout: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.with_suffix(".log")
    command = [
        args.libero_python,
        "evaluation/run_libero_evaluation.py",
        "--config",
        "evaluation/configs/libero/example_dm05_libero.yaml",
        "--set",
        "benchmark",
        "libero_spatial",
        "--set",
        "task_ids",
        "8",
        "--set",
        "num_trails_per_task",
        str(trials),
        "--set",
        "base_url",
        f"http://localhost:{args.port}",
        "--set",
        "api_style",
        "v1",
        "--set",
        "sampling_seed_base",
        str(sampling_seed_base),
        "--set",
        "save_rollout_npz",
        str(save_rollout).lower(),
        "--set",
        "output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    libero_lib = str(Path(args.libero_python).parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = (
        f"{libero_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            cwd=args.sim_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_update(
    args: argparse.Namespace,
    checkpoint: Path,
    rollout_dir: Path,
    output_dir: Path,
    round_index: int,
) -> None:
    config = ROUND_CONFIGS[round_index - 1]
    command = [
        args.opendm_python,
        "script/check_dm05_fpo_rollout_update.py",
        "--checkpoint",
        str(checkpoint),
        "--rollout",
        str(rollout_dir),
        "--norm-stats",
        args.norm_stats,
        "--max-actor-chunks",
        str(config["max_actor_chunks"]),
        "--actor-micro-batch-size",
        "4",
        "--n-cfm-samples",
        "4",
        "--value-learning-rate",
        str(config["value_learning_rate"]),
        "--actor-learning-rate",
        str(config["actor_learning_rate"]),
        "--target-kl",
        "0.1",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = f"{args.opendm_root}:{args.sim_root}"
    log_path = output_dir.with_suffix(".log")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            cwd=args.opendm_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def protocol_payload(args: argparse.Namespace) -> dict:
    seed_offset = (args.seed - 7) * 100
    return {
        "seed": args.seed,
        "gpu": args.gpu,
        "port": args.port,
        "task_id": 8,
        "sft_checkpoint": args.sft_checkpoint,
        "norm_stats": args.norm_stats,
        "environment_seed": 7,
        "eval_trials": args.eval_trials,
        "collect_trials": args.collect_trials,
        "eval_sampling_seed_base": 31 + seed_offset,
        "collect_sampling_seed_bases": [
            31 + seed_offset,
            41 + seed_offset,
            51 + seed_offset,
        ],
        "round_configs": [dict(config) for config in ROUND_CONFIGS],
        "n_cfm_samples": 4,
        "actor_micro_batch_size": 4,
        "target_kl": 0.1,
    }


def main() -> None:
    args = parse_args()
    if args.eval_trials <= 0 or args.collect_trials <= 0:
        raise ValueError("Trial counts must be positive.")
    if args.seed < 7:
        raise ValueError("Seed must be at least 7 for the registered protocol.")
    root = Path(args.output_root) / f"task8_seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args)
    protocol_path = root / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            completed = [
                root / f"checkpoint_round{round_index}"
                for round_index in range(1, 4)
                if checkpoint_ready(root / f"checkpoint_round{round_index}")
            ]
            if completed:
                raise RuntimeError(
                    "Existing protocol differs after checkpoints were created: "
                    f"{protocol_path}"
                )
            protocol_path.write_text(
                json.dumps(protocol, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2) + "\n",
            encoding="utf-8",
        )

    checkpoint = Path(args.sft_checkpoint)
    if not checkpoint_ready(checkpoint):
        raise FileNotFoundError(f"Incomplete SFT checkpoint: {checkpoint}")

    baseline_dir = root / "baseline_eval"
    first_collection = root / "round1_collect"
    if (
        not evaluation_ready(baseline_dir, args.eval_trials)
        or not collection_ready(first_collection, args.collect_trials)
    ):
        process, log_handle = start_server(
            args,
            checkpoint,
            root / "sft_server.log",
        )
        try:
            if not evaluation_ready(baseline_dir, args.eval_trials):
                run_evaluation(
                    args,
                    baseline_dir,
                    args.eval_trials,
                    protocol["eval_sampling_seed_base"],
                    save_rollout=False,
                )
            if not collection_ready(first_collection, args.collect_trials):
                run_evaluation(
                    args,
                    first_collection,
                    args.collect_trials,
                    protocol["collect_sampling_seed_bases"][0],
                    save_rollout=True,
                )
        finally:
            stop_server(process, log_handle)

    for round_index in range(1, 4):
        collection_dir = root / f"round{round_index}_collect"
        output_checkpoint = root / f"checkpoint_round{round_index}"
        if not checkpoint_ready(output_checkpoint):
            if not collection_ready(collection_dir, args.collect_trials):
                raise RuntimeError(f"Incomplete rollout collection: {collection_dir}")
            run_update(
                args,
                checkpoint,
                collection_dir / "rollouts",
                output_checkpoint,
                round_index,
            )
        checkpoint = output_checkpoint
        if round_index == 3:
            break

        next_collection = root / f"round{round_index + 1}_collect"
        if not collection_ready(next_collection, args.collect_trials):
            process, log_handle = start_server(
                args,
                checkpoint,
                root / f"round{round_index}_server.log",
            )
            try:
                run_evaluation(
                    args,
                    next_collection,
                    args.collect_trials,
                    protocol["collect_sampling_seed_bases"][round_index],
                    save_rollout=True,
                )
            finally:
                stop_server(process, log_handle)

    final_eval = root / "final_eval"
    if not evaluation_ready(final_eval, args.eval_trials):
        process, log_handle = start_server(
            args,
            checkpoint,
            root / "final_server.log",
        )
        try:
            run_evaluation(
                args,
                final_eval,
                args.eval_trials,
                protocol["eval_sampling_seed_base"],
                save_rollout=False,
            )
        finally:
            stop_server(process, log_handle)

    summary = {
        "seed": args.seed,
        "baseline": json.loads(
            (baseline_dir / "results.json").read_text(encoding="utf-8")
        ),
        "final": json.loads(
            (final_eval / "results.json").read_text(encoding="utf-8")
        ),
        "final_checkpoint": str(checkpoint),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "baseline_success_rate": summary["baseline"]["success_rate"],
                "final_success_rate": summary["final"]["success_rate"],
                "final_checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
