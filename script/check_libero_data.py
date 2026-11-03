#!/usr/bin/env python3
"""Preflight-check a LIBERO JSONL dataset before launching DM05 training."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def is_finite_vector(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def image_path(image_dir: Path, image_meta: dict) -> Path:
    relative_url = image_meta["url"].lstrip("./")
    return image_dir / relative_url


def check_frame(frame: dict, image_dir: Path) -> tuple[int, int]:
    required = {"images_1", "images_2", "state", "action", "prompt"}
    missing = required - frame.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    state = frame["state"]
    action = frame["action"]
    if not is_finite_vector(state) or not is_finite_vector(action):
        raise ValueError("state/action contains NaN or Inf")
    if not isinstance(frame["prompt"], str) or not frame["prompt"].strip():
        raise ValueError("prompt is empty")

    for key in ("images_1", "images_2"):
        meta = frame[key]
        if meta.get("type") != "image":
            raise ValueError(f"{key} is not an image entry: {meta}")
        path = image_path(image_dir, meta)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()

    return len(state), len(action)


def main() -> None:
    args = parse_args()
    jsonl_files = sorted(args.jsonl_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL found under {args.jsonl_dir}")

    rng = random.Random(args.seed)
    file_line_counts: dict[Path, int] = {}
    suite_counts: Counter[str] = Counter()
    candidates: list[tuple[Path, int]] = []

    for path in jsonl_files:
        with path.open() as file:
            line_count = sum(1 for line in file if line.strip())
        file_line_counts[path] = line_count
        suite_counts[path.relative_to(args.jsonl_dir).parts[0]] += 1
        candidates.extend((path, line_index) for line_index in range(line_count))

    sample_count = min(args.samples, len(candidates))
    sampled = rng.sample(candidates, sample_count)
    dimensions: Counter[tuple[int, int]] = Counter()
    prompts: Counter[str] = Counter()

    for path, line_index in sampled:
        with path.open() as file:
            for current_index, line in enumerate(file):
                if current_index == line_index:
                    frame = json.loads(line)
                    break
            else:
                raise RuntimeError(f"Line {line_index} disappeared from {path}")
        dimensions[check_frame(frame, args.image_dir)] += 1
        prompts[frame["prompt"]] += 1

    total_frames = sum(file_line_counts.values())
    print(f"jsonl_dir={args.jsonl_dir}")
    print(f"image_dir={args.image_dir}")
    print(f"episodes={len(jsonl_files)}")
    print(f"frames={total_frames}")
    print(f"sampled_frames={sample_count}")
    print(f"suites={dict(sorted(suite_counts.items()))}")
    print(f"(state_dim, action_dim)={dict(dimensions)}")
    print(f"unique_prompts_in_sample={len(prompts)}")
    print("status=PASS")


if __name__ == "__main__":
    main()
