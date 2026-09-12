"""Train the MIMII DUE diffusion anomaly detector."""

from __future__ import annotations

import argparse
from copy import deepcopy

from diffusion.config import load_config
from diffusion.data_check import check_data_pipeline
from diffusion.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="diffusion.yaml")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--machine-type", action="append", dest="machine_types")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real training step with batch size 1 and save under checkpoints/smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.check_data:
        check_data_pipeline(
            config,
            machine_types=args.machine_types,
            max_files=args.max_files,
            build_cache=args.build_cache,
        )
        return
    if args.smoke_test:
        config = deepcopy(config)
        config["training"]["batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 1
        config["training"]["num_workers"] = 0
        config["training"]["epochs"] = 1
        config["project"]["checkpoint_directory"] = "./checkpoints/smoke"
        machine_types = args.machine_types or [config["data"]["machine_types"][0]]
        max_steps = args.max_steps or 1
    else:
        machine_types = args.machine_types
        max_steps = args.max_steps
    train(
        config,
        machine_types=machine_types,
        max_steps=max_steps,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
