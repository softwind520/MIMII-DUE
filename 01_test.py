"""Evaluate trained diffusion models on the MIMII DUE development test set."""

from __future__ import annotations

import argparse
from copy import deepcopy

from diffusion.config import load_config
from diffusion.engine import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="diffusion.yaml")
    parser.add_argument("--machine-type", action="append", dest="machine_types")
    parser.add_argument("--max-files-per-group", type=int)
    parser.add_argument("--ddim-stride", type=int)
    parser.add_argument("--test-patch-hop", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate one file per label/section/domain with sparse patches and five DDIM steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.ddim_stride is not None:
        config["diffusion"]["ddim_stride"] = args.ddim_stride
    if args.test_patch_hop is not None:
        config["data"]["test_patch_hop"] = args.test_patch_hop
    if args.batch_size is not None:
        config["evaluation"]["batch_size"] = args.batch_size
    max_files_per_group = args.max_files_per_group
    if args.smoke_test:
        config = deepcopy(config)
        config["data"]["test_patch_hop"] = 128
        config["diffusion"]["ddim_stride"] = 70
        config["evaluation"]["batch_size"] = 8
        config["evaluation"]["num_workers"] = 0
        config["project"]["output_directory"] = "./outputs/smoke"
        max_files_per_group = max_files_per_group or 1
    evaluate(
        config,
        machine_types=args.machine_types,
        max_files_per_group=max_files_per_group,
    )


if __name__ == "__main__":
    main()
