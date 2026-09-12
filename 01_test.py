"""Evaluate the MIMII DUE diffusion anomaly detector."""

from __future__ import annotations

import argparse

from diffusion.config import load_config
from diffusion.engine import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="diffusion.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
