"""Evaluate trained diffusion models on the MIMII DUE development test set."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from diffusion.config import load_config
from diffusion.engine import evaluate
from diffusion.gmm_scoring import evaluate_fan_gmm
from diffusion.sweep import evaluate_fan_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="diffusion.yaml")
    parser.add_argument("--machine-type", action="append", dest="machine_types")
    parser.add_argument("--max-files-per-group", type=int)
    parser.add_argument("--ddim-stride", type=int)
    parser.add_argument("--test-patch-hop", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--fan-sweep",
        action="store_true",
        help="Sweep DDIM start steps and AF settings on fan without retraining",
    )
    parser.add_argument(
        "--fan-gmm",
        action="store_true",
        help="Fit residual GMMs on normal fan audio without retraining",
    )
    parser.add_argument("--gmm-start-step", type=int, default=400)
    parser.add_argument("--gmm-patch-hop", type=int, default=32)
    parser.add_argument(
        "--gmm-residual-modes",
        nargs="+",
        choices=["signed", "absolute", "relu"],
        default=["signed", "absolute", "relu"],
    )
    parser.add_argument(
        "--gmm-scopes",
        nargs="+",
        choices=["global", "section"],
        default=["global", "section"],
    )
    parser.add_argument(
        "--gmm-covariances",
        nargs="+",
        choices=["full", "diag"],
        default=["full", "diag"],
    )
    parser.add_argument("--gmm-components", type=int, default=2)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-5)
    parser.add_argument(
        "--sweep-start-steps",
        type=int,
        nargs="+",
        default=[100, 200, 280, 400],
    )
    parser.add_argument(
        "--sweep-topk-ratios",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.1, 0.2, 1.0],
    )
    parser.add_argument(
        "--sweep-aggregations",
        nargs="+",
        choices=["mean", "max", "median", "topk_mean"],
        default=["mean", "max", "topk_mean"],
    )
    parser.add_argument(
        "--patch-topk-ratio",
        type=float,
        default=0.1,
        help="Fraction of highest patch scores used by topk_mean aggregation",
    )
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
        output_directory = Path(config["project"]["output_directory"])
        config["project"]["output_directory"] = str(output_directory / "smoke")
        max_files_per_group = max_files_per_group or 1
    if args.fan_sweep and args.fan_gmm:
        raise ValueError("--fan-sweep and --fan-gmm cannot be used together")
    if args.fan_sweep:
        if args.machine_types not in (None, ["fan"]):
            parser_value = ", ".join(args.machine_types)
            raise ValueError(
                f"--fan-sweep only supports --machine-type fan, got {parser_value}"
            )
        evaluate_fan_sweep(
            config,
            start_steps=args.sweep_start_steps,
            topk_ratios=args.sweep_topk_ratios,
            aggregations=args.sweep_aggregations,
            patch_topk_ratio=args.patch_topk_ratio,
            max_files_per_group=max_files_per_group,
        )
    elif args.fan_gmm:
        if args.machine_types not in (None, ["fan"]):
            parser_value = ", ".join(args.machine_types)
            raise ValueError(
                f"--fan-gmm only supports --machine-type fan, got {parser_value}"
            )
        evaluate_fan_gmm(
            config,
            start_step=args.gmm_start_step,
            patch_hop=args.gmm_patch_hop,
            residual_modes=args.gmm_residual_modes,
            scopes=args.gmm_scopes,
            covariance_types=args.gmm_covariances,
            components=args.gmm_components,
            reg_covar=args.gmm_reg_covar,
            max_test_files_per_group=max_files_per_group,
            max_train_files_per_section_domain=2 if args.smoke_test else None,
        )
    else:
        evaluate(
            config,
            machine_types=args.machine_types,
            max_files_per_group=max_files_per_group,
        )


if __name__ == "__main__":
    main()
