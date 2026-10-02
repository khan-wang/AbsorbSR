#!/usr/bin/env python3
"""Build a deterministic AbsorbSR schedule from endpoint calibration CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from absorbsr import (
    PairObservation,
    fit_geometry_regularized_kernel,
    save_schedule,
    solve_pairwise_schedule,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--stage-boundaries", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method-id", default="")
    parser.add_argument("--backbone", required=True)
    args = parser.parse_args()

    site_risk = np.full((args.num_steps, args.num_blocks), np.nan, dtype=float)
    for row in _read_csv(args.sites):
        site_risk[int(row["step"]), int(row["layer"])] = float(row["endpoint_l1"])
    if np.isnan(site_risk).any():
        raise RuntimeError("site CSV does not cover the complete time-depth grid")

    observations = [
        PairObservation(
            left_step=int(row["left_step"]),
            left_block=int(row["left_layer"]),
            right_step=int(row["right_step"]),
            right_block=int(row["right_layer"]),
            observed_response=float(row["observed_endpoint_l1"]),
        )
        for row in _read_csv(args.pairs)
    ]
    boundaries = _parse_ints(args.stage_boundaries)
    if not 0.0 <= args.lambda_value <= 1.0:
        raise ValueError("--lambda-value must be in [0, 1]")
    method_id = args.method_id or f"pairwise_subadditive_lambda_{round(args.lambda_value * 100):03d}"
    kernel, fit_audit = fit_geometry_regularized_kernel(
        site_risk,
        observations,
        stage_boundaries=boundaries,
        lambda_value=args.lambda_value,
    )
    schedule, solve_audit = solve_pairwise_schedule(
        site_risk,
        kernel,
        stage_boundaries=boundaries,
        omission_budget=args.budget,
        uniform_over_time=True,
        method_id=method_id,
        backbone=args.backbone,
        restarts=args.restarts,
        iterations=args.iterations,
        seed=args.seed,
    )
    save_schedule(schedule, args.output)
    audit = {
        "selection_data": (
            "endpoint singleton and pair interventions; ground truth is not used"
        ),
        "kernel": fit_audit,
        "solver": solve_audit,
        "lambda": args.lambda_value,
        "block_calls": schedule.block_calls,
    }
    audit_path = args.audit_output or args.output.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
