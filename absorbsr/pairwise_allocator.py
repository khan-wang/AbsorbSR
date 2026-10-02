"""Pairwise endpoint calibration and fixed-budget schedule allocation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stage_planner import ExplicitSchedule


@dataclass(frozen=True)
class PairObservation:
    """Measured endpoint response when two time-depth sites are omitted."""

    left_step: int
    left_block: int
    right_step: int
    right_block: int
    observed_response: float


@dataclass(frozen=True)
class CancellationKernel:
    """Ridge model for normalized pairwise cancellation."""

    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    clip_range: tuple[float, float]
    stage_boundaries: tuple[int, ...]
    feature_indices: tuple[int, ...] = tuple(range(9))

    def predict(self, feature: np.ndarray) -> float:
        selected = np.asarray(feature, dtype=float)[list(self.feature_indices)]
        standardized = (selected - self.feature_mean) / self.feature_scale
        value = float(np.r_[1.0, standardized] @ self.coefficients)
        return float(np.clip(value, *self.clip_range))


@dataclass(frozen=True)
class GeometryRegularizedKernel:
    """Interpolate geometry-only and full-descriptor cancellation fields."""

    full_kernel: CancellationKernel
    geometry_kernel: CancellationKernel
    lambda_value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_value <= 1.0:
            raise ValueError("lambda_value must lie in [0, 1]")
        if self.full_kernel.stage_boundaries != self.geometry_kernel.stage_boundaries:
            raise ValueError("full and geometry kernels use different stage boundaries")

    @property
    def stage_boundaries(self) -> tuple[int, ...]:
        return self.full_kernel.stage_boundaries

    def predict(self, feature: np.ndarray) -> float:
        geometry = self.geometry_kernel.predict(feature)
        full = self.full_kernel.predict(feature)
        return geometry + self.lambda_value * (full - geometry)


def _validate_boundaries(num_steps: int, boundaries: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in boundaries)
    if len(parsed) < 2 or parsed[0] != 0 or parsed[-1] != num_steps:
        raise ValueError("stage boundaries must start at 0 and end at num_steps")
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("stage boundaries must be strictly increasing")
    return parsed


def _stage_index(step: int, boundaries: Sequence[int]) -> int:
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= step < end:
            return index
    raise IndexError(f"step {step} is outside the stage boundaries")


def pair_features(
    left_step: int,
    left_block: int,
    left_risk: float,
    right_step: int,
    right_block: int,
    right_risk: float,
    *,
    num_steps: int,
    num_blocks: int,
    stage_boundaries: Sequence[int],
) -> np.ndarray:
    """Build the compact time-depth descriptor used by the kernel."""

    risk_sum = max(left_risk + right_risk, 1e-12)
    time_scale = max(num_steps - 1, 1)
    block_scale = max(num_blocks - 1, 1)
    return np.asarray(
        [
            math.log(risk_sum),
            abs(left_risk - right_risk) / risk_sum,
            abs(left_step - right_step) / time_scale,
            abs(left_block - right_block) / block_scale,
            0.5 * (left_step + right_step) / time_scale,
            0.5 * (left_block + right_block) / block_scale,
            float(left_step == right_step),
            float(left_block == right_block),
            float(
                _stage_index(left_step, stage_boundaries)
                == _stage_index(right_step, stage_boundaries)
            ),
        ],
        dtype=float,
    )


def fit_cancellation_kernel(
    site_risk: np.ndarray,
    observations: Sequence[PairObservation],
    *,
    stage_boundaries: Sequence[int],
    ridge_alpha: float = 1.0,
    clip_quantiles: tuple[float, float] = (0.05, 0.95),
    feature_indices: Sequence[int] | None = None,
) -> tuple[CancellationKernel, dict[str, object]]:
    """Fit cancellation from singleton risks and measured pair responses."""

    risk = np.asarray(site_risk, dtype=float)
    if risk.ndim != 2 or not np.isfinite(risk).all() or np.any(risk < 0):
        raise ValueError("site_risk must be a finite nonnegative 2-D array")
    if not observations:
        raise ValueError("at least one pair observation is required")
    num_steps, num_blocks = risk.shape
    boundaries = _validate_boundaries(num_steps, stage_boundaries)
    features: list[np.ndarray] = []
    cancellation: list[float] = []
    for row in observations:
        for step, block in (
            (row.left_step, row.left_block),
            (row.right_step, row.right_block),
        ):
            if not 0 <= step < num_steps or not 0 <= block < num_blocks:
                raise IndexError(f"pair site ({step}, {block}) is outside the risk grid")
        left_risk = risk[row.left_step, row.left_block]
        right_risk = risk[row.right_step, row.right_block]
        additive = max(left_risk + right_risk, 1e-12)
        features.append(
            pair_features(
                row.left_step,
                row.left_block,
                left_risk,
                row.right_step,
                row.right_block,
                right_risk,
                num_steps=num_steps,
                num_blocks=num_blocks,
                stage_boundaries=boundaries,
            )
        )
        cancellation.append((additive - row.observed_response) / additive)

    all_features = np.asarray(features)
    selected_indices = (
        tuple(range(all_features.shape[1]))
        if feature_indices is None
        else tuple(int(index) for index in feature_indices)
    )
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("feature_indices must be a nonempty sequence without duplicates")
    if any(index < 0 or index >= all_features.shape[1] for index in selected_indices):
        raise ValueError("feature_indices contains an out-of-range descriptor index")
    x = all_features[:, selected_indices]
    y = np.asarray(cancellation)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(x.shape[0]), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    predicted = design @ coefficients
    clip = tuple(float(value) for value in np.quantile(y, clip_quantiles))
    kernel = CancellationKernel(
        coefficients, mean, scale, clip, boundaries, selected_indices
    )
    audit: dict[str, object] = {
        "pair_count": len(observations),
        "ridge_alpha": ridge_alpha,
        "cancellation_mean": float(y.mean()),
        "cancellation_median": float(np.median(y)),
        "cancellation_clip": list(clip),
        "fit_rmse": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "feature_indices": list(selected_indices),
    }
    return kernel, audit


def fit_geometry_regularized_kernel(
    site_risk: np.ndarray,
    observations: Sequence[PairObservation],
    *,
    stage_boundaries: Sequence[int],
    lambda_value: float,
    ridge_alpha: float = 1.0,
    clip_quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[GeometryRegularizedKernel, dict[str, object]]:
    """Fit the full and time-depth-only fields used by the paper interpolation.

    The full nine-dimensional descriptor is interpolated with a lower-capacity
    kernel using only normalized time and block separations (descriptor indices
    2 and 3).
    """

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must lie in [0, 1]")
    full_kernel, full_audit = fit_cancellation_kernel(
        site_risk,
        observations,
        stage_boundaries=stage_boundaries,
        ridge_alpha=ridge_alpha,
        clip_quantiles=clip_quantiles,
    )
    geometry_kernel, geometry_audit = fit_cancellation_kernel(
        site_risk,
        observations,
        stage_boundaries=stage_boundaries,
        ridge_alpha=ridge_alpha,
        clip_quantiles=clip_quantiles,
        feature_indices=(2, 3),
    )
    kernel = GeometryRegularizedKernel(
        full_kernel, geometry_kernel, float(lambda_value)
    )
    return kernel, {
        "lambda": float(lambda_value),
        "full_descriptor": full_audit,
        "geometry_descriptor": geometry_audit,
    }


def make_stage_slots(
    stage_boundaries: Sequence[int],
    budget_per_stage: Sequence[int],
) -> list[int]:
    """Spread each stage's omission budget over its denoising steps."""

    boundaries = tuple(int(value) for value in stage_boundaries)
    budgets = tuple(int(value) for value in budget_per_stage)
    if len(boundaries) != len(budgets) + 1:
        raise ValueError("one omission budget is required per stage")
    output: list[int] = []
    for start, end, budget in zip(boundaries, boundaries[1:], budgets):
        step_count = end - start
        if step_count <= 0 or budget < 0:
            raise ValueError("stages must be nonempty and budgets nonnegative")
        base, remainder = divmod(budget, step_count)
        counts = [base] * step_count
        if remainder:
            indices = [
                round((index + 0.5) * step_count / remainder - 0.5)
                for index in range(remainder)
            ]
            for index in indices:
                counts[min(index, step_count - 1)] += 1
        for step, count in zip(range(start, end), counts):
            output.extend([step] * count)
    return output


def make_uniform_time_slots(num_steps: int, omission_budget: int) -> list[int]:
    """Give every step one omission, then spread remaining slots uniformly."""

    if num_steps <= 0 or omission_budget < num_steps:
        raise ValueError("uniform-time allocation requires budget >= num_steps > 0")
    base, remainder = divmod(omission_budget, num_steps)
    counts = [base] * num_steps
    if remainder:
        indices = [
            round((index + 0.5) * num_steps / remainder - 0.5)
            for index in range(remainder)
        ]
        for index in indices:
            counts[min(index, num_steps - 1)] += 1
    return [step for step, count in enumerate(counts) for _ in range(count)]


def _schedule_from_layers(
    slot_steps: Sequence[int],
    layers: np.ndarray,
    num_steps: int,
) -> tuple[tuple[int, ...], ...]:
    schedule: list[list[int]] = [[] for _ in range(num_steps)]
    for step, layer in zip(slot_steps, layers.tolist()):
        schedule[step].append(int(layer))
    for blocks in schedule:
        blocks.sort()
        if len(blocks) != len(set(blocks)):
            raise RuntimeError("one step contains a duplicate omitted block")
    return tuple(tuple(blocks) for blocks in schedule)


def schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    import json

    payload = json.dumps(schedule, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _objective(
    slot_steps: Sequence[int],
    layers: np.ndarray,
    unary: np.ndarray,
    pair_risk: np.ndarray,
) -> float:
    count = len(slot_steps)
    unary_mean = float(
        np.mean([unary[step, layer] for step, layer in zip(slot_steps, layers)])
    )
    pair_sum = 0.0
    for left in range(count):
        for right in range(left + 1, count):
            pair_sum += pair_risk[left, right, layers[left], layers[right]]
    pair_mean = pair_sum / max(count * (count - 1) / 2.0, 1.0)
    return pair_mean + 1e-6 * unary_mean


def solve_pairwise_schedule(
    site_risk: np.ndarray,
    kernel: CancellationKernel | GeometryRegularizedKernel,
    *,
    stage_boundaries: Sequence[int],
    budget_per_stage: Sequence[int] | None = None,
    omission_budget: int | None = None,
    uniform_over_time: bool = False,
    method_id: str = "pairwise_subadditive_v1",
    backbone: str = "DiT4SR",
    restarts: int = 12,
    iterations: int = 8000,
    seed: int = 0,
    max_per_block: int | None = None,
) -> tuple[ExplicitSchedule, dict[str, object]]:
    """Solve the fixed-budget pairwise allocation by multi-start annealing."""

    risk = np.asarray(site_risk, dtype=float)
    if risk.ndim != 2 or not np.isfinite(risk).all():
        raise ValueError("site_risk must be a finite 2-D array")
    if restarts <= 0 or iterations <= 0:
        raise ValueError("restarts and iterations must be positive")
    num_steps, num_blocks = risk.shape
    boundaries = _validate_boundaries(num_steps, stage_boundaries)
    if tuple(kernel.stage_boundaries) != boundaries:
        raise ValueError("kernel and solver stage boundaries differ")
    if uniform_over_time:
        if omission_budget is None:
            raise ValueError("omission_budget is required for uniform-time allocation")
        slots = make_uniform_time_slots(num_steps, int(omission_budget))
    else:
        if budget_per_stage is None:
            raise ValueError("budget_per_stage is required for stage-constrained allocation")
        slots = make_stage_slots(boundaries, budget_per_stage)
    if not slots:
        raise ValueError("the omission budget must be positive")
    per_step_counts = np.bincount(slots, minlength=num_steps)
    if np.any(per_step_counts > num_blocks):
        raise ValueError("a step's omission budget exceeds num_blocks")

    pair_risk = np.zeros(
        (len(slots), len(slots), num_blocks, num_blocks),
        dtype=np.float32,
    )
    for left in range(len(slots)):
        left_step = slots[left]
        for right in range(left + 1, len(slots)):
            right_step = slots[right]
            for left_block in range(num_blocks):
                for right_block in range(num_blocks):
                    left_risk = risk[left_step, left_block]
                    right_risk = risk[right_step, right_block]
                    cancellation = kernel.predict(
                        pair_features(
                            left_step,
                            left_block,
                            left_risk,
                            right_step,
                            right_block,
                            right_risk,
                            num_steps=num_steps,
                            num_blocks=num_blocks,
                            stage_boundaries=boundaries,
                        )
                    )
                    pair_risk[left, right, left_block, right_block] = (
                        1.0 - cancellation
                    ) * (left_risk + right_risk)

    candidates: list[tuple[float, int, np.ndarray]] = []
    cap = 0 if max_per_block is None else int(max_per_block)
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        for _ in range(100):
            layers = np.empty(len(slots), dtype=int)
            global_counts = np.zeros(num_blocks, dtype=int)
            valid = True
            for index, step in enumerate(slots):
                used = {
                    int(layers[other])
                    for other in range(index)
                    if slots[other] == step
                }
                choices = [
                    block
                    for block in range(num_blocks)
                    if block not in used and (cap <= 0 or global_counts[block] < cap)
                ]
                if not choices:
                    valid = False
                    break
                chosen = int(rng.choice(choices))
                layers[index] = chosen
                global_counts[chosen] += 1
            if valid:
                break
        else:
            raise RuntimeError("could not initialize a feasible schedule")

        current = _objective(slots, layers, risk, pair_risk)
        best_layers = layers.copy()
        best = current
        temperature = max(current * 0.03, 1e-8)
        for _ in range(iterations):
            index = int(rng.integers(len(slots)))
            step = slots[index]
            used = {
                int(layers[other])
                for other in range(len(slots))
                if other != index and slots[other] == step
            }
            global_counts = np.bincount(layers, minlength=num_blocks)
            choices = [
                block
                for block in range(num_blocks)
                if block not in used
                and block != layers[index]
                and (cap <= 0 or global_counts[block] < cap)
            ]
            if not choices:
                continue
            old = int(layers[index])
            layers[index] = int(rng.choice(choices))
            candidate = _objective(slots, layers, risk, pair_risk)
            accept = candidate <= current or rng.random() < math.exp(
                (current - candidate) / max(temperature, 1e-12)
            )
            if accept:
                current = candidate
                if current < best:
                    best = current
                    best_layers = layers.copy()
            else:
                layers[index] = old
            temperature = max(temperature * 0.9997, 1e-10)
        candidates.append((best, restart, best_layers))

    objective, winner_restart, winner_layers = min(candidates, key=lambda row: row[0])
    per_step = _schedule_from_layers(slots, winner_layers, num_steps)
    digest = schedule_sha256(per_step)
    schedule = ExplicitSchedule(
        method_id=method_id,
        backbone=backbone,
        num_steps=num_steps,
        num_blocks=num_blocks,
        skip_blocks_by_step=per_step,
        metadata={
            "stage_boundaries": list(boundaries),
            "budget_per_stage": (
                [int(value) for value in budget_per_stage]
                if budget_per_stage is not None and not uniform_over_time
                else None
            ),
            "uniform_over_time": bool(uniform_over_time),
            "omission_budget": len(slots),
            "schedule_sha256": digest,
        },
    )
    audit: dict[str, object] = {
        "objective": float(objective),
        "winner_restart": winner_restart,
        "restarts": restarts,
        "iterations_per_restart": iterations,
        "uniform_over_time": bool(uniform_over_time),
        "omission_budget": len(slots),
        "schedule_sha256": digest,
    }
    return schedule, audit
