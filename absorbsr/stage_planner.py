"""Backbone-agnostic stage schedules for deterministic block omission."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, TypeAlias

BlockId: TypeAlias = int | str


@dataclass(frozen=True)
class StagePlan:
    """A half-open interval of denoising steps with one fixed skip set."""

    name: str
    start_step: int
    end_step: int
    skip_blocks: tuple[BlockId, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name cannot be empty")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError(
                f"invalid stage interval [{self.start_step}, {self.end_step})"
            )
        if len(set(self.skip_blocks)) != len(self.skip_blocks):
            raise ValueError(f"stage {self.name!r} contains duplicate skip blocks")

    def contains(self, step_index: int) -> bool:
        return self.start_step <= step_index < self.end_step

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "skip_blocks": list(self.skip_blocks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StagePlan":
        skip_blocks = payload.get("skip_blocks", [])
        if not isinstance(skip_blocks, list):
            raise TypeError("skip_blocks must be a JSON list")
        if not all(isinstance(block, (int, str)) for block in skip_blocks):
            raise TypeError("skip block identifiers must be integers or strings")
        return cls(
            name=str(payload["name"]),
            start_step=int(payload["start_step"]),
            end_step=int(payload["end_step"]),
            skip_blocks=tuple(skip_blocks),
        )


@dataclass(frozen=True)
class StageSchedule:
    """A deterministic depth schedule over all denoising steps."""

    method_id: str
    backbone: str
    num_steps: int
    num_blocks: int
    stages: tuple[StagePlan, ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.method_id or not self.backbone:
            raise ValueError("method_id and backbone cannot be empty")
        if self.num_steps <= 0 or self.num_blocks <= 0:
            raise ValueError("num_steps and num_blocks must be positive")
        if not self.stages:
            raise ValueError("a schedule must contain at least one stage")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")

        expected_start = 0
        identifier_kind: type[int] | type[str] | None = None
        for stage in self.stages:
            if stage.start_step != expected_start:
                raise ValueError(
                    "stages must be contiguous and ordered: "
                    f"expected start {expected_start}, got {stage.start_step}"
                )
            expected_start = stage.end_step
            for block in stage.skip_blocks:
                current_kind = int if isinstance(block, int) else str
                if identifier_kind is None:
                    identifier_kind = current_kind
                elif identifier_kind is not current_kind:
                    raise TypeError(
                        "one schedule cannot mix integer and string block identifiers"
                    )
                if isinstance(block, int) and not 0 <= block < self.num_blocks:
                    raise ValueError(
                        f"block index {block} is outside [0, {self.num_blocks})"
                    )
        if expected_start != self.num_steps:
            raise ValueError(
                f"stages end at step {expected_start}, expected {self.num_steps}"
            )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    @property
    def boundaries(self) -> tuple[int, ...]:
        return (0, *(stage.end_step for stage in self.stages))

    @property
    def full_block_calls(self) -> int:
        """Number of calls over the plannable blocks without omissions."""

        return self.num_steps * self.num_blocks

    @property
    def block_calls(self) -> int:
        """Number of calls over the plannable blocks after omissions."""

        skipped = sum(
            (stage.end_step - stage.start_step) * len(stage.skip_blocks)
            for stage in self.stages
        )
        return self.full_block_calls - skipped

    def stage_at(self, step_index: int) -> StagePlan:
        if not 0 <= step_index < self.num_steps:
            raise IndexError(
                f"step index {step_index} is outside [0, {self.num_steps})"
            )
        for stage in self.stages:
            if stage.contains(step_index):
                return stage
        raise RuntimeError(f"no stage covers step {step_index}")

    def skip_blocks_at(self, step_index: int) -> tuple[BlockId, ...]:
        return self.stage_at(step_index).skip_blocks

    def expand(self) -> tuple[tuple[BlockId, ...], ...]:
        """Expand the compact stage plan to one skip tuple per denoising step."""

        return tuple(self.skip_blocks_at(step) for step in range(self.num_steps))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "backbone": self.backbone,
            "num_steps": self.num_steps,
            "num_blocks": self.num_blocks,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StageSchedule":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"unsupported schedule schema version: {schema_version}")
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise TypeError("stages must be a JSON list")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise TypeError("metadata must be a JSON object")
        return cls(
            method_id=str(payload["method_id"]),
            backbone=str(payload["backbone"]),
            num_steps=int(payload["num_steps"]),
            num_blocks=int(payload["num_blocks"]),
            stages=tuple(StagePlan.from_dict(stage) for stage in raw_stages),
            metadata=raw_metadata,
        )


@dataclass(frozen=True)
class ExplicitSchedule:
    """A deterministic per-step block-omission schedule."""

    method_id: str
    backbone: str
    num_steps: int
    num_blocks: int
    skip_blocks_by_step: tuple[tuple[BlockId, ...], ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.method_id or not self.backbone:
            raise ValueError("method_id and backbone cannot be empty")
        if self.num_steps <= 0 or self.num_blocks <= 0:
            raise ValueError("num_steps and num_blocks must be positive")
        if len(self.skip_blocks_by_step) != self.num_steps:
            raise ValueError(
                "skip_blocks_by_step length must match num_steps: "
                f"{len(self.skip_blocks_by_step)} != {self.num_steps}"
            )
        identifier_kind: type[int] | type[str] | None = None
        for step, blocks in enumerate(self.skip_blocks_by_step):
            if len(set(blocks)) != len(blocks):
                raise ValueError(f"step {step} contains duplicate skip blocks")
            for block in blocks:
                current_kind = int if isinstance(block, int) else str
                if identifier_kind is None:
                    identifier_kind = current_kind
                elif identifier_kind is not current_kind:
                    raise TypeError(
                        "one schedule cannot mix integer and string block identifiers"
                    )
                if isinstance(block, int) and not 0 <= block < self.num_blocks:
                    raise ValueError(
                        f"block index {block} is outside [0, {self.num_blocks})"
                    )

    @property
    def full_block_calls(self) -> int:
        return self.num_steps * self.num_blocks

    @property
    def block_calls(self) -> int:
        return self.full_block_calls - sum(
            len(blocks) for blocks in self.skip_blocks_by_step
        )

    def skip_blocks_at(self, step_index: int) -> tuple[BlockId, ...]:
        if not 0 <= step_index < self.num_steps:
            raise IndexError(
                f"step index {step_index} is outside [0, {self.num_steps})"
            )
        return self.skip_blocks_by_step[step_index]

    def expand(self) -> tuple[tuple[BlockId, ...], ...]:
        return self.skip_blocks_by_step

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "schedule_type": "explicit",
            "method_id": self.method_id,
            "backbone": self.backbone,
            "num_steps": self.num_steps,
            "num_blocks": self.num_blocks,
            "skip_blocks_by_step": [
                list(blocks) for blocks in self.skip_blocks_by_step
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExplicitSchedule":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"unsupported schedule schema version: {schema_version}")
        raw_steps = payload.get("skip_blocks_by_step")
        if not isinstance(raw_steps, list):
            raise TypeError("skip_blocks_by_step must be a JSON list")
        parsed_steps: list[tuple[BlockId, ...]] = []
        for blocks in raw_steps:
            if not isinstance(blocks, list):
                raise TypeError("each per-step skip set must be a JSON list")
            if not all(isinstance(block, (int, str)) for block in blocks):
                raise TypeError("skip block identifiers must be integers or strings")
            parsed_steps.append(tuple(blocks))
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise TypeError("metadata must be a JSON object")
        return cls(
            method_id=str(payload["method_id"]),
            backbone=str(payload["backbone"]),
            num_steps=int(payload["num_steps"]),
            num_blocks=int(payload["num_blocks"]),
            skip_blocks_by_step=tuple(parsed_steps),
            metadata=raw_metadata,
        )


Schedule: TypeAlias = StageSchedule | ExplicitSchedule


@dataclass(frozen=True)
class OperatingPoint:
    """Measured quality and latency for one candidate execution schedule."""

    method_id: str
    wall_ms: float
    psnr: float
    lpips: float

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id cannot be empty")
        if not all(math.isfinite(value) for value in (self.wall_ms, self.psnr, self.lpips)):
            raise ValueError("operating-point metrics must be finite")
        if self.wall_ms <= 0:
            raise ValueError("wall_ms must be positive")


def equal_stage_boundaries(num_steps: int, num_stages: int) -> tuple[int, ...]:
    """Return integer boundaries using the protocol's floor-based partition."""

    if num_steps <= 0 or num_stages <= 0:
        raise ValueError("num_steps and num_stages must be positive")
    if num_stages > num_steps:
        raise ValueError("num_stages cannot exceed num_steps")
    return tuple(index * num_steps // num_stages for index in range(num_stages + 1))


def load_schedule(path: str | Path) -> Schedule:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("schedule file must contain a JSON object")
    if payload.get("schedule_type") == "explicit" or "skip_blocks_by_step" in payload:
        return ExplicitSchedule.from_dict(payload)
    return StageSchedule.from_dict(payload)


def save_schedule(schedule: Schedule, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(schedule.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
