"""Apply a frozen per-step schedule to an official DiT4SR pipeline."""

from __future__ import annotations

from typing import Any

from .stage_planner import ExplicitSchedule


class DiT4SRSkipAdapter:
    """Pass the configured block bypasses to DiT4SR on every tile call.

    DiT4SR exposes ``skip_layers`` on its transformer forward. For tiled
    inference, ``tiles_per_step`` must equal the number of transformer calls
    made for each denoising step.
    """

    def __init__(
        self,
        pipeline: Any,
        schedule: ExplicitSchedule,
        *,
        tiles_per_step: int = 1,
    ) -> None:
        if tiles_per_step <= 0:
            raise ValueError("tiles_per_step must be positive")
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None or not hasattr(transformer, "transformer_blocks"):
            raise TypeError("pipeline must expose transformer.transformer_blocks")
        if len(transformer.transformer_blocks) != schedule.num_blocks:
            raise ValueError(
                "schedule block count does not match DiT4SR transformer: "
                f"{schedule.num_blocks} != {len(transformer.transformer_blocks)}"
            )
        self.pipeline = pipeline
        self.schedule = schedule
        self.tiles_per_step = int(tiles_per_step)
        self.transformer = transformer
        self.call_count = 0
        self.execution_counts = [0] * schedule.num_steps
        self._original_forward = transformer.forward
        self._active = False

    def __enter__(self) -> "DiT4SRSkipAdapter":
        if self._active:
            raise RuntimeError("adapter is already active")
        self.transformer.forward = self._wrapped_forward
        self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def close(self) -> None:
        if self._active:
            self.transformer.forward = self._original_forward
            self._active = False

    def _wrapped_forward(self, *args, **kwargs):
        total_calls = self.schedule.num_steps * self.tiles_per_step
        if self.call_count >= total_calls:
            raise RuntimeError(
                f"received more than {total_calls} transformer calls for schedule"
            )
        step = self.call_count // self.tiles_per_step
        self.call_count += 1
        blocks = self.schedule.skip_blocks_at(step)
        if blocks:
            kwargs["skip_layers"] = list(blocks)
        self.execution_counts[step] += 1
        return self._original_forward(*args, **kwargs)

    def assert_complete(self) -> None:
        expected = self.schedule.num_steps * self.tiles_per_step
        if self.call_count != expected:
            raise RuntimeError(
                f"expected {expected} transformer calls, observed {self.call_count}"
            )
