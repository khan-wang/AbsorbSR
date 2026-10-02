from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from absorbsr.stage_planner import (
    ExplicitSchedule,
    StagePlan,
    StageSchedule,
    equal_stage_boundaries,
    load_schedule,
    save_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StageScheduleTest(unittest.TestCase):
    def test_equal_stage_boundaries_cover_steps_without_gaps(self):
        self.assertEqual(equal_stage_boundaries(6, 3), (0, 2, 4, 6))

    def test_json_round_trip(self):
        schedule = StageSchedule(
            method_id="toy",
            backbone="ToyTransformer",
            num_steps=4,
            num_blocks=3,
            stages=(
                StagePlan("early", 0, 2, (0,)),
                StagePlan("late", 2, 4, ()),
            ),
            metadata={"source": "synthetic-example"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            save_schedule(schedule, path)
            restored = load_schedule(path)
        self.assertEqual(restored, schedule)
        self.assertEqual(restored.metadata, {"source": "synthetic-example"})

    def test_example_explicit_schedule_has_a_valid_call_count(self):
        schedule = load_schedule(REPO_ROOT / "examples" / "toy_schedule.json")
        self.assertIsInstance(schedule, ExplicitSchedule)
        self.assertEqual(schedule.num_steps, 6)
        self.assertEqual(schedule.num_blocks, 4)
        self.assertEqual(schedule.block_calls, 15)

    def test_rejects_noncontiguous_stages(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            StageSchedule(
                method_id="bad",
                backbone="ToyTransformer",
                num_steps=4,
                num_blocks=3,
                stages=(
                    StagePlan("early", 0, 2, (0,)),
                    StagePlan("late", 3, 4, ()),
                ),
            )


if __name__ == "__main__":
    unittest.main()
