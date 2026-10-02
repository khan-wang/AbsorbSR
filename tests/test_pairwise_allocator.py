from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from absorbsr import (
    ExplicitSchedule,
    PairObservation,
    fit_cancellation_kernel,
    fit_geometry_regularized_kernel,
    load_schedule,
    make_stage_slots,
    make_uniform_time_slots,
    solve_pairwise_schedule,
)
from absorbsr.pairwise_allocator import schedule_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


class PairwiseAllocatorTest(unittest.TestCase):
    def _toy_problem(self):
        risk = np.asarray(
            [
                [0.10, 0.13, 0.16, 0.20],
                [0.11, 0.14, 0.17, 0.21],
                [0.12, 0.15, 0.18, 0.22],
                [0.13, 0.16, 0.19, 0.23],
                [0.14, 0.17, 0.20, 0.24],
                [0.15, 0.18, 0.21, 0.25],
            ]
        )
        observations = []
        sites = [(step, block) for step in range(6) for block in range(4)]
        for index, (left, right) in enumerate(zip(sites, sites[5:] + sites[:5])):
            if index >= 18:
                break
            left_risk = risk[left]
            right_risk = risk[right]
            cancellation = 0.20 + 0.03 * (left[0] == right[0])
            observations.append(
                PairObservation(
                    left[0],
                    left[1],
                    right[0],
                    right[1],
                    (1.0 - cancellation) * (left_risk + right_risk),
                )
            )
        return risk, observations

    def test_stage_slots_preserve_each_budget(self):
        slots = make_stage_slots((0, 2, 4, 6), (3, 3, 3))
        self.assertEqual(len(slots), 9)
        self.assertEqual(sum(step < 2 for step in slots), 3)
        self.assertEqual(sum(2 <= step < 4 for step in slots), 3)
        self.assertEqual(sum(step >= 4 for step in slots), 3)

    def test_uniform_time_slots_distribute_extra_slots(self):
        slots = make_uniform_time_slots(6, 9)
        counts = np.bincount(slots, minlength=6)
        self.assertEqual(int(counts.sum()), 9)
        self.assertTrue(np.all(counts >= 1))
        self.assertEqual(counts.tolist(), [2, 1, 2, 1, 2, 1])

    def test_kernel_and_solver_return_a_valid_deterministic_schedule(self):
        risk, observations = self._toy_problem()
        kernel, fit_audit = fit_cancellation_kernel(
            risk,
            observations,
            stage_boundaries=(0, 2, 4, 6),
        )
        first, first_audit = solve_pairwise_schedule(
            risk,
            kernel,
            stage_boundaries=(0, 2, 4, 6),
            budget_per_stage=(2, 2, 2),
            restarts=2,
            iterations=120,
            seed=17,
        )
        second, second_audit = solve_pairwise_schedule(
            risk,
            kernel,
            stage_boundaries=(0, 2, 4, 6),
            budget_per_stage=(2, 2, 2),
            restarts=2,
            iterations=120,
            seed=17,
        )
        self.assertEqual(fit_audit["pair_count"], 18)
        self.assertEqual(first.expand(), second.expand())
        self.assertEqual(first_audit["schedule_sha256"], second_audit["schedule_sha256"])
        self.assertEqual(first.block_calls, 18)

    def test_geometry_interpolation_and_uniform_schedule_are_deterministic(self):
        risk, observations = self._toy_problem()
        kernel, audit = fit_geometry_regularized_kernel(
            risk,
            observations,
            stage_boundaries=(0, 2, 4, 6),
            lambda_value=0.5,
        )
        self.assertEqual(audit["full_descriptor"]["feature_indices"], list(range(9)))
        self.assertEqual(audit["geometry_descriptor"]["feature_indices"], [2, 3])
        first, first_audit = solve_pairwise_schedule(
            risk,
            kernel,
            stage_boundaries=(0, 2, 4, 6),
            omission_budget=9,
            uniform_over_time=True,
            restarts=2,
            iterations=120,
            seed=23,
        )
        second, second_audit = solve_pairwise_schedule(
            risk,
            kernel,
            stage_boundaries=(0, 2, 4, 6),
            omission_budget=9,
            uniform_over_time=True,
            restarts=2,
            iterations=120,
            seed=23,
        )
        counts = [len(step) for step in first.expand()]
        self.assertEqual(counts, [2, 1, 2, 1, 2, 1])
        self.assertEqual(first.expand(), second.expand())
        self.assertEqual(first_audit["schedule_sha256"], second_audit["schedule_sha256"])
        self.assertTrue(first_audit["uniform_over_time"])
        self.assertEqual(first.block_calls, 15)

    def test_uniform_allocation_rejects_budget_below_step_count(self):
        with self.assertRaisesRegex(ValueError, "budget >= num_steps"):
            make_uniform_time_slots(6, 5)

    def test_toy_schedule_config_round_trips(self):
        schedule = load_schedule(REPO_ROOT / "examples" / "toy_schedule.json")
        self.assertIsInstance(schedule, ExplicitSchedule)
        self.assertEqual(schedule.full_block_calls, 24)
        self.assertEqual(schedule.block_calls, 15)
        self.assertEqual(schedule.num_steps, 6)


if __name__ == "__main__":
    unittest.main()
