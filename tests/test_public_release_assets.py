from __future__ import annotations

import unittest
from pathlib import Path

from absorbsr.stage_planner import load_schedule


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseAssetsTest(unittest.TestCase):
    def test_toy_example_is_explicitly_synthetic(self):
        schedule = load_schedule(ROOT / "examples" / "toy_schedule.json")
        self.assertEqual(schedule.backbone, "ToyTransformer")
        self.assertEqual(schedule.num_steps, 6)
        self.assertEqual(schedule.num_blocks, 4)
        self.assertEqual(schedule.block_calls, 15)

    def test_repository_has_release_metadata_and_no_paper_results(self):
        for relative in (
            "CITATION.cff",
            "LICENSE",
            "MODEL_CARD.md",
            "THIRD_PARTY_NOTICES.md",
            ".github/workflows/tests.yml",
            "examples/toy_sites.csv",
            "examples/toy_pairs.csv",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("synthetic", readme)
        self.assertFalse((ROOT / "assets" / "absorbsr_method_overview.png").exists())
        self.assertFalse((ROOT / "results").exists())


if __name__ == "__main__":
    unittest.main()
