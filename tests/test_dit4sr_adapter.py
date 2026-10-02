from __future__ import annotations

import unittest
from types import SimpleNamespace

from absorbsr import DiT4SRSkipAdapter, ExplicitSchedule


class _Transformer:
    def __init__(self):
        self.transformer_blocks = [object(), object(), object()]
        self.received = []

    def forward(self, value, **kwargs):
        self.received.append(kwargs.get("skip_layers", []))
        return value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DiT4SRAdapterTest(unittest.TestCase):
    def test_schedule_repeats_for_each_tile_and_restores_forward(self):
        transformer = _Transformer()
        pipeline = SimpleNamespace(transformer=transformer)
        original_forward = transformer.forward
        schedule = ExplicitSchedule(
            method_id="toy",
            backbone="DiT4SR",
            num_steps=2,
            num_blocks=3,
            skip_blocks_by_step=((1,), (0, 2)),
        )
        with DiT4SRSkipAdapter(pipeline, schedule, tiles_per_step=2) as adapter:
            for index in range(4):
                transformer("tile", marker=index)
            adapter.assert_complete()
            self.assertEqual(transformer.received, [[1], [1], [0, 2], [0, 2]])
            self.assertEqual(adapter.execution_counts, [2, 2])
        self.assertIs(transformer.forward.__func__, original_forward.__func__)

    def test_rejects_wrong_block_count(self):
        schedule = ExplicitSchedule(
            method_id="toy",
            backbone="DiT4SR",
            num_steps=1,
            num_blocks=4,
            skip_blocks_by_step=((1,),),
        )
        with self.assertRaisesRegex(ValueError, "block count"):
            DiT4SRSkipAdapter(SimpleNamespace(transformer=_Transformer()), schedule)


if __name__ == "__main__":
    unittest.main()
