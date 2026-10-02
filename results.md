# Reporting Protocol

This repository release contains no benchmark outcomes. When publishing
results generated with AbsorbSR, record:

- host model, checkpoint identifier, and checkpoint license;
- dataset source, split, preprocessing, crop or tiling protocol, and sample
  count;
- calibration split and pair-sampling protocol;
- schedule configuration, omission budget, and executed block calls;
- random seeds, solver settings, precision, hardware, and software versions;
- distortion and perceptual metrics with their direction and reference;
- latency boundary, warm-up procedure, synchronization, and aggregation rule;
- paired uncertainty estimates for comparisons.

Keep calibration inputs separate from held-out evaluation data. Report
non-finite values and exclusions explicitly; do not present a finite-only
aggregate as a complete benchmark result.
