# AbsorbSR

### Pairwise Trajectory Allocation for Diffusion Super-Resolution

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AbsorbSR is a research codebase for training-free block allocation in frozen
diffusion super-resolution models. It provides:

- pairwise endpoint-response calibration and cancellation-kernel fitting;
- geometry-regularized interpolation between full and time-depth descriptors;
- fixed-budget allocation with a uniform per-step omission budget;
- explicit schedule loading and a DiT4SR integration adapter.

The repository contains source code and a synthetic toy example. It does not
contain manuscripts, paper figures, benchmark results, private experiment
records, pretrained weights, or benchmark datasets.

## Install

Install the lightweight allocation package:

    python -m pip install -e .

Run the unit tests:

    python -m unittest discover -s tests -v

The allocation package uses NumPy. Install the DiT4SR model and its PyTorch
environment separately using the upstream instructions.

## Build a Schedule

The allocator consumes singleton and pairwise endpoint measurements. The
included inputs are synthetic and only demonstrate the file format:

    python scripts/build_pairwise_schedule.py \
      --sites examples/toy_sites.csv \
      --pairs examples/toy_pairs.csv \
      --output examples/toy_schedule.json \
      --num-steps 6 \
      --num-blocks 4 \
      --stage-boundaries 0,2,4,6 \
      --budget 9 \
      --lambda-value 0.5 \
      --restarts 2 \
      --iterations 120 \
      --seed 23 \
      --backbone ToyTransformer

For a real model, supply calibration measurements produced with that frozen
backbone. The calibration procedure uses no ground-truth images. Keep benchmark
and held-out evaluation data separate from schedule construction.

## Apply a Schedule to DiT4SR

Load a pipeline by following the official
[DiT4SR instructions](https://github.com/Adam-duan/DiT4SR), then attach the
schedule adapter:

    from absorbsr import DiT4SRSkipAdapter, load_schedule

    schedule = load_schedule("path/to/your_model_schedule.json")
    adapter = DiT4SRSkipAdapter(pipe, schedule, tiles_per_step=1)
    with adapter:
        output = pipe(
            prompt=prompt,
            control_image=control_image,
            num_inference_steps=schedule.num_steps,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            args=args,
        )
        adapter.assert_complete()

Use a schedule calibrated for the loaded model. For tiled inference, set
tiles_per_step to the number of
transformer calls made at each denoising step. The adapter validates the block
count and total call count, then restores the original forward method.

## Calibration Input Format

sites.csv:

    step,layer,endpoint_l1
    0,0,0.10
    ...

pairs.csv:

    left_step,left_layer,right_step,right_layer,observed_endpoint_l1
    0,0,1,2,0.14
    ...

The builder fits a full descriptor and a geometry-only descriptor using the
same measured pair responses. It interpolates their predictions with
lambda_value, then minimizes the predicted joint endpoint risk under the
specified omission budget.

## Repository Contents

- absorbsr/: pairwise allocator, schedule schema, and host adapter.
- scripts/build_pairwise_schedule.py: calibration-to-schedule command.
- examples/: synthetic input tables and an illustrative schedule.
- tests/: allocator, schedule, and adapter tests.
- MODEL_CARD.md: intended use, dependencies, and release boundary.
- THIRD_PARTY_NOTICES.md: upstream model and dataset terms.
- CITATION.cff: citation metadata for this software repository.

## Weights and Data

No AbsorbSR-trained weights are needed; the method operates on a frozen host
model. This repository does not redistribute DiT4SR, Stable Diffusion 3.5,
datasets, or generated benchmark outputs. Download upstream assets from their
official sources and follow their individual license and access terms.

## License

AbsorbSR-authored source code and documentation are licensed under MIT.
Third-party models, checkpoints, and datasets retain their own terms.
