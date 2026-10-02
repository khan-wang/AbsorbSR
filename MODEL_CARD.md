# AbsorbSR Research Code Card

## Summary

AbsorbSR is research software for training-free inference allocation in a
frozen diffusion super-resolution backbone. It fits a pairwise response model
from singleton and pairwise endpoint measurements, then constructs an offline
fixed schedule subject to a block-call budget.

## Intended Use

The DiT4SR adapter applies an explicit schedule to the official pipeline. The
repository includes only a synthetic schedule example; it does not include a
paper deployment schedule.

## Method and Artifacts

- absorbsr/pairwise_allocator.py: endpoint calibration and geometry-regularized
  schedule allocation.
- absorbsr/dit4sr_adapter.py: routes a loaded DiT4SR pipeline through a frozen
  per-step schedule.
- examples/: synthetic calibration tables and an illustrative schedule.

## Evaluation

The software can be applied to image super-resolution pipelines that expose
the required transformer-block bypass interface. No benchmark outcomes are
included in this release.

## Limitations

The adapter targets the skip_layers interface in the official DiT4SR
transformer. It requires a known, fixed number of transformer calls per
denoising step. Other backbones need a compatible adapter. Quality and latency
depend on the backbone, hardware, preprocessing, and evaluation protocol.

## Training and Checkpoints

AbsorbSR does not train or modify the DiT4SR checkpoint. No AbsorbSR-trained
weights are required or included. Obtain DiT4SR and any base-model checkpoints
from their official sources and follow their licenses.

## Release Boundary

The repository contains generic implementation code and synthetic examples.
It excludes manuscripts, publication figures, frozen paper schedules, result
tables, raw experiment records, checkpoints, and benchmark images.
