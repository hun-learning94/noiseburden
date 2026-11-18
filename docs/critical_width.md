# A1: depth versus model width at comparable parameter counts

This experiment tests whether increasing depth at the critical model width
$d_h=d_p$ recovers the denoising and generation gains of widening. All four
models are ordinary noise predictors trained on ImageNet128. An $8\times8$ RGB
patch has dimension $d_p=192$. No noisy-input shortcut is added, and affine
input recovery is not enforced by these architectures.

| Noise predictor | Model width | Blocks | Trainable parameters | MSE $t=0.1$ | MSE $t=0.5$ | MSE $t=0.9$ | FID-10K |
|---|---:|---:|---:|---:|---:|---:|---:|
| $d_h=d_p$ | 192 | 12 | 8,439,552 | 0.481416 | 0.027037 | 0.002928 | 396.26 |
| Increased depth at $d_h=d_p$ | 192 | 24 | 16,445,952 | 0.479622 | 0.025610 | 0.002740 | 389.89 |
| Unrestricted widening | 256 | 12 | 14,834,096 | 0.100294 | 0.019440 | 0.002571 | 114.38 |
| Fixed widening | 384 | 12 | 16,653,312 | 0.101070 | 0.019086 | 0.002565 | 102.68 |

Doubling depth at $d_h=d_p$ reduces high-noise MSE at $t=0.1$ by 0.37%, while
the gap to either wider model remains large. The 24-block model has 1.25% fewer
parameters than the Fixed-width-384 model. This comparison supports a role for
model width beyond total parameter count. It does not enforce the recovery
assumption of the conditional theorem; [A2](synthetic_width.md) tests that theorem
in its assumed representation class.

## Reproduce the table on CPU

From the repository root, with the standard requirements installed:

```bash
python -m rebuttal verify
python -m rebuttal report --output artifacts/rebuttal
```

`TABLES.md` is computed from saved per-image denoising losses and FID summaries.
The new critical-depth losses are in `metrics/rebuttal/a1/`; the other three
per-image arrays are reused from the original numerical release. The checker
validates the source/data manifest and the identity between induced clean-image
and noise errors, using their actual finite-precision tolerances.

## Train the models

Install a CUDA-compatible PyTorch/torchvision pair, the GPU requirements, and the
pinned FID implementation described in [the original workflow](arxiv.md).
`configs/rebuttal_width.json` records each model shape, training arguments,
checkpoint checksum, and source hashes. The exact critical-depth implementation
is in `training/controls/critical/`. The other three models reuse the released
implementations. For example:

```bash
python -m rebuttal.critical train --case i128_eps192_d24 \
  --data-root datasets/imagenet --output work/train/i128_eps192_d24 \
  --microbatch 256 --accumulation 4 --epochs 200
```

Repeat with `i128_eps192`, `i128_eps256`, and `i128_restricted384` for the other
rows. Add `--dry-run` to inspect a configuration without loading images or
allocating a GPU. Node-local image caching is enabled by default and preserves
the original image pixels and augmentation protocol.

All models use 200 completed epochs, seed 0, effective batch 1024, learning rate
$2\times10^{-4}$, five warmup epochs followed by a constant rate, no weight decay,
and EMA decay 0.9999 for EMA1. Corruption is
$X_t=tX_1+(1-t)\varepsilon$, with clean data at time one. Training uses the
released image-space weighted-velocity loss, logit-normal time distribution
with mean -0.8 and standard deviation 0.8, and endpoint floor 0.05.
The token stream has width 192, 256, or 384. For the Fixed and critical-depth
implementations, attention, conditioning, and SwiGLU intermediate dimensions
are 192, 192, and 512. Context tokens enter after block 4.

## Repeat denoising and FID evaluation

Create the full 10,000-image validation cache with `evaluation.prepare` as
documented in [the original workflow](arxiv.md). Supply model checkpoints using
`configs/rebuttal_checkpoints.example.json`; relative paths are resolved against
that JSON file. The published SHA-256 checksums identify the recorded EMA1
checkpoints. To evaluate newly trained checkpoints, explicitly use
`--accept-new-checkpoints` at setup.

```bash
python -m rebuttal.critical setup --workspace work/a1 \
  --checkpoints configs/rebuttal_checkpoints.example.json
python -m revision.native --workspace work/a1 --case i128_eps192_d24 \
  --cache datasets/validation-cache --output work/a1/native/i128_eps192_d24
python -m revision.generate --workspace work/a1 --case i128_eps192_d24 \
  --fid-stats datasets/jit_in128_stats.npz --output work/a1/fid/i128_eps192_d24
```

Run each evaluator in a separate process for each case, so imports of the pinned
model implementations remain isolated. Denoising uses the same 2,000 held-out
images (indices 8000–9999 in `configs/validation_samples.json`) for every model.
Per-image CPU Gaussian seeds are 23102026 plus the original fixture index.
The batch is 16; the backbone uses native BF16, while prediction conversion and
per-image squared error use FP32. Reported MSE is averaged over RGB coordinates.
The supplied arrays also retain $t=0.05,0.2,0.8,0.95$.

Generation uses EMA1, guidance 1, 50 Heun steps with final Euler, endpoint floor
0.05, single-rank batch 8, seed 0 reset after model construction, and ten samples
for each of 1,000 classes. The ImageNet128 FID-reference SHA-256 is
`4442907dce30a06cc0449d90bfe6450462c40ca157c507c679ec1d5eb64d7ad5`.
ImageNet images, predictor checkpoints, and FID reference statistics are external
inputs. The repository includes the numerical measurements needed for the CPU
table, and runnable code for training and evaluation.
