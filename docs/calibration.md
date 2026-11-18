# Scalar calibration of pretrained image predictors

This package reproduces the scalar-calibration comparison in the rebuttal using
the released ImageNet256 JiT-B/16 EMA1 checkpoint. It contains the fitting and
evaluation code, the fitted coefficients, per-image numerical measurements,
the actual encoder row basis, and all nine paired FID-10K results.
No model weights or ImageNet pixels are redistributed.

## Rebuild the reported tables and plot

From the repository root, with the CPU requirements installed:

```bash
python -m calibration verify
python -m calibration report --output artifacts/calibration
```

The report generates `REPORT.md`, `coefficients.csv`, `denoising.csv`, `fid10k.csv`,
and a three-panel PNG/PDF. Verification recomputes the coefficients from their
saved per-image moments and the denoising means and gains from paired per-image
errors. FID values are preserved measurements; recomputing FID requires generated
images and the reference statistics. The CPU report does not run GPU inference.

## Method

Use the paper's noise-to-data convention $X_t=tX_1+(1-t)\varepsilon$ and
$Y_t=X_t/t=X_1+\sigma_t\varepsilon$, where $\sigma_t=(1-t)/t$.
The frozen image predictor $f_t$ accesses the input through a rank-$r$ linear
patch encoder $A$. Let $P_A$ project onto its row space. With patch dimension
$d_p$, measure the predictor's error per discarded coordinate,
$v_t=\mathbb E\|(I_{d_p}-P_A)(f_t-X_1)\|^2/(d_p-r)$, on calibration images.
The corrected estimate is

$$
\widehat X_{1,t}^{\mathrm{cal}}
=f_t+\frac{v_t}{v_t+\sigma_t^2}(I_{d_p}-P_A)(Y_t-f_t).
$$

The expectation defining $v_t$ includes the frozen predictor's approximation
error. It is not asserted to be the population conditional variance. With its
population value, the MSE reduction is $(d_p-r)v_t^2/(v_t+\sigma_t^2)$, assuming
the predictor's only access to the relevant input is through the linear encoder
and the corruption is independent isotropic Gaussian noise. Estimated coefficients
are evaluated on separate validation images. This fixed-time MSE identity is
distinct from an assertion about FID.

The released checkpoint has $d_p=768$ and $r=128$. Its expansion after the rank-128
projection is full column rank, so the saved row basis describes the effective
input restriction. The code checks both ranks and the row basis against EMA1.
Fit one scalar at each time, shared across all 256 patch positions and all 640
discarded directions. Sampling evaluates the algebraically equivalent expression
using $X_t-tf_t$ to avoid dividing by $t$ near zero; the correction at $t=0$ is zero.

## Model and data

The frozen official model source is in `calibration/frozen/jit/`, from
[LTH14/JiT](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17),
commit `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`. Source hashes, the public checkpoint
download URL, and its expected SHA-256 are in `configs/calibration/protocol.json`.
Use `model_ema1` from that checkpoint; its SHA-256 is
`4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695`.
The source and checkpoint are external JiT releases, not models trained for this
calibration experiment. The checkpoint does not contain training-epoch metadata.

Install a CUDA-compatible PyTorch/torchvision pair, then `requirements-gpu.txt`.
The measured runs used PyTorch 2.11.0+cu126, NumPy 2.4.3, Pillow 12.1.1 and one
H200. Other CUDA GPUs can run the worker; available memory controls batch size.
FP32 backbone/sampling, disabled TF32, FP64 fitting/projection/error statistics,
and the FP32 sampling projector are explicit in the implementation. Different
hardware, libraries or batch sizes may change floating-point results.

The calibration set contains 10,000 ImageNet training images, ten per class.
The independent evaluation set contains 2,000 official validation images, two
per class. `configs/calibration/samples.json` records every relative filename,
class label, bank position identity and noise seed. Training noise uses CPU
seed `41000000 + source_index`; validation noise uses `23102026 + source_index`
with indices 8000 through 9999 in the recorded validation bank. These indices
are bank identities, not the ImageNet validation filename numbers.
Noise fields are reused across times, producing paired comparisons.

The image preprocessing is the frozen lossless ADM-style center crop to 256,
with RGB uint8 storage and normalization to [-1,1]. There are no training
augmentations during calibration. Supply official ImageNet training images in
class subdirectories and validation JPEGs in a flat directory:

```bash
python -m calibration prepare \
  --train /data/imagenet/train \
  --validation /data/imagenet/validation-jpegs \
  --output work/calibration/assets
```

Preparation is CPU-only. It checks the resulting image, label, index and Gaussian
noise bank hashes against the original experiment. The banks occupy approximately
12 GB. A checksum mismatch is an error rather than silent use of a different
cohort or crop. No fitting image is part of the validation set.

## Fit and evaluate coefficients

Coefficients are already supplied in `metrics/calibration/fit/`. To fit them again:

```bash
python -m calibration fit \
  --checkpoint /models/jit-b-16/checkpoint-last.pth \
  --assets work/calibration/assets \
  --output work/calibration/fit

python -m calibration denoise \
  --checkpoint /models/jit-b-16/checkpoint-last.pth \
  --assets work/calibration/assets \
  --coefficients work/calibration/fit \
  --output work/calibration/test
```

Fitting covers all 49 positive Heun-grid times plus 0.05 and 0.95, giving 51
times. Diagnostics use 0.05, 0.1, 0.2, ..., 0.9, 0.95. Fit unguided and guided
coefficients separately. The guided predictor uses CFG 3 for strict
$0.1<t<1$ and CFG 1 outside that interval, with FP32 boundary comparisons.
The denoising table in the rebuttal uses this same guidance convention.

Per-time JSON/NPZ outputs allow fitting and denoising to resume. Changing the
checkpoint or coefficients requires a fresh output directory. The optional
`--batch-size` defaults to 64. The worker checks native velocity and baseline
trajectory parity before evaluating. The denoising command additionally reports
the half-late schedule; the distributed table contains the original baseline
and full-strength measurements.

## Sample and score all nine conditions

Use one GPU for each invoked sampling command. The following command produces
the unclamped, half-late condition; change `--floor` among 0.05, 0.02 and 0, and
`--method` among `baseline`, `full` and `late_half` to reproduce the complete table:

```bash
python -m calibration sample \
  --checkpoint /models/jit-b-16/checkpoint-last.pth \
  --coefficients metrics/calibration/fit \
  --floor 0 --method late_half \
  --output work/calibration/zero-half

python -m calibration score \
  --input work/calibration/zero-half \
  --fid-stats /data/fid/jit_in256_stats.npz
```

For a new fit, replace `metrics/calibration/fit` by `work/calibration/fit`.
There is no dataset cache requirement for generation when using the supplied
coefficients. FID reference statistics remain an external input; the worker
requires SHA-256 `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`,
the same JiT ImageNet256 reference used by the manuscript evaluation. The bundled
`evaluation/third_party/torch_fidelity` source is identical to the metric
implementation used for these scores. Inception feature weights may be downloaded
by that implementation on first use.

Each condition uses the same 10,000 initial CUDA seeds 0 through 9999, with
class `image_id // 10`. The sampler uses 50 uniform steps, 49 Heun updates and
a final Euler update (99 guided denoiser evaluations; each calls conditional and
unconditional branches). It never evaluates the network at $t=1$.
The velocity denominator is $\max(1-t,\delta)$ for the specified floor $\delta$.
The original checkpoint was trained with floor 0.05; the relaxed conditions
change sampling while leaving the checkpoint fixed.

Full calibration uses the fitted correction at every positive sampling time.
Half-late uses full strength before $t=0.9$ and half strength from $t=0.9$ onward.
Its boundary uses six-decimal time identities, including the FP32 representation
of 0.9. Pixel quantization is `round(clamp((x + 1) * 127.5, 0, 255))` to uint8.
Images are written atomically and interrupted generation resumes missing IDs.

The half-late strength was examined in an exploratory FID-2K screen. These
FID-10K cohorts include those screen samples and are not independent confirmation
cohorts. All original and relaxed-clamp outcomes are included. The CPU report
keeps FID-10K distinct from the upstream paper's reported FID-50K.

## Release scope

The calibration formula applies to other image predictors satisfying the stated
linear input restriction; this runnable model adapter targets the released
ImageNet256 JiT-B/16 experiment. Supporting another model requires its own encoder
map, model interface and fitted coefficients. The package does not include matrix
calibration, cluster schedulers, provider credentials, unpublished training
checkpoints, or a claim that lower denoising MSE universally improves generation.

`CALIBRATION_MANIFEST.json` records checksums independently of the arXiv and
revision manifests. The existing manuscript reproduction workflows are unchanged.
