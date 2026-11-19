# Scalar calibration of released Pyramid-JiT512

We apply the input-bottleneck correction to the released 2.18B-parameter
Pyramid-JiT512 text-to-image model, retaining its native sampler. On the first
fixed COCO FID-10K block, full calibration reduces Inception-FID from
**19.7415 to 18.8867 (4.3%)**. The backbone is frozen. This is our matched
COCO evaluation of the released model; it is not a reproduction of the
authors' FD-DINOv2 evaluation.

This release includes the implementation, all 51 pairs of fitted coefficients,
per-image fitting errors and held-out losses, exact data identities, source and
asset hashes, and completed FID audit receipts. Image pixels, model weights,
generated PNGs, text features, and Inception reference statistics remain external
inputs. The commands below rebuild them.

## Results and completion status

Snapshot: November 19, 2026 UTC. Only completed, audited comparisons are numerical
results here. The first 2,000 generations are a subset of the first 10,000.

| Cohort | Uncalibrated | Full calibration | Half-late calibration |
|---|---:|---:|---:|
| First 2,000 | 33.34746063 | 32.57026738 | 33.16996545 |
| First 10,000 | 19.74148339 | **18.88668921** | 19.48029046 |
| Indices 10,000–19,999 | Pending | Pending | Pending |
| Indices 20,000–29,999 | Pending | Pending | Pending |
| Full 50,000 | Pending | Pending | Pending |

The three FID-10K cohorts are fixed, disjoint blocks of the same 50K caption/noise
bank: zero-based indices `[0,10000)`, `[10000,20000)`, and `[20000,30000)`.
Each variant uses the same caption and initial-noise seed at a given index.
The blocks have distinct caption IDs and noise seeds, but different captions can
describe the same COCO image. They are **caption-and-noise subset evaluations**,
not three repeats with fixed captions and different seeds, and not independent
training or calibration repeats. The blocks were designated during the ongoing
50K extension, after the first-block result was available; later block scores
were not used to choose their boundaries. The initial 50K caption bank was fixed
before generation. No additional seed-repeat result is included.

We will report all three block scores and their descriptive mean and sample SD
once complete. Those summaries are pending. The mean of three FID-10K values is
not pooled FID-30K or FID-50K; the three blocks also overlap the final 50K sample.
We make no significance or independent-replication claim from these blocks.

Caption-conditioned held-out clean-image MSE:

| Clean time | Baseline | Full | Half-late | Full reduction |
|---:|---:|---:|---:|---:|
| 0.5 | 0.01925921 | 0.01925888 | 0.01925888 | 0.0017% |
| 0.9 | 0.00343066 | 0.00340416 | 0.00341021 | 0.77% |
| 0.95 | 0.00209190 | 0.00198706 | 0.00201072 | 5.01% |
| 0.98 | 0.00145503 | 0.00102814 | 0.00112847 | 29.34% |

The CPU report includes all twelve diagnostic times and both conditional and
negative-prompt branches. Raw per-image losses support recomputation of every
mean, paired improvement, and standard error.

## Model and correction

Pyramid-JiT predicts the clean image (`x`-prediction); we call this image
prediction below. Clean data are at time one:

\[
X_t=tX_1+2(1-t)\epsilon,\qquad
Y_t=X_t/t,\qquad \sigma_t=2(1-t)/t.
\]

There are 256 patches per image, with patch dimension $d_p=3072$. The linear
embedding maps $3072\to1024\to2944$ and has rank 1024. We verify both factors'
rank and construct the orthogonal projector $P_A$ onto the first factor's row
space. Thus $I-P_A$ selects 2048 discarded directions per patch. We use the
**BF16-cast released patch weights**, matching native autocast, to construct this
projector. Using the nominal FP32 weights instead gives relative row-space
leakage 0.001348763 against the operative BF16 matrix. The released FP64
orthonormal row basis is included in `metrics/pyramid_calibration/encoder_rows.npy`.

For each time and each raw prediction branch, we estimate the mean squared
prediction error per discarded coordinate on the calibration images, denoted
$v_t$. We then apply

\[
f_t^{\mathrm{cal}}=f_t+c_t(I-P_A)(Y_t-f_t),\qquad
c_t=\frac{v_t}{v_t+\sigma_t^2}.
\]

One scalar is shared across patches and discarded coordinates, separately for
the caption-conditioned prediction and the negative-prompt prediction. The
implementation uses the algebraically equivalent increment
$v_t t\,(I-P_A)(X_t-tf_t)/(v_t t^2+4(1-t)^2)$, avoiding division by $t$.
At the initial $t=0$ sampling step, the correction is zero. No fit at $t=1$ is
needed. We fit all 49 positive Euler sampling times plus 0.05 and 0.95, without
interpolating coefficients. Exact FP32 source timesteps are retained in each
coefficient record; the released model embeds integer source time $1000(1-t)$.

Full calibration applies this increment at all positive sampling times.
Half-late calibration applies full strength for $t<0.9$ and half strength for
$t\ge0.9$. Both raw branches are corrected **before** adaptive projected
guidance (APG). The ideal Gaussian risk identity motivates the correction for
the raw image predictions; it is not asserted for the history-dependent APG
output or as a bit-exact identity under BF16 input/output rounding. The measured
denoising losses include the actual implemented correction and rounding.

## Data and numerical protocol

- **Calibration:** 10,000 distinct COCO2014 training images, one fixed caption
  per image. No validation image/caption is used to fit coefficients.
- **Held-out denoising:** 2,000 distinct COCO2014 validation images and captions,
  disjoint from calibration. One fixed per-image CPU Gaussian draw is shared
  across times and variants. Fit seeds are `81000000 + index`; test seeds are
  `82000000 + index`.
- **Generation:** 50,000 distinct validation caption IDs, sampled uniformly
  without replacement. Caption selection uses NumPy seed `20261118`, following
  the fit/test draws in the frozen selection code. Initial generation seed is
  the caption's zero-based index, `0..49999`, using the released CUDA generator.
  The exact ordered cohorts are in `configs/pyramid_calibration/cohorts.json.gz`.
- **Reference:** all 40,504 COCO2014 validation images. Fit, test, and real-reference
  pixels use the same ADM/JiT deterministic square center crop to 512 pixels:
  repeated BOX halving while the shorter side is at least 1024, bicubic resize
  to shorter side 512, then center crop. Generated PNGs are already 512×512.
  COCO train/validation separation does not establish separation from the
  released model's pretraining data; pretraining overlap is unknown.
- **Native sampler:** 50 Euler steps; noise scale 2; APG scale 15, momentum
  −0.75, eta 0, rescale 10; negative prompt
  `watermark, signature, logo, copyright, url`; no endpoint denominator floor.
  Export clips/quantizes to uint8 exactly as the released sampler does.
- **Precision:** released FP32 master parameters, native BF16 forward/sampling,
  FP32 calibration projection, FP64 fitting/projector/loss statistics; SDPA;
  TF32 disabled. Fitting/denoising batch 64, generation batch 40, text batch 16.
  The measured GPU was H200. The commands retain those batch sizes and require
  native BF16 support for model inference.
- **FID:** the frozen JiT fork of `torch-fidelity`, Inception-v3-compatible
  2048-dimensional features, FP64 means/covariances, one common reference.
  Source files and Inception weights are identified by SHA-256. All numerical
  scores here are Inception-FID. No EDM2-style FD-DINOv2 computation is included.

The 2K screening prefix, first 10K prefix, and subsequent 50K extension share
captions/seeds. The extension was retained regardless of the sign of the first
FID result. The two correction schedules were specified before FID evaluation;
a held-out denoising gate retained both when low-noise conditional MSE improved
at at least one tested time and no tested time at or above 0.8 showed more than
three standard errors of deterioration. This gate uses validation denoising,
not FID; it does not turn the denoising measurements into an untouched final
test of schedule selection. Its exact receipt is included. No coefficient or
correction strength was selected using FID.

## Reproduce tables and numerical audits on CPU

From a checkout, using Python 3.11 and NumPy (the main repository requirements
are sufficient):

```bash
python -m pyramid_calibration verify
python -m pyramid_calibration report --output artifacts/pyramid-calibration
```

Verification checks the release hashes, exact cohorts, all per-image fitting
moments and coefficients, all held-out MSE/paired statistics, row-basis
orthonormality, and AST identity of 21 scientific worker definitions against
the run snapshot. Published FIDs are checked against retained audits that
recomputed them from the original saved features and reference. The CPU command
does not recompute Inception features or rerun GPU sampling.

`artifacts/pyramid-calibration/REPORT.md`, `fid.csv`, and `denoising.csv` are
rebuilt from the measurements. The report explicitly retains pending entries.

## Repeat fitting and generation

Use a separate Python 3.11 CUDA environment. The exact measured package versions
are in `configs/pyramid_calibration/environment.json`. For the recorded CUDA
12.6 PyTorch build:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-pyramid.txt
```

Run from the repository root. Choose a workspace with enough persistent storage
for COCO archives, cropped pixel banks, downloaded model/text weights, text
features, generated PNGs and Inception features. These commands do not submit
scheduler jobs or provision GPUs.

```bash
# CPU: pinned model/text downloads, official COCO archives, exact banks/projector.
python -m pyramid_calibration prepare --workspace work/pyramid

# GPU: refit all scalars, held-out denoising, and real-reference features.
python -m pyramid_calibration fit --workspace work/pyramid

# GPU: repeat for each variant; each command owns one variant's output directory.
python -m pyramid_calibration sample --workspace work/pyramid --variant baseline --count 10000
python -m pyramid_calibration sample --workspace work/pyramid --variant calibrated --count 10000
python -m pyramid_calibration sample --workspace work/pyramid --variant late_half --count 10000
```

Preparation downloads the original COCO2014 annotation, train, and validation
archives from `images.cocodataset.org`. Existing complete archives may instead
be placed at `work/pyramid/output/coco/{annotations,train2014,val2014}.zip` before
preparation. CRC checks and pixel-bank hashes must match. The public
[COCO dataset](https://cocodataset.org/#download) retains its own image/annotation
terms and attribution. Only selected public annotations and identifiers are
redistributed here, not image pixels.

To reuse the released fit instead of refitting, use a **separate workspace**:

```bash
python -m pyramid_calibration prepare --workspace work/pyramid-supplied
python -m pyramid_calibration use-supplied-fit --workspace work/pyramid-supplied
python -m pyramid_calibration reference --workspace work/pyramid-supplied
python -m pyramid_calibration sample --workspace work/pyramid-supplied --variant calibrated --count 10000
```

The explicit `use-supplied-fit` command copies only released coefficients and the
original denoising decision. It records that no local refit/denoising rerun was
performed and refuses to replace different local coefficients. Preparation
checks the released row basis against the actual downloaded BF16-cast encoder.

For three FID-10K blocks, extend each variant with `--count 30000`; for the full
bank, use `--count 50000`. Committed images/features are reused. Then score the
fixed blocks, repeating for each variant:

```bash
python -m pyramid_calibration score --workspace work/pyramid --variant baseline --start 0 --count 10000
python -m pyramid_calibration score --workspace work/pyramid --variant baseline --start 10000 --count 10000
python -m pyramid_calibration score --workspace work/pyramid --variant baseline --start 20000 --count 10000
```

The scoring command checks exact, finite, ordered feature coverage and uses the
same frozen FID implementation/reference. Generation also scores its complete
prefix automatically; a 30K prefix score is distinct from the three block scores.

Workers write atomic images/features and use one writer lock per stage/variant.
They stop resumably with exit code 75 after 3h45m or a termination signal; rerun
the same command until complete. Native baseline PNG equality against upstream
generation is checked at each model-worker startup before measurement. A failed
check is an error, not a successful evaluation. Do not change batch size,
precision, projector, captions, or seed assignment when claiming this protocol.

## Source and validation provenance

- [Upstream Pyramid-JiT](https://github.com/Linum-AI/pyramid-jit) commit
  `dbe6d654d8c2770e00299d5ee562430447a40f8c`.
- [Released model](https://huggingface.co/Linum-AI/pyramid-jit) revision
  `05b4ca725545096b5d1485e185c074e70a72a769`; model SHA-256
  `bc8249846b26b93ea4a7f91cd997f5e04b7cff7d158097b14de5365ebe64d430`.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; released chat template and
  layers 7/15/27. Individual asset hashes are in `configs/pyramid_calibration/assets.json`.

Upstream code and licenses are preserved under `pyramid_calibration/frozen/`.
The portable worker preserves the run's scientific functions; adaptations cover
imports, workspace paths, reporting, execution identifiers, prefix choices, and
strict preparation checks. The release manifest is
`PYRAMID_CALIBRATION_MANIFEST.json`. Original run-worker and AST hashes are in
`configs/pyramid_calibration/source_parity.json`.

The measured run passed actual GPU native-sampler parity and numerical checks.
The portable wrapper and numerical reproduction were checked on CPU for this
release; a fresh end-to-end GPU run of the portable wrapper has not been made.
The retained FID audits also check PNG CRC/mode/shape, ordered finite features,
unchanged earlier-prefix outputs, model/cohort identities, and recomputed FID.
These distinctions separate source/interface validation from a new inference run.
