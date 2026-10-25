# NoiseBurden

Reproducibility code and measurements for **The Noise-Transmission Burden of
Diffusion Networks: Why Image Prediction Can Outperform Noise Prediction**.

This repository contains four parts:

1. **Training code and configurations** for the 26 JiT checkpoints used in the
   target/preconditioning and model-width experiments.
2. **Evaluation code** for generation/FID, fixed-time denoising, layerwise noise
   probes, affine input recovery, and residual spectra with ADM references.
3. **Saved numerical results**, including per-image denoising errors, probe
   scores, covariance eigenvalues, learned coefficients, and generation metrics.
4. **Rendering code** for every manuscript table and Figures 2–9. The conceptual
   architecture diagram, Figure 1, is outside this reproduction workflow.

The checkout is self-contained: no original cluster workspace is needed.
Training and new model evaluations require your own ImageNet images and model
weights. Rebuilding the published tables and figures requires neither.

## Reproduce the tables and figures on CPU

Use Python 3.11 and run from this checkout:

```bash
python -m pip install -r requirements.txt
python -m noiseburden --output artifacts --strict
```

The command verifies the included inputs, recomputes metric consistency checks
and paired gain intervals, produces six LaTeX/CSV tables and eight PDF/PNG
figures, and writes `artifacts/verification.json`. In the pinned environment,
all table source blocks and figure PNGs match the manuscript references exactly.
PDF creation timestamps may differ.
The strict PNG check also depends on font/rendering libraries; omit `--strict`
on a different rendering platform. Table values are always checked exactly.

```bash
python -m noiseburden --check-only
```

This independently checks 182 native-loss cells against their per-image arrays,
78 noise-probe scores, 16 paired input-probe cases, and 14 central ADM spectra
against their included covariance eigenvalues. `MANIFEST.json` records SHA-256
identities of the distributed data, configurations, and reference outputs.

| Manuscript item | Output / underlying measurement |
| --- | --- |
| Table 1 | Target/estimator notation; fixed definition template |
| Table 2 | Fixed preconditioning coefficients; fixed definition template |
| Table 3 | 13 layers × 6 times of affine noise recovery |
| Tables 4–5 | Absolute ImageNet256/ImageNet128 clean-image MSE grids |
| Table 6 | ADM/JiT reference denoising quality at both resolutions |
| Figure 2 | Bottleneck FIDs and fixed-time denoising with theoretical floors |
| Figure 3 | Final-block noise recovery and the 1/6 ceiling |
| Figure 4 | ImageNet256 width FIDs, relative denoising gains, residual structure |
| Figure 5 | Paired ImageNet256 denoising and affine input recovery |
| Figure 6 | Learned preconditioning coefficients relative to exact values |
| Figure 7 | Complete bottleneck denoising-loss heatmap |
| Figure 8 | ImageNet128 counterpart to Figure 4 |
| Figure 9 | 1×4 denoising/input-recovery comparison, both resolutions and targets |

Outputs are under `artifacts/figures/`, `artifacts/tables/`, and
`artifacts/metrics/`. Generated table source preserves the manuscript's labels
and notation; Table 1's equation references resolve when included in the paper.
The `reference/` directory contains the published experimental figure assets
and table blocks used to check reproduction, including the 1×4 Figure 9.

## Training the checkpoints

The earlier anonymous training release is pinned from
[kilonova801/jit](https://github.com/kilonova801/jit) at commit
`25667dd44e24e619f2e3300e4b16dbe4e91680ad`. Its source is included in
`training/base/`; no access to that repository is required after cloning this one.
`training/snapshots/` overlays the exact run-specific files, with original file
hashes in `training/source_manifest.json`. This preserves width, patch-embedding,
preconditioning, and loss implementations that differ from the older release.

Install a compatible CUDA PyTorch/torchvision pair and the additional packages:

```bash
python -m pip install -r requirements-gpu.txt
python -m evaluation.audit_sources
python -m training.run --case i128_eps256 \
  --data-root /data/imagenet --output /results/i128_eps256 --dry-run
```

To train, omit `--dry-run`. For multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=2 -m training.run \
  --case i256_eps896 --data-root /data/imagenet \
  --output /results/i256_eps896 --microbatch 256
```

The launcher retains the effective batch size of 1024, seed 0, 200 epochs,
learning rate 0.0002, five warm-up epochs, constant subsequent learning rate,
zero weight decay, and the saved target/loss/preconditioning configuration.
Microbatch size and accumulation may be changed while preserving their product
with GPU count. Incomplete final effective batches are omitted. New training is
a replication rather than a promise of byte-identical checkpoint weights;
distributed kernels and hardware can change the stochastic trajectory.

Node-local center-crop caching is enabled by default. Set `JIT_CACHE_ROOT` to
node-local disk and keep `--output` on persistent storage. The adapter preserves
pixels, labels, sample order, random flips, and augmentation RNG consumption.
Use `--no-cache` for a different data environment. Do not run historical
continuation scripts inside the snapshots directly; use the portable launcher.

## Recompute the numerical measurements

The frozen mathematical implementations are in `evaluation/frozen/`.
The portable setup command supplies local paths and checksums in an external
work directory; it does not modify those implementations or the included paper
metrics. Use one checkpoint or teacher per process to isolate model namespaces.

1. Obtain the original official 50,000 ImageNet validation JPEGs. Image pixels
   are not redistributed here. `configs/validation_samples.json` specifies the
   exact class-balanced 10,000-image subset and class labels. The canonical
   class order and all validation synsets are included in `configs/`.
2. Supply checkpoint paths in a JSON map. Start with
   `cp configs/checkpoints.example.json checkpoints.json`, edit paths relative
   to that file or use absolute paths. IDs and expected hashes are in
   `configs/experiments.json`. ADM checkpoint URLs and a downloader are included.
3. Prepare arrays, configure the run, and execute the relevant workers below.

Image preprocessing is CPU-only:

```bash
python -m evaluation.prepare --images /data/imagenet-validation-jpegs \
  --cache /scratch/validation-cache
```

On the evaluation GPU, prepare the input-probe corruption bank. Its first 640
fields reproduce the historical single CUDA draw and the CUDA 2×2 reduction for
128px. The remainder use the frozen per-image CPU seeds:

```bash
python -m evaluation.prepare --cache /scratch/validation-cache \
  --noise-only --input-noise
python -m evaluation.run setup --workspace work/evaluation \
  --cache /scratch/validation-cache --checkpoints checkpoints.json
```

The original input-corruption fingerprints require the recorded CUDA generator
implementation (PyTorch 2.11.0+cu126 on H200). The preparation command verifies
the first-image hashes. Native/spectral corruptions use per-image CPU generators.

```bash
python -m evaluation.run native --workspace work/evaluation --case i256_eps768
python -m evaluation.run input --workspace work/evaluation --case i256_eps768
python -m evaluation.run noise --workspace work/evaluation
python -m evaluation.run spectra --workspace work/evaluation \
  --resolution 256 --teacher adm256
python -m evaluation.run analyze --workspace work/evaluation --resolution 256
```

Run native evaluation for all 26 JiT IDs, input probes for the 16 width IDs,
and each spectral teacher before analysis. At 256px the teachers are `adm256`,
`i256_eps1536`, and `i256_x768`; at 128px they are `adm128`, `i128_eps384`,
and `i128_x192`. Use `--cpu` on `analyze` for CPU covariance/eigenspectrum
analysis. `native`, `input`, and `spectra` support `--smoke`. The noise worker
evaluates the complete paired six-time cohort. Reconfigure with
`--accept-new-checkpoints` when evaluating fresh training runs; their actual
hashes are recorded separately from the published checkpoint identities.

The model weights are external large inputs. To download an official ADM model:

```bash
python -m evaluation.download_adm --resolution 128 --output checkpoints/adm128.pth
```

To generate images and evaluate FID using the same JiT reference-statistics
format:

```bash
python -m evaluation.generate --workspace work/evaluation --case i256_eps896 \
  --output /results/i256_eps896/samples --fid-stats /data/fid/jit_in256_stats.npz
```

Generation uses EMA1, seed 0, 50-step Heun with an Euler last step, 10 images per
class, and batch 8. Width runs use guidance 1; bottleneck runs use guidance 2.9
over [0.1,1.0]. Supply the resolution-matched reference statistics used by JiT;
the evaluator records their checksum. `training/base/prepare_ref.py` contains
the original clean-reference image preparation code. The frozen fork of
`torch-fidelity` is included with its license. Regenerating the published plots
uses the included FIDs and does not rerun image generation.

## Build figures from newly computed numerics

After completing a numerical rerun, extract its learned coefficients and collect
the outputs into the renderer's input format:

```bash
python -m evaluation.coefficients --workspace work/evaluation \
  --output /results/coefficients
python -m evaluation.collect --workspace work/evaluation \
  --generation /results/generation --coefficients /results/coefficients \
  --output /results/new-metrics
python -m noiseburden --metrics /results/new-metrics --output /results/new-figures
```

The generation folder contains `<case-id>/metrics.json` from
`evaluation.generate` for each JiT case. Collection requires the full native,
probe, and teacher output set and independently checks spectral summaries
against their saved eigenvalues. If diagnostics are repeated with the identical
published checkpoint hashes, use `evaluation.collect --reuse-saved-generation`
instead of the generation/coefficient arguments. This explicitly reuses those
saved measurements and is refused for fresh checkpoint weights. The original
distributed paper metrics are preserved.

## Protocol and interpretation

All denoising, probe, and residual-spectrum measurements distributed here use
official validation images held out from backbone training. Native losses use
2,000 test images per resolution. Noise probes use 8,000 fit / 2,000 test images;
input probes use 384 fit / 128 tuning / 2,000 test images. The input-probe native
MSE uses the same images and corruptions as its fitted input readout; its draws
differ from the native-loss grid. Residual baselines use 6,000 fit / 4,000
evaluation images. These splits are disjoint within each diagnostic.

The time convention is `X_t=t X_1+(1-t) epsilon`, with
`sigma_t=(1-t)/t`; small time means high noise. Native backbones use BF16 and
FP32 conversion; noise probes use FP32 inference and FP64 solves; residual
proxies use FP32 inference and FP64 covariance estimates. ADM is a practical
proxy for the population-optimal denoiser. The spectral shading spans patch
positions, not a confidence interval. FID values describe individual training
runs, not run-to-run uncertainty.

The theorem, corollary, and supporting-lemma proofs are in Appendix A of the
manuscript; experiment details and qualifications are in Appendix B. This
repository reproduces the numerical evidence and does not replace those proofs.

## Validation and provenance

The CPU workflow was checked in the pinned rendering environment against the
paper: all six table blocks are identical and all eight PNGs match pixel for
pixel. Model-source hashes, configuration construction, and portable runtime
setup are checked separately. Full checkpoint training and GPU inference are
not rerun when packaging this release; their numerical outputs and frozen
implementations are included. See `metrics/provenance.json`,
`metrics/protocol.json`, `training/source_manifest.json`, and
`THIRD_PARTY_NOTICES.md` for identities, versions, and source attribution.
