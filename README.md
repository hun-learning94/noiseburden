# NoiseBurden

Code and numerical measurements for the noise-transmission burden in diffusion networks.
This revision extends the arXiv release with restricted-computation and deeper
width controls, and noisy-input recovery probes of public noise predictors.

The repository includes training implementations and configurations, evaluation
workers, saved numerical measurements, and CPU code to reproduce all seven tables
and Figures 2–10 of the revision. Figure 1 is a conceptual diagram and is excluded.
ImageNet images, model checkpoints, and FID reference statistics are external inputs.
Rendering the supplied measurements requires none of those large inputs.

## Reproduce the revision on CPU

Use Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m noiseburden --output artifacts --strict
```

This verifies the included data, checks per-image losses and recovery scores,
regenerates figures and tables, and writes `artifacts/verification.json`.
All seven tables and all nine figure PNGs match the manuscript in the pinned
rendering environment. PDF creation timestamps may differ. On another rendering
platform, omit `--strict`; numerical and table checks still run.

```bash
python -m noiseburden --check-only
```

| Figure | Measurement |
| --- | --- |
| 2 | Bottleneck generation and fixed-time denoising floors |
| 3 | Final-block noise recovery and the rank ceiling |
| 4 | ImageNet128: width, parameter count, denoising gains, residual structure |
| 5 | ImageNet256: generation versus affine noisy-input recovery |
| 6 | Learned versus exact preconditioning coefficients |
| 7 | Full bottleneck denoising-loss grid |
| 8 | ImageNet256 counterpart of Figure 4, at epoch 120 |
| 9 | Additional ImageNet256 denoising and recovery curves |
| 10 | Additional ImageNet128 generation, denoising, and recovery curves |

Tables 1–2 define prediction notation and coefficients; Table 3 gives the
layerwise noise probes; Table 4 contains matched ImageNet256 epoch-120 controls;
Tables 5–6 contain the historical epoch-200 width loss grids; Table 7 compares
ADM and JiT denoising quality. Missing deeper ImageNet256 results remain blank.
Measurements from different epochs or numerical protocols are never substituted
for one another. See [the protocol notes](docs/revision.md).

The previous reproduction workflow remains available:

```bash
python -m noiseburden --arxiv --output artifacts/arxiv --strict
```

Its source, measurements, and references are preserved. See
[the arXiv workflow](docs/arxiv.md) for its training and evaluation commands.

## Train the width controls

Install a CUDA-compatible PyTorch/torchvision pair and then:

```bash
python -m pip install -r requirements-gpu.txt
python -m evaluation.audit_sources
python -m training.run --case i128_restricted256 \
  --data-root /data/imagenet --output /results/restricted256 --dry-run
```

Omit `--dry-run` to train. Use `i128_restricted256_d16`,
`i128_restricted320_d20`, or `i128_restricted384_d24` for deeper controls.
The ImageNet256 counterparts are listed in `configs/controls.json`.
Standard image/noise and bottleneck cases remain in `configs/experiments.json`.

Restricted models widen the residual representation while fixing the attention
projection, MLP intermediate, and conditioning dimensions. The dimensions are
192/512/192 for ImageNet128 and 768/2048/768 for ImageNet256. Extra depth is
introduced at each already-expanded width. The model code includes the input
projection, context-token lift, attention, SwiGLU, AdaLN, and final readout.

The launcher uses effective batch 1024, seed 0, learning rate 0.0002, five warm-up
epochs, constant subsequent learning rate, zero weight decay, and the saved
prediction/loss conventions. Its default recipe is 200 epochs; `--epochs 120`
replicates the shorter comparison endpoint under the same constant-rate schedule.
Microbatch × accumulation × number of GPUs must equal 1024. For example:

```bash
torchrun --standalone --nproc_per_node=2 -m training.run \
  --case i256_restricted896 --microbatch 128 \
  --data-root /data/imagenet --output /results/restricted896
```

Local lossless center-crop caching is enabled by default. Set `JIT_CACHE_ROOT`
to node-local storage and keep outputs on persistent storage. Checkpoints retain
optimizer and EMA state; source versions supporting RNG/loader resumes retain
those states as well. Training on different hardware is a replication and may
produce different weights. Large checkpoints are not included in this repository.

## Evaluate new or supplied checkpoints

The arXiv commands in [docs/arxiv.md](docs/arxiv.md) cover the original bottleneck
suite, noise probes, and residual spectra. The revision adds a separate FP32,
shared-intercept input probe; it preserves the historical BF16 loss grid separately.

First prepare the common validation cache using the user-supplied official JPEGs:

```bash
python -m evaluation.prepare --images /data/imagenet-validation-jpegs \
  --cache external/validation-cache
python -m revision.evaluate prepare --cache external/validation-cache \
  --output external/recovery-cache
```

The second command selects the recorded 384 fitting / 128 tuning / 2000 test
images, preserves their original indices, and uses CPU-only image operations.
Noise is generated on the evaluation GPU from per-image **CPU** seeds.

Prepare `checkpoints.json` from `configs/revision_checkpoints.example.json`.
Prepare `public-assets.json` from `configs/public_assets.example.json` when
including public models, or use `{}` for JiT-only work. All checkpoint paths are
relative to the map file unless absolute. Public download URLs and expected
hashes are in `configs/public_assets.json`; official models retain their original
licenses. Then configure a separate work directory:

```bash
python -m revision.evaluate setup --workspace work/revision \
  --cache external/recovery-cache --checkpoints checkpoints.json \
  --assets public-assets.json
python -m revision.evaluate input --workspace work/revision \
  --case i128_restricted256 --times 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

For new training weights, add `--accept-new-checkpoints` during setup. The actual
hashes and the published hashes are recorded separately. Each command handles
one model in its own process. JiT outputs appear in
`work/revision/<case>/results/`. Use `mse` instead of `input` for image predictors.
Epoch-120 aliases end in `_ep120`, e.g. `i256_eps896_ep120` or
`i256_restricted896_ep120`. The loader checks the completed epoch and EMA1 state.

To repeat the historical BF16 native-loss curves used in Figure 4(c):

```bash
python -m revision.native --workspace work/revision --case i128_restricted256 \
  --cache external/validation-cache --output /results/restricted256-native
```

To evaluate generation using the original JiT single-rank batch-eight Heun-50
protocol and supplied, resolution-matched reference statistics:

```bash
python -m revision.generate --workspace work/revision --case i128_restricted256 \
  --output /results/restricted256-fid --fid-stats /data/fid/jit_in128_stats.npz
```

For public models, install `requirements-public.txt`, obtain the weights listed
in the asset manifest, and supply their paths during setup. The source versions
are pinned in `configs/public_sources.json`. For example:

```bash
python -m revision.evaluate input --workspace work/revision --case adm256
python -m revision.evaluate input --workspace work/revision --case dit256
python -m revision.evaluate validate --workspace work/revision --case adm256
python -m revision.evaluate fid --workspace work/revision --case adm256
```

ADM128 uses `adm128`. U-ViT and LDM use `uvit` and `ldm`; run `validate`, `input`,
then `fid` for each. The public sampler uses adapted Heun-50, no guidance, 10
samples per class, and the same FID-10K reference format. Its CPU checks validate
the schedule/conversion math; workers additionally check actual model outputs.
BF16 is selected only on GPUs with native support; LDM/U-ViT FID remains FP32.

DiT, U-ViT, and LDM recover **noisy latents**. Their latent denoising losses are
not plotted against RGB MSE. ADM recovery uses the 3×3 receptive field of its final
convolution, while JiT/DiT/U-ViT use the features entering their final readout.

## Data, provenance, and anonymity

`metrics/revision/` contains saved measurements and per-image arrays;
`revision/plotting.py` and `revision/tables.py` build the figures and tables from
those records. Rendered CSVs expose the plotted numbers. Checksums are in
`MANIFEST.json` (arXiv) and `REVISION_MANIFEST.json` (revision). Original source
hashes and distributed, anonymized source hashes are distinguished in the source
manifest. Anonymization changes comments, identifying metadata, and path handling;
it does not alter model equations, numerical measurements, or licensed attribution.

The source distribution omits author names, private cluster paths, personal
repository links, scheduling/account records, credentials, and Git history.
Third-party licenses and public model citations remain intact. For an anonymous
mirror, export the current source tree without `.git`; GitHub account and commit
history metadata belong to the hosting repository rather than the source tree.

The CPU reproduction and source/interface checks do not retrain models or rerun
GPU inference. The manuscript's Appendix A contains the mathematical proofs;
Appendix B gives experimental protocols and qualifications.
