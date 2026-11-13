# Revision protocol and measurement boundaries

All diagnostic images are from the official ImageNet validation split, held out
from backbone training. Pixels are center-cropped with the included JiT crop code,
scaled to [-1,1], and never redistributed. The clean-data convention is
`X_t=t X_1+(1-t) epsilon`, `Y_t=X_t/t`, `sigma_t=(1-t)/t`.

## Width controls

Figure 4 uses ImageNet128, RGB patch dimension 192, and completed epoch 200 EMA1
checkpoints. The seven original image/noise cases are accompanied by three
restricted and three deeper restricted cases. Restricted blocks fix attention,
SwiGLU intermediate, and conditioning dimensions at 192, 512, and 192 while
varying representation width through 256, 320, and 384. The corresponding deeper
models use 16, 20, and 24 blocks. The context-token insertion stays after block 4.
Image predictors are displayed at their actual width plus the patch dimension;
noise predictors are displayed at their actual width.

Figure 8 uses ImageNet256, patch dimension 768, and **epoch 120 throughout**.
Restricted widths are 896, 1024, 1152, and 1536. Fixed internal dimensions are
768, 2048, and 768. Deeper controls have 14, 16, 18, and 24 blocks. Only completed
measurements in the manuscript snapshot are included. Missing deeper cases remain
blank even when training/evaluation infrastructure is provided for reproducing them.

The ImageNet128 relative-loss curves use the historical BF16 backbone / FP32
conversion cohort, shared with the unrestricted baseline. They use seven times
0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95 and 2000 common validation images. Imagewise
CPU noise seeds are 8100000 plus the original index in the validation fixture.
The ImageNet256 epoch-120 curves use FP32 backbones, TF32 disabled, FP64 loss
accumulation, and eleven times (also 0.3, 0.4, 0.6, 0.7). Relative gain is
100 * mean(baseline_loss - model_loss) / mean(baseline_loss). These two precision
protocols are labeled separately and are not pooled. Figures show point estimates.

Generation uses FID-10K, 1000 classes × 10 samples, EMA1, guidance 1, seed 0,
single-rank generation batch 8, and 50 Heun steps with the final step Euler.
Model construction precedes the generation RNG reset. The shorter epoch-120
comparison is an execution endpoint of the constant-rate training recipe.

## Input recovery

The current shared-intercept probe pools all patch observations from 384 fitting
images. One of six ridges (0, 1e-8, 1e-6, 1e-4, 1e-2, 1) is selected on 128
disjoint tuning images; 2000 disjoint test images measure explained variance.
The affine intercept is shared across positions. Statistics, solves, predictions,
and metric accumulation use FP64; model inference uses FP32 with TF32 disabled.
No spatial-position offsets are fitted in this protocol. Historical position-
intercept/BF16 results in the arXiv archive remain separately labeled.

The 2512-image subset retains the fixture's original image indices. Corruption
seeds are 8100000 + original index, with one CPU Gaussian draw per image, reused
across model widths and times. Public VAE posterior sampling, when applicable,
uses seeds 7200000 + original index. Native schedules choose the closest log-noise
level for probes; saved records include the actual noise level. Public probes and
JiT curves use reference times 0.1 through 0.9 in increments of 0.1.

The raw final-head input is probed. ADM's 3×3 readout context yields 2304 features
for the 256px model and 2304 for the 128px model. It predicts the center pixel;
512 deterministic pixel positions per image are used. JiT and DiT use tokenwise
readouts. U-ViT uses its token decoder input; LDM uses the final-convolution 3×3
context in its native latent space. The actual original network output is
reconstructed from captured features before evaluating recovery.

Figure 5 includes epoch-200 unrestricted JiT and completed restricted endpoints;
restricted widths 1024/1152 are explicitly marked epoch 120. It does not treat
that panel as a matched-epoch causal comparison. Figures 9–10 supply denoising and
time-dependent recovery. Public DiT/U-ViT/LDM points are labeled latent recovery;
their latent losses are excluded from RGB-MSE panels. The public LDM's FID under
adapted Heun-50 is high; strong recovery is not sufficient to establish good
sampling performance. Measurements support an idealized affine-recovery strategy,
without establishing that every successful noise predictor must use it.

## Residual spectra

A pretrained ADM is a proxy for the population-optimal denoiser, not its exact
value. An affine baseline is fitted on 6000 images and residual covariance is
estimated on 4000 disjoint images. Residual magnitude is the covariance trace
divided by patch dimension. Directions for 95% is the minimum number of leading
eigenvalues whose sum reaches 95% of the total. The central patch is displayed;
additional patch positions remain available in the numerical archive. No shaded
band is interpreted as a confidence interval.

## New computations versus supplied results

The release checks all displayed arrays and stored summaries on CPU. It supplies
model implementations, inference adapters, sampling code, asset hashes, and
validation-image identities to repeat GPU measurements with externally supplied
inputs. Hardware changes may alter training trajectories and low-order numerical
results. Checkpoint checksums and measured epochs are validated before inference;
new weights require an explicit `--accept-new-checkpoints` setup flag.
