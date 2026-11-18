# A2: controlled tests of the conditional width theorem

These experiments test Theorem 5 with exact population-optimal denoisers and
affine noisy-input recovery enforced by construction. They separate the
theorem's quantitative prediction from approximation of the reference denoiser
by a learned model.

## Quantity being tested

Let $X_1\in\mathbb R^{d_p}$ be a clean patch,
$X_t=tX_1+(1-t)\varepsilon$, and $Y_t=X_t/t$, where
$\varepsilon\sim\mathcal N(0,I)$ and $\sigma_t=(1-t)/t$. Clean data are at
time one. The exact posterior mean is $\mathbb E[X_1\mid Y_t]$.
The best affine estimator from $Y_t$ is

$$
\ell_t(Y_t)=\mu+\Sigma(\Sigma+\sigma_t^2I)^{-1}(Y_t-\mu),
\qquad R_t=\mathbb E[X_1\mid Y_t]-\ell_t(Y_t).
$$

The model predicts noise with an affine head on
$h(X_t)=[X_t;g(X_t)]\in\mathbb R^{d_p+k}$. Selecting its first $d_p$ coordinates
and dividing by $t$ recovers $Y_t$ exactly. Thus Assumption 1 holds, and the
additional model width is $k=d_h-d_p$. Theorem 5 predicts the minimum
approximation error within this representation class:

$$
\inf\mathbb E\left\|\widehat X_1-\mathbb E[X_1\mid Y_t]\right\|^2
=\sum_{j>k}\lambda_j\!\left(\operatorname{Cov}(R_t)\right).
$$

The infimum permits arbitrary square-integrable features subject to affine
input recovery. The trained finite MLPs test whether this optimum can be
approached in practice. The experiment compares their held-out error after
refitting the affine head with the predicted spectral tail.

## Two distributions with exact denoisers

**Prescribed spectra.** In dimension $d_p=64$, sample
$X_1=Uz+0.1n$, where $U$ has orthonormal columns, $n$ is standard Gaussian,
and independent $z_j$ take values $\pm1$ with equal probability. The four
reported ranks are 4, 8, 16, and 32. The posterior mean factorizes into one
tanh term per latent coordinate and agrees with the affine estimator on
$\operatorname{span}(U)^\perp$. Consequently the residual covariance has the
chosen rank, with equal nonzero eigenvalues. At $t=0.5$, each is approximately
0.0487605. The predicted error is $\max(r-k,0)$ times this eigenvalue.
`gmm.py` computes these eigenvalues by one-dimensional quadrature; the
Monte Carlo spectrum supplies the directions used for the readout diagnostics.

**ImageNet-fitted patch mixture.** In dimension $d_p=192$, fit a 128-component
full-covariance Gaussian mixture to 1,216,000 $8\times8$ RGB patches, 32 from
each of 38,000 ImageNet training images. Another 2,000 disjoint training images
provide 64,000 validation patches. Images are center-cropped to 128 pixels and
scaled to $[-1,1]$; each patch is flattened in HWC order. The fitted mixture is
then the data distribution: new clean patches are sampled from it. Its
responsibility-weighted sum of component Wiener estimators is the exact
population-optimal denoiser for that distribution. This does not make it the
exact denoiser for the original ImageNet distribution.

The fitted mixture tensors are included in
`synthetic_width/data/patch_gmm_K128.pt` (about 38 MB), so repeating this synthetic
experiment requires no ImageNet images. `prior.json` records its checksum,
distribution digest `9dd9d694daee1200`, fitting dimensions, and validation
log-likelihood. Only numerical mixture parameters and public fitting metadata
are distributed. To refit the mixture from ImageNet, use:

```bash
python synthetic_width/fit_patch_gmm.py --root datasets/imagenet/train \
  --Ks 128 --out_dir work/synthetic/refitted-prior \
  --cache work/synthetic/patches.npz
```

This uses the included crop implementation and the original FP64 EM procedure
(150 iterations, covariance regularizer $10^{-5}$, seed 0). A refitted mixture
can differ numerically from the supplied distribution.

## Training and readout refitting

For each fixed time and width, $g$ is a six-block residual MLP with RMSNorm,
SwiGLU, and zero-initialized modulation gates. Its internal width is 256 for
the prescribed-spectrum family and 512 for the ImageNet-fitted mixture; its
output has $k$ coordinates. At $k=0$ the features are just $X_t$.
The affine noise head is initialized to realize the best affine clean-image
estimator. We train using the exact posterior mean as the clean target. This
has the same population minimizer as training against $X_1$, while removing
the irreducible regression noise.

Each run uses 30,000 AdamW steps, batch 4096, initial learning rate 0.001,
1,000 warmup steps then cosine decay, betas (0.9, 0.95), epsilon $10^{-12}$,
no weight decay, and seed 0. New samples are drawn every step. Network
parameters and operations are FP32; exact posterior means and distribution
statistics are computed in FP64, with TF32 disabled.

After training, freeze the features and refit the affine noise readout by least
squares against the exact population noise predictor. Fitting and testing use
separate fresh sample sets of 65,536 patches each (probe seed 20261109); the
relative ridge is $10^{-8}$. `approx_refit` is the resulting held-out
approximation error and is the primary comparison with the theorem. A separate
262,144-sample evaluation (seed 20261108) retains errors of the trained head,
Bayes risk, and affine-baseline error. Errors are sums over patch coordinates.

For the ImageNet-fitted mixture, estimate the residual covariance from two
million noisy samples at each time, using seed 1234 and FP64 moment
accumulation/eigendecomposition. The predicted tail and its normalizing residual
energy come from this same spectrum estimate. The posterior mean itself is
analytic; the expectations used to measure errors and spectra are sampled.
For the binary family, the predicted eigenvalues instead use quadrature.

## Results and CPU reproduction

```bash
python -m pip install -r requirements.txt
python -m rebuttal verify
python -m rebuttal report --output artifacts/rebuttal
```

The generated `TABLES.md` contains all 32 mixture width–noise settings and all
20 prescribed-spectrum settings quoted in the response. Full numerical records
are in `metrics/rebuttal/a2/`. For the mixture, times are 0.3, 0.5, 0.7, 0.9 and
$k=0,8,16,32,64,96,128,192$. Across all 32 settings, the measured fraction of
residual energy differs from the predicted fraction by less than 0.7 percentage
points. For example:

| $k$ | $t=0.5$: measured / predicted (%) | $t=0.9$: measured / predicted (%) |
|---:|---:|---:|
| 0 | 100.1 / 100 | 100.3 / 100 |
| 8 | 25.8 / 25.7 | 87.4 / 87.1 |
| 32 | 2.06 / 2.02 | 54.5 / 54.2 |
| 64 | 0.25 / 0.20 | 22.3 / 22.1 |
| 128 | 0.07 / <0.01 | 1.85 / 1.48 |

For the prescribed spectra, errors decrease linearly with additional width
until the residual rank is reached; all reported cells with $k\ge r$ have error
below $10^{-5}$. CPU verification checks these claims directly from the records.
Source and data checksums are in `REBUTTAL_MANIFEST.json`, and
`configs/rebuttal_sources.json` identifies unchanged original scientific sources
and the small portability changes made for this release.

## Rerun the synthetic experiments

Install a suitable PyTorch build (CUDA for the full sweeps). The full commands
are generated by the portable launcher; `--dry-run` only prints them.

```bash
python synthetic_width/run.py --family gmm --output work/synthetic/gmm --dry-run
python synthetic_width/run.py --family gmm --output work/synthetic/gmm
python synthetic_width/run.py --family binary --output work/synthetic/binary
```

The mixture sweep uses the original `train_resume.py`, which atomically saves
model, optimizer, sampling RNG, step and learning curve every 5,000 steps;
restarting the same command resumes that state. The binary sweep uses the
original `train.py`. Both save final model weights and JSON results. Full GPU
sweeps are substantial computations; they are distinct from CPU reproduction
of the supplied tables. To inspect an individual condition:

```bash
python synthetic_width/spectrum.py --dist gmm \
  --gmm_path synthetic_width/data/patch_gmm_K128.pt --t .9 \
  --out work/synthetic/spectrum-t0.9.pt
python synthetic_width/train_resume.py --dist gmm \
  --gmm_path synthetic_width/data/patch_gmm_K128.pt --t .9 \
  --spectrum work/synthetic/spectrum-t0.9.pt --arch constrained --extra 64 \
  --target dstar --affine_init --g_width 512 --adam_eps 1e-12 \
  --out work/synthetic/gmm-t0.9-k64.json
```

These released comparisons use only the constrained architecture. The source
also retains a free-feature model for separate diagnostics; the conditional
theorem's spectral-tail lower bound applies to representations satisfying
Assumption 1.
