# Matched public-model FID-10K

Inference uses the provided public checkpoint identities and native noise schedules.
All probes use disjoint 384 fitting, 128 tuning, and 2000 test validation images,
FP32 models with TF32 disabled, and FP64 shared-intercept affine fitting.
Public latent models recover noisy latents; their losses are not RGB losses.
Sampling uses adapted Heun-50, no guidance, 1000 balanced classes, 10 images per
class, per-image CUDA seed equal to sample index, and the same supplied FID
reference statistics. The final Heun step is Euler (99 network evaluations).
See docs/revision.md for representation choices, time mappings, finite maximum
noise initialization, precision guards, and interpretation qualifications.
