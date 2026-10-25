# Default node-local training cache

New project ImageNet128/256 width and EDM training launches enable the local
cache by default (`JIT_LOCAL_CACHE=1`). The fresh width-64 training run already
uses its own frozen implementation of the same cache. Existing submitted Slurm
scripts and existing frozen pipeline directories keep their recorded code;
these bundle changes apply when a new pipeline snapshot is created.

The cache normally lives at `/scratch/$USER/epredbad` on the assigned compute
node, with `/tmp/$USER/epredbad` as a fallback when `/scratch` is absent. This
is disposable node storage, separate from shared `/hpc/group` storage, CPU RAM,
and GPU VRAM. A different node may require another cache build. Training
checkpoints, logs, and results continue to use their persistent shared paths.

The cache contains lossless uint8 RGB center crops, using the run's frozen
crop implementation. Image order, labels, per-sample random horizontal flips,
and random-number consumption are preserved. Supported transforms are the
original center crop, RandomHorizontalFlip, and PILToTensor sequence. Other
augmentation protocols require disabling this adapter or a separately validated
cache implementation.

All 1,281,167 images occupy about 59 GiB at 128 resolution and 235 GiB at 256.
Each resolution/crop implementation has a separate cache. Completed compatible
caches are reused by jobs on the same node; a writer lock serializes setup
across ranks and jobs. Interrupted builds resume from verified chunk markers.
Fresh builds verify image count, class count, ordering, labels, and sampled pixel
equality before marking the cache ready. Datasets are assumed immutable: replace
or remove the cache if the source images change in place.

Environment overrides:

- `JIT_LOCAL_CACHE=0`: read the original images directly.
- `JIT_CACHE_ROOT=/local/path`: choose another cache directory.
- `JIT_CACHE_WORKERS=32`: change the number of I/O/decode threads used for setup.

Launcher use (also works as the script passed to torchrun):

```bash
python /hpc/group/chenglab/gk149/epredbad/scripts/training_cache/launch.py \
  --source /hpc/group/chenglab/datasets/imagenet \
  --crop-file /path/to/frozen/code/util/crop.py --size 128 \
  --report /shared/run/cache-report.json -- /path/to/train.py
```

The report records whether caching was enabled, the source/cache identity, and
adapter code hashes. The ImageNet256 width and EDM bundles include frozen copies
of this package in their source manifests. Model, loss, and optimizer source
snapshots are unchanged.

Validation: `python scripts/test_training_cache.py` from the project root checks
both resolutions, RGB/grayscale/CMYK conversion, original ordering and labels,
random flips and RNG state, filtering, concurrent setup, interrupted-build
integrity, default launcher behavior, and the opt-out.
