# Source attribution

The training base is pinned from `kilonova801/jit` at commit
`25667dd44e24e619f2e3300e4b16dbe4e91680ad`, the source of the earlier anonymous
training release. It derives from Tianhong Li's JiT implementation. Its MIT
license is retained in `LICENSE` and in the training source snapshots.

Each run snapshot overlays the files actually used for that checkpoint,
verified against the recorded file hashes. The supplied base and current
run snapshots are distinguished in `training/source_manifest.json`.

ADM inference uses OpenAI's `guided-diffusion` source at commit
`22e0df8183507e13a7813f8d38d51b072ca1e67c`; its MIT license is retained in
the corresponding source snapshot. Official checkpoint URLs and SHA-256
identities are listed in `configs/experiments.json`.

The frozen JiT fork of `torch-fidelity` is included under
`evaluation/third_party/torch_fidelity`, with its original license. Model
weights, ImageNet image pixels, and FID reference statistics are external
inputs and are not part of this source-and-metrics distribution.
