# Source attribution

The training base is bundled from the earlier anonymous training release;
distributed file hashes identify its exact source version. It derives from
Tianhong Li's JiT implementation. Its MIT
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

Public DiT inference uses the original Facebook Research DiT implementation;
its license is retained in `revision/frozen/probe/DiT-LICENSE.txt` and the FID
worker distribution. The source snapshot includes the original diffusion
schedule/parsing code and its headers.

U-ViT, CompVis latent-diffusion, and taming-transformers sources are pinned in
`configs/public_sources.json`. Their inference source and original licenses are
retained under `revision/frozen/latent/code/`. Official pretrained weights are
external inputs; their URLs and hashes are in `configs/public_assets.json`.

Revision source copies neutralize collaborator names and machine paths. The
original source hashes and distributed source hashes are distinguished where
applicable; model definitions and numerical arithmetic are retained. Third-party
authorship and license notices are preserved.
