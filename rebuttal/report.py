"""Check supplied A1/A2 measurements and render their tables without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'REBUTTAL_MANIFEST.json').read_text())['sha256']
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f'Changed release file: {name}'
    comparison = json.loads((ROOT / 'metrics/rebuttal/a1/comparison.json').read_text())
    a1 = []
    for row in comparison['rows']:
        with np.load(ROOT / row['native_per_image']) as arrays:
            losses = []
            for t in [.1, .5, .9]:
                clean, noise = arrays[f'x_{t}'], arrays[f'eps_{t}']
                assert clean.shape == noise.shape == (2000,)
                assert np.isfinite(clean).all() and (clean >= 0).all()
                np.testing.assert_allclose(clean, ((1-t)/t)**2*noise, atol=5e-5, rtol=1e-4)
                losses.append(float(clean.mean()))
        a1.append(dict(row, mse=losses))
    gmm = [json.loads(p.read_text()) for p in sorted((ROOT / 'metrics/rebuttal/a2/gmm').glob('*.json'))]
    binary = [json.loads(p.read_text()) for p in sorted((ROOT / 'metrics/rebuttal/a2/binary').glob('*.json'))]
    assert len(gmm) == 32 and len(binary) == 20
    assert {(d['config']['t'], d['k']) for d in gmm} == {
        (t, k) for t in [.3, .5, .7, .9] for k in [0, 8, 16, 32, 64, 96, 128, 192]}
    discrepancies = []
    for d in gmm + binary:
        assert d['config']['arch'] == 'constrained' and d['config']['target'] == 'dstar'
        assert d['config']['affine_init'] and d['probe_r2_input'] > 99.999
        assert d['approx_refit'] >= 0 and d['theory'] >= 0 and d['theory_resid'] > 0
        if d in gmm:
            discrepancies.append(100 * abs(d['approx_refit'] - d['theory']) / d['theory_resid'])
        else:
            rank = int(d['config']['scales'].split(':')[1])
            np.testing.assert_allclose(d['theory'], max(rank-d['k'], 0)*d['theory_resid']/rank, atol=1e-12)
            if d['k'] >= rank:
                assert d['approx_refit'] < 1e-5
    maximum = max(discrepancies)
    assert maximum < .7
    return a1, gmm, binary, dict(files=len(manifest), a1_models=len(a1),
        gmm_settings=len(gmm), binary_settings=len(binary), max_gmm_gap_percentage_points=maximum)


def render(a1, gmm, binary):
    text = ['# A1: matched ImageNet128 epoch-200 comparison', '',
        '| Noise predictor | Width | Blocks | Parameters | MSE t=0.1 | MSE t=0.5 | MSE t=0.9 | FID-10K |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in a1:
        values = ' | '.join(f'{x:.6f}' for x in r['mse'])
        text.append(f"| {r['label']} | {r['width']} | {r['depth']} | {r['parameters']/1e6:.2f}M | {values} | {r['fid10k']:.2f} |")
    text += ['', '# A2: exact-denoiser GMM comparison', '',
        'Errors are percentages of the estimated residual energy. Entries are measured / predicted.', '',
        '| k | t=0.3 | t=0.5 | t=0.7 | t=0.9 |', '|---:|---:|---:|---:|---:|']
    lookup = {(d['config']['t'], d['k']): d for d in gmm}
    def percentage(v):
        return '<0.01' if v < .01 else f'{v:.2f}'
    for k in [0, 8, 16, 32, 64, 96, 128, 192]:
        cells = []
        for t in [.3, .5, .7, .9]:
            d = lookup[t, k]
            cells.append(' / '.join(percentage(100*d[key]/d['theory_resid']) for key in ['approx_refit', 'theory']))
        text.append(f"| {k} | " + ' | '.join(cells) + ' |')
    text += ['', '# A2: prescribed binary spectra at t=0.5', '',
        'Errors are per-patch sums. Entries are measured / predicted; ≈0 means below 1e-5.', '',
        '| k | Rank 4 | Rank 8 | Rank 16 | Rank 32 |', '|---:|---:|---:|---:|---:|']
    lookup = {(int(d['config']['scales'].split(':')[1]), d['k']): d for d in binary}
    for k in [0, 4, 8, 16, 32]:
        cells = []
        for rank in [4, 8, 16, 32]:
            d = lookup[rank, k]
            measured = '≈0' if d['approx_refit'] < 1e-5 else f"{d['approx_refit']:.4f}"
            cells.append(f"{measured} / {d['theory']:.4f}")
        text.append(f"| {k} | " + ' | '.join(cells) + ' |')
    return '\n'.join(text) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['verify', 'report'])
    p.add_argument('--output', type=Path, default=Path('artifacts/rebuttal'))
    args = p.parse_args()
    a1, gmm, binary, result = verify()
    if args.mode == 'report':
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'TABLES.md').write_text(render(a1, gmm, binary))
        (args.output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
