"""Workspace initialization and scoring of fixed contiguous sample blocks."""
import gzip
import math
import shutil
import sys
import numpy as np
from .common import CONFIG, MEASUREMENTS, ROOT, SOURCE, STORE, digest, read, save


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    target = ROOT/'cohorts.json'
    expected = read(CONFIG/'assets.json')['data']['cohorts_sha256']
    if not target.exists():
        with gzip.open(CONFIG/'cohorts.json.gz', 'rb') as src:
            data = src.read()
        import hashlib
        assert hashlib.sha256(data).hexdigest() == expected
        tmp = target.with_suffix('.tmp')
        tmp.write_bytes(data)
        tmp.replace(target)
    assert digest(target) == expected, 'Workspace contains a different cohort'
    protocol = read(CONFIG/'protocol.json')
    if (ROOT/'manifest.json').exists():
        assert read(ROOT/'manifest.json') == protocol, 'Workspace protocol differs'
    else:
        save(ROOT/'manifest.json', protocol)


def use_supplied_fit():
    """Import fitted scalars explicitly; never silently replace a local refit."""
    folder = STORE/'fit'
    folder.mkdir(exist_ok=True)
    for src in sorted((MEASUREMENTS/'fit').glob('*.json')):
        dest = folder/src.name
        if dest.exists():
            assert digest(dest) == digest(src), 'Use a separate workspace for a different fit'
        else:
            shutil.copy2(src, dest)
    decision = ROOT/'GENERATION_DECISION.json'
    original = read(CONFIG/'generation_decision.json')
    if decision.exists():
        assert read(decision) == original, 'Workspace has a different denoising decision'
    else:
        save(decision, original)
    save(ROOT/'SUPPLIED_FIT.json', {'coefficients': 'released 10,000-image fit',
                                  'local_refit': False, 'local_denoising_rerun': False})


def score_block(variant, start, count):
    """Use the same Inception FID implementation, selecting [start, start+count)."""
    sys.path.insert(0, str(SOURCE))
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    selected, ids = [], []
    for path in sorted((STORE/'generated'/variant/'features').glob('*.npz')):
        with np.load(path) as batch:
            mask = (batch['ids'] >= start) & (batch['ids'] < start + count)
            if mask.any():
                selected.append(batch['features'][mask])
                ids.extend(batch['ids'][mask].tolist())
    assert ids == list(range(start, start + count)), 'Incomplete, unordered or duplicate block'
    features = np.concatenate(selected).astype(np.float64)
    assert features.shape == (count, 2048) and np.isfinite(features).all()
    stat = {'mu': features.mean(0), 'sigma': np.cov(features, rowvar=False)}
    reference = STORE/'fid/reference-statistics.npz'
    with np.load(reference) as ref:
        assert int(ref['count']) == 40504
        result = fid_statistics_to_metric(stat, {'mu': ref['mu'], 'sigma': ref['sigma']}, False)
    fid = float(result['frechet_inception_distance'])
    assert math.isfinite(fid) and fid >= 0
    record = {'complete': True, 'variant': variant, 'start': start,
              'stop_exclusive': start + count, 'generated_images': count,
              'reference_images': 40504, 'fid': fid,
              'cohorts_sha256': digest(ROOT/'cohorts.json'),
              'reference_sha256': digest(reference)}
    save(STORE/'fid'/f'{variant}-block-{start}-{start+count}.json', record)
    return record
