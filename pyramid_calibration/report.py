"""CPU checks and tables; no model weights, images, or PyTorch import required."""
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .common import CONFIG, MEASUREMENTS, REPO, TEST_TIMES, TIMES, digest, key, read, verify_source


def verify():
    verify_source()
    expected = read(CONFIG/'assets.json')['data']['cohorts_sha256']
    with gzip.open(CONFIG/'cohorts.json.gz', 'rb') as f:
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest() == expected
    cohort = json.loads(raw)
    for split, count, seed in [('fit', 10000, 81000000), ('test', 2000, 82000000), ('generation', 50000, 0)]:
        records = cohort[split]
        assert len(records) == count
        assert [r['sample_id'] for r in records] == list(range(count))
        assert [r['noise_seed'] for r in records] == list(range(seed, seed + count))
        assert len({r['id'] for r in records}) == count
    assert len({r['image_id'] for r in cohort['fit']}) == 10000
    assert len({r['image_id'] for r in cohort['test']}) == 2000
    assert not {r['image_id'] for r in cohort['fit']} & {r['image_id'] for r in cohort['test']}
    assert len(cohort['reference']) == len({r['id'] for r in cohort['reference']}) == 40504
    for t in TIMES:
        path = MEASUREMENTS/'fit'/key(t)
        record = read(Path(str(path)+'.json'))
        assert record['complete'] and record['images'] == 10000 and record['t'] == t
        with np.load(str(path)+'.npz') as values:
            np.testing.assert_array_equal(values['ids'], np.arange(10000))
            for branch in ['conditional', 'negative']:
                ar = values[branch]
                assert ar.shape == (10000,) and np.isfinite(ar).all() and (ar >= 0).all()
                v = float(ar.mean())
                np.testing.assert_allclose(v, record[branch]['vbar'], rtol=1e-13)
                sigma = record[branch]['effective_sigma']
                # sigma was evaluated on the exact FP32 CUDA Euler grid.
                np.testing.assert_allclose(sigma, 2*record['source_time']/(1-record['source_time']), rtol=2e-7)
                np.testing.assert_allclose(v/(v+sigma*sigma), record[branch]['coefficient'], rtol=1e-13)
    for t in TEST_TIMES:
        record = read(MEASUREMENTS/'denoising'/f'{key(t)}.json')
        assert record['complete'] and record['images'] == 2000 and record['t'] == t
        with np.load(MEASUREMENTS/'denoising'/f'{key(t)}.npz') as values:
            np.testing.assert_array_equal(values['ids'], np.arange(2000))
            for name, mean in record['mse'].items():
                assert np.isfinite(values[name]).all() and (values[name] >= 0).all()
                np.testing.assert_allclose(values[name].mean(), mean, rtol=1e-13)
            for name, stats in record['paired_improvement'].items():
                baseline = values[name.split('_')[0]+'_baseline']
                difference = baseline-values[name]
                actual = [difference.mean(), difference.std(ddof=1)/math.sqrt(2000),
                          100*difference.mean()/baseline.mean()]
                np.testing.assert_allclose(actual, [stats[k] for k in ['mean', 'standard_error', 'relative_percent']], rtol=1e-12)
    rows = np.load(MEASUREMENTS/'encoder_rows.npy')
    assert rows.shape == (1024, 3072) and rows.dtype == np.float64
    np.testing.assert_allclose(rows@rows.T, np.eye(1024), atol=1e-10, rtol=0)
    assert digest(MEASUREMENTS/'encoder_rows.npy') == read(MEASUREMENTS/'encoder.json')['rows_sha256']
    for count in [2000, 10000]:
        audit = read(MEASUREMENTS/'audits'/f'FID{count}_RECOMPUTE_AUDIT.json')
        assert audit['passed'] and audit['cohort_sha256'] == expected
        for variant, value in audit['fids'].items():
            metric = read(MEASUREMENTS/'fid'/f'{variant}-{count}.json')
            assert metric['complete'] and metric['generated_images'] == count
            assert metric['reference_sha256'] == audit['reference_sha256']
            np.testing.assert_allclose(metric['fid'], value, rtol=1e-12)
    parity = read(CONFIG/'source_parity.json')['scientific_functions_ast_sha256']
    definitions = {n.name: n for n in ast.parse((REPO/'pyramid_calibration/worker.py').read_text()).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name, sha in parity.items():
        assert hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest() == sha, name
    status = read(MEASUREMENTS/'status.json')
    assert status['fid10k_blocks'][0]['fids'] == read(MEASUREMENTS/'audits/FID10000_RECOMPUTE_AUDIT.json')['fids']
    for b, block in enumerate(status['fid10k_blocks']):
        assert block['start'] == b*10000 and block['stop_exclusive'] == (b+1)*10000
        if block['status'] == 'audited':
            audit = read(MEASUREMENTS/'blocks'/f'block-{b}.json')
            assert audit['passed'] and audit['fids'] == block['fids']
            assert audit['start'] == block['start'] and audit['stop_exclusive'] == block['stop_exclusive']
            assert audit['reference_sha256'] == read(MEASUREMENTS/'audits/REFERENCE_AUDIT.json')['reference_sha256']
            detail_path = MEASUREMENTS/'audits'/f'FID10K_BLOCK{b+1}_AUDIT.json'
            if detail_path.exists():
                assert digest(detail_path) == audit['detail_sha256']
                detail = read(detail_path)
                assert detail['passed'] and detail['fids'] == audit['fids']
        else:
            assert block['status'] == 'pending' and 'fids' not in block
    pooled = status['pooled_fid30k']
    assert pooled['start'] == 0 and pooled['stop_exclusive'] == 30000
    if pooled['status'] == 'audited':
        audit = read(MEASUREMENTS/'pooled-30000.json')
        assert audit['passed'] and audit['count'] == 30000 and audit['fids'] == pooled['fids']
        assert all(b['status'] == 'audited' for b in status['fid10k_blocks'])
        assert audit['reference_sha256'] == read(MEASUREMENTS/'audits/REFERENCE_AUDIT.json')['reference_sha256']
        detail_path = MEASUREMENTS/'audits/FID30K_RECOMPUTE_AUDIT.json'
        assert digest(detail_path) == audit['detail_sha256']
        detail = read(detail_path)
        assert detail['passed'] and detail['fids'] == audit['fids']
        for b, block in enumerate(status['fid10k_blocks']):
            source = audit['source_blocks'][str(b)]
            assert source['metric_sha256'] == digest(MEASUREMENTS/'blocks'/f'block-{b}.json')
            assert source['fids'] == block['fids']
        for variant, fid in pooled['fids'].items():
            np.testing.assert_allclose(fid, detail['variants'][variant]['symmetric_psd_crosscheck'], atol=1e-4, rtol=0)
    else:
        assert pooled['status'] == 'pending' and 'fids' not in pooled
    return {'passed': True, 'fit_times': len(TIMES), 'fit_images_per_time': 10000,
            'denoising_times': len(TEST_TIMES), 'held_out_images_per_time': 2000,
            'original_scientific_definitions_unchanged': len(parity),
            'fid_validation': 'Published scores match the retained recomputation audits; raw images/features are external.',
            'status': status}


def render(output, measurements=MEASUREMENTS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# Pyramid-JiT512 calibration', '',
             'Frozen released model; native Euler50/APG sampling; COCO2014 Inception FID.', '',
             '| Images (prefix) | Baseline | Half-late | Full calibration |',
             '|---:|---:|---:|---:|']
    fid_rows = []
    for count in [2000, 10000, 50000]:
        if not all((measurements/'fid'/f'{variant}-{count}.json').exists()
                   for variant in ['baseline', 'calibrated', 'late_half']):
            continue
        row = {'images': count}
        for variant in ['baseline', 'late_half', 'calibrated']:
            path = measurements/'fid'/f'{variant}-{count}.json'
            row[variant] = read(path)['fid'] if path.exists() else None
        lines.append('| '+str(count)+' | '+' | '.join(f'**{row[v]:.8f}**' if v == 'calibrated' else f'{row[v]:.8f}'
                     for v in ['baseline', 'late_half', 'calibrated'])+' |')
        fid_rows.append(row)
    lines += ['', 'The 2K and first 10K results are nested. Three disjoint 10K caption/noise blocks',
              'are defined at indices [0,10000), [10000,20000), and [20000,30000).',
              'They are not seed-only repeats; image IDs can overlap and all blocks belong to the 50K bank.', '',
              '| Clean time | Baseline conditional MSE | Half-late | Full | Full reduction (%) |',
              '|---:|---:|---:|---:|---:|']
    denoising_rows = []
    for t in TEST_TIMES:
        path = measurements/'denoising'/f'{key(t)}.json'
        if not path.exists():
            continue
        data = read(path)
        row = {'t': t,
               **{f'{branch}_{variant}': data['mse'][f'{branch}_{variant}']
                  for branch in ['conditional', 'negative']
                  for variant in ['baseline', 'late_half', 'calibrated']},
               'full_reduction_percent': data['paired_improvement']['conditional_calibrated']['relative_percent']}
        denoising_rows.append(row)
        lines.append(f'| {t:g} | {row["conditional_baseline"]:.8g} | {row["conditional_late_half"]:.8g} | '
                     f'{row["conditional_calibrated"]:.8g} | {row["full_reduction_percent"]:.6g} |')
    status_file = measurements/'status.json'
    comparison_rows = []
    if status_file.exists():
        status = read(status_file)
        lines += ['', '## Three FID-10K blocks and pooled FID-30K', '',
                  '| Evaluation | Uncalibrated | Half-late calibration | Full calibration | ΔFID, full |',
                  '|---|---:|---:|---:|---:|']
        evaluations = [(f'FID-10K, block {i+1}', b) for i,b in enumerate(status['fid10k_blocks'])]
        evaluations.append(('Pooled FID-30K', status['pooled_fid30k']))
        for label, evaluation in evaluations:
            fids = evaluation.get('fids', {})
            values = [fids.get(v) for v in ['baseline','late_half','calibrated']]
            values += [fids['calibrated']-fids['baseline'] if fids else None]
            cells = ['Pending' if v is None else (f'**{v:.2f}**' if i == 2 else (f'{v:.2f}' if i<3 else f'{v:+.2f}')) for i,v in enumerate(values)]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
            comparison_rows.append(dict(zip(['evaluation','status','baseline','late_half','calibrated',
                                             'full_minus_baseline'],
                                            [label,evaluation['status'],*values])))
        lines += ['', 'Lower FID is better; ΔFID is full calibration minus uncalibrated.',
                  'Half-late uses full correction before clean time 0.9 and half strength thereafter.',
                  'Pooled FID-30K recomputes statistics over the union of the three blocks, using the same 40,504-image reference.',
                  'It is not an average of block FIDs or an additional independent evaluation.']
    lines += ['', 'See docs/pyramid_calibration.md for equations, precision, data selection, reproduction commands, and limitations.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for name, records in [('fid', fid_rows), ('denoising', denoising_rows), ('fid-comparison', comparison_rows)]:
        if records:
            with (output/f'{name}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return output/'REPORT.md'


def render_workspace(root):
    return render(root/'report', measurements=root/'output')
