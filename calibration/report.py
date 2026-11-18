"""Verify and render the released scalar-calibration results without a GPU."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'metrics/calibration'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = read(ROOT/'CALIBRATION_MANIFEST.json')
    for name, expected in manifest['sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Calibration release file changed: {name}')
    protocol = read(ROOT/'configs/calibration/protocol.json')
    rows = np.load(DATA/'encoder_rows.npy')
    assert rows.shape == (128,768)
    np.testing.assert_allclose(rows@rows.T, np.eye(128), atol=1e-10, rtol=0)
    fits = sorted((DATA/'fit').glob('*.json'))
    assert len(fits) == 51
    for path in fits:
        d = read(path)
        assert d['images']==10000 and d['complete']
        with np.load(path.with_suffix('.npz')) as arrays:
            for mode in ['unguided','guided']:
                moment = arrays[mode]
                assert moment.shape == (10000,) and np.isfinite(moment).all()
                v = moment.mean()/(640/768)
                expected = d['coefficients'][mode]
                np.testing.assert_allclose(v, expected['vbar'], atol=1e-15, rtol=1e-12)
                sigma = (1-d['t'])/d['t']
                np.testing.assert_allclose(v/(v+sigma*sigma), expected['c_t'], atol=1e-15, rtol=1e-12)
    tests = sorted((DATA/'test').glob('*.json'))
    assert len(tests)==11
    for path in tests:
        d = read(path)
        assert d['images']==2000 and d['complete']
        with np.load(path.with_suffix('.npz')) as arrays:
            for mode in ['unguided','guided']:
                base = arrays[mode+'__baseline']
                for method in ['baseline','full']:
                    x = arrays[mode+'__'+method]
                    assert x.shape==(2000,) and np.isfinite(x).all()
                    expected = d['records'][mode][method]
                    np.testing.assert_allclose(x.mean(), expected['mse'], rtol=1e-12)
                    np.testing.assert_allclose(100*np.mean(base-x)/base.mean(),
                        expected['reduction_percent'], rtol=1e-10, atol=1e-12)
    fids = [read(p) for p in sorted((DATA/'fid').glob('*.json'))]
    assert len(fids)==9
    assert {(d['endpoint_clamp'],d['method']) for d in fids} == {
        (floor,method) for floor in protocol['floors'] for method in protocol['methods']}
    for d in fids:
        assert d['complete'] and d['images']==10000 and d['cfg']==3.
        assert d['reference_sha256']==protocol['fid_reference_sha256']
        assert np.isfinite(d['fid'])
    samples = read(ROOT/'configs/calibration/samples.json')
    for split,count in [('train',10000),('validation',2000)]:
        cohort=[r for r in samples if r['split']==split]
        assert len(cohort)==len({r['filename'] for r in cohort})==count
        assert np.array_equal(np.bincount([r['label'] for r in cohort]),np.full(1000,count//1000))
    return {'files_verified':len(manifest['sha256']), 'fit_times':51,
            'held_out_denoising_times':11, 'fid10k_cells':9,
            'image_identities':12000, 'gpu_inference_rerun':False}


def write_csv(path, records):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def render(output):
    checks=verify()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    denoising=[];coefficients=[]
    for path in sorted((DATA/'test').glob('*.json')):
        d=read(path)
        for mode in ['unguided','guided']:
            b=d['records'][mode]['baseline'];c=d['records'][mode]['full']
            denoising.append({'t':d['t'],'guidance':mode,'baseline_mse':b['mse'],
                'calibrated_mse':c['mse'],'reduction_percent':c['reduction_percent']})
    for path in sorted((DATA/'fit').glob('*.json')):
        d=read(path)
        for mode,values in d['coefficients'].items():
            coefficients.append({'t':d['t'],'guidance':mode,**values})
    fids=[read(p) for p in (DATA/'fid').glob('*.json')]
    fids=sorted(fids,key=lambda d:(-d['endpoint_clamp'],['baseline','full','late_half'].index(d['method'])))
    write_csv(output/'denoising.csv',denoising)
    write_csv(output/'coefficients.csv',coefficients)
    write_csv(output/'fid10k.csv',[{k:d[k] for k in ['endpoint_clamp','method','fid','images']} for d in fids])
    text=['# Scalar calibration of the released JiT-B/16 image predictor','',
        'These tables reproduce saved measurements; no backbone training or GPU inference is performed.',
        'The checkpoint, calibration cohort, guidance, precision, and FID reference are fixed across the comparison.',
        'See [the full protocol](../../docs/calibration.md) for preparation and evaluation commands.','',
        '## Held-out denoising (guided, full calibration)','',
        '| Time | Baseline MSE | Calibrated MSE | Reduction |',
        '|---:|---:|---:|---:|']
    for d in denoising:
        if d['guidance']=='guided':
            text.append(f"| {d['t']:g} | {d['baseline_mse']:.8f} | {d['calibrated_mse']:.8f} | {d['reduction_percent']:.4f}% |")
    text += ['', '## Paired FID-10K','', '| Denominator floor | Baseline | Full | Half late |',
             '|---:|---:|---:|---:|']
    for floor in [.05,.02,0.]:
        values={d['method']:d['fid'] for d in fids if d['endpoint_clamp']==floor}
        text.append(f"| {floor:g} | {values['baseline']:.5f} | {values['full']:.5f} | {values['late_half']:.5f} |")
    text += ['', 'The half-late schedule uses full strength before t=0.9 and half strength thereafter.',
        'The FID-10K cohorts include exploratory FID-2K screening samples; they are not independent confirmation cohorts.',
        'Full calibration lowers held-out denoising MSE; its generation effect depends on endpoint handling. All three clamp settings are retained.',
        '', '![Calibration coefficients, denoising gains, and paired FID](calibration.png)', '']
    (output/'REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(10,2.7),layout='constrained')
    for mode,label in [('unguided','CFG 1'),('guided','Interval CFG 3')]:
        c=[r for r in coefficients if r['guidance']==mode]
        d=[r for r in denoising if r['guidance']==mode]
        axs[0].plot([r['t'] for r in c],[r['c_t'] for r in c],label=label)
        axs[1].plot([r['t'] for r in d],[r['reduction_percent'] for r in d],label=label)
    axs[0].set(xlabel='Time t',ylabel='Fitted scalar coefficient',title='(a) Calibration')
    axs[1].set(xlabel='Time t',ylabel='MSE reduction (%)',title='(b) Held-out denoising')
    axs[0].legend(frameon=False,fontsize=8)
    x=np.arange(3)
    for i,(method,label) in enumerate([('baseline','Baseline'),('full','Full'),('late_half','Half late')]):
        y=[next(d['fid'] for d in fids if d['method']==method and d['endpoint_clamp']==floor) for floor in [.05,.02,0.]]
        axs[2].bar(x+(i-1)*.25,y,width=.25,label=label)
    axs[2].set(xticks=x,xticklabels=['0.05','0.02','0'],xlabel='Denominator floor',ylabel='FID-10K',title='(c) Generation')
    axs[2].set_ylim(0,10.5)
    axs[2].legend(frameon=False,fontsize=7,ncol=3,loc='upper center',handlelength=1.4,columnspacing=.8)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'calibration.png',dpi=180)
    fig.savefig(output/'calibration.pdf')
    plt.close(fig)
    (output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks
