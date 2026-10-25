"""Integrity, independent metric consistency, and manuscript-output checks."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inputs(metrics=None):
    data=Path(metrics).resolve() if metrics else ROOT/'metrics'
    base=data if metrics else ROOT
    expected=json.loads((base/'MANIFEST.json').read_text())
    for path,digest in expected['sha256'].items():
        assert sha(base/path)==digest,f'Input checksum mismatch: {path}'
    cases=json.loads(((data/'plot_cases.json') if metrics else ROOT/'configs/plot_cases.json').read_text())['cases']
    cells=0
    for c in cases:
        name=data/f'native/native-{c["id"]}'
        records=json.loads(name.with_suffix('.json').read_text())['records']
        with np.load(str(name)+'-per-image.npz') as arrays:
            for r in records:
                values=arrays[f'x_{r["t"]}'];assert values.shape==(2000,)
                assert np.isclose(values.mean(),r['clean_mse']['mean'],rtol=1e-12,atol=1e-12)
                cells+=1
    assert cells==182
    noise=list(csv.DictReader((data/'noise_layerwise_200_ema1.csv').open()))
    assert len(noise)==78
    for r in noise:
        records=json.loads(((base/r['source']) if metrics else ROOT/r['source']).read_text())['records']
        actual=next(x for x in records if x['layer']==int(r['layer']))
        assert float(r['noise_recovery_score'])==actual['noise_recovery_score']<1/6
    probes=list(csv.DictReader((data/'affine_input_recovery_200_ema1.csv').open()))
    assert len(probes)==16
    for r in probes:
        actual=next(x for x in json.loads(((base/r['source']) if metrics else ROOT/r['source']).read_text())['records'] if x['test_images']==2000)
        assert float(r['native_clean_mse'])==actual['native_clean_mse']
        assert float(r['input_r2_percent'])==actual['input']['r2_percent']
    spectral_cells=0
    for res in [128,256]:
        records=json.loads((data/f'spectra{res}/summary.json').read_text())['records']
        with np.load(data/f'spectra{res}/spectra.npz') as arrays:
            for r in records:
                if r['teacher']!=f'adm{res}' or r['patch']!=0:continue
                values=np.sort(np.maximum(arrays[f'adm{res}_t{r["t"]}_p0'],0))[::-1]
                assert np.isclose(values.mean(),r['energy_per_coordinate'],rtol=1e-9,atol=1e-12)
                count=int(np.searchsorted(np.cumsum(values),.95*values.sum())+1)
                assert count==r['k95'];spectral_cells+=1
    assert spectral_cells==14
    return dict(input_files=len(expected['sha256']),native_cells=cells,noise_probe_cells=78,input_probe_cases=16,independent_adm_spectra=14)

def outputs(output,strict=False):
    output=Path(output);tables={};figures={}
    for number in range(1,7):
        name=f'table_{number:02d}.tex';tables[str(number)]=sha(output/'tables'/name)==sha(ROOT/'reference/tables'/name)
    assert all(tables.values()),f'Table mismatch: {tables}'
    mapping=json.loads((ROOT/'configs/figures.json').read_text())
    for number,name in mapping.items():figures[number]=sha(output/'figures'/f'{name}.png')==sha(ROOT/'reference/figures'/f'{name}.png')
    if strict:assert all(figures.values()),f'Pixel mismatch (use the pinned rendering environment): {figures}'
    return dict(tables_match_manuscript=tables,figure_pngs_match_manuscript=figures)
