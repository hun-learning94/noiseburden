"""Independently check revision measurements and manuscript outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'metrics/revision'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inputs():
    expected=json.loads((ROOT/'REVISION_MANIFEST.json').read_text())['sha256']
    for p,digest in expected.items():assert sha(ROOT/p)==digest,p
    data=json.loads((DATA/'section42_figures_provenance.json').read_text())
    counts={'i128_epoch200_cases':len(data['rows']),'i128_loss_cells':0,'i256_loss_cells':0,'recovery_cells':0}
    base=ROOT/'metrics/native/native-i128_eps192-per-image.npz'
    for r in data['gains']:
        family,w=r['family'],r['width'];t=r['t']
        if family=='noise':p=ROOT/f'metrics/native/native-i128_eps{w}-per-image.npz'
        else:
            name=f'i128_eps{w}'+(f'_d{w//16}' if family=='deep' else '')
            p=DATA/f'controls128/{family}/{name}/native-per-image.npz'
        with np.load(base) as a,np.load(p) as b:
            x=a[f'x_{t}'];y=b[f'x_{t}'];assert x.shape==y.shape==(2000,)
            gain=100*np.mean(x-y)/x.mean()
            np.testing.assert_allclose(gain,r['mean'],rtol=1e-12)
        counts['i128_loss_cells']+=1
    grid=json.loads((DATA/'width_controls_i256_ep120_provenance.json').read_text())
    for row in grid['rows']:
        if row['fid'] is None:continue
        assert row['epoch']==120 and row['checkpoint_sha256']
        for p in (DATA/'controls256').glob(row['case']+'-t*.json'):
            d=json.loads(p.read_text());assert d['complete'] and d['epoch120_provenance']['completed_epochs']==120
            assert d['checkpoint_sha256']==row['checkpoint_sha256']
            x=np.asarray(d['native_clean_mse_by_image']);assert x.shape==(2000,) and np.isfinite(x).all()
            np.testing.assert_allclose(x.mean(),d['native_clean_mse'],rtol=1e-12)
            counts['i256_loss_cells']+=1
    for r in grid['gains']:
        base=json.loads((DATA/'controls256'/f'jit256w768-eps-unrestricted-ep120-t{r["t"]:g}.json').read_text())
        model=json.loads((DATA/'controls256'/f'{r["case"]}-t{r["t"]:g}.json').read_text())
        x=np.asarray(base['native_clean_mse_by_image']);y=np.asarray(model['native_clean_mse_by_image'])
        np.testing.assert_allclose(100*np.mean(x-y)/x.mean(),r['mean'],rtol=1e-12,atol=1e-12)
    for p in (DATA/'recovery').rglob('*.json'):
        d=json.loads(p.read_text())
        if not d.get('complete') or 'native_clean_mse' not in d:continue
        if 'native_clean_mse_by_image' in d:
            np.testing.assert_allclose(np.mean(d['native_clean_mse_by_image']),d['native_clean_mse'],rtol=1e-12)
        for v in d.get('probes',{'probe':d.get('probe',{})}).values():
            if 'mse_by_image' not in v:continue
            np.testing.assert_allclose(np.mean(v['mse_by_image']),v['mse'],rtol=1e-12)
            np.testing.assert_allclose(100*(1-v['mse']/v['variance']),v['r2_percent'],rtol=1e-10,atol=1e-10)
            counts['recovery_cells']+=1
    counts['pending_i256_models']=sum(r['fid'] is None for r in grid['rows'])
    return counts

def outputs(output,strict=False):
    out=Path(output);tables={};figures={}
    # Whitespace in LaTeX formatting does not change table content.
    for n in range(1,8):
        name=f'table_{n:02d}.tex'
        tables[str(n)]=(out/'tables'/name).read_text().split()==(ROOT/'reference/revision/tables'/name).read_text().split()
    assert all(tables.values()),tables
    for number,name in json.loads((ROOT/'configs/revision_figures.json').read_text()).items():
        figures[number]=sha(out/f'figures/{name}.png')==sha(ROOT/f'reference/revision/figures/{name}.png')
    if strict:assert all(figures.values()),figures
    return dict(tables_match_manuscript=tables,figure_pngs_match_manuscript=figures)
