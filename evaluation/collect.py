"""Export a complete numerical rerun for python -m noiseburden --metrics DIR."""
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
from .runtime import ROOT,digest,save

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('work/evaluation'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--generation',type=Path)
    p.add_argument('--coefficients',type=Path);p.add_argument('--reuse-saved-generation',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out==(ROOT/'metrics').resolve():p.error('Choose a new output folder; preserve the distributed paper measurements.')
    info=json.loads((a.workspace/'runtime.json').read_text());run=Path(info['run']);m=json.loads((run/'manifest.json').read_text())
    cases=m['cases'];published={c['id']:c for c in json.loads((ROOT/'configs/plot_cases.json').read_text())['cases']}
    same=all(c['checkpoint_sha256']==published[c['id']]['checkpoint_sha256'] for c in cases)
    if a.reuse_saved_generation and not same:p.error('Saved generation/coefficient results can only be reused with the identical published checkpoints.')
    if not a.reuse_saved_generation and (not a.generation or not a.coefficients):
        p.error('Provide --generation and --coefficients for a fresh run, or explicitly --reuse-saved-generation for original checkpoints.')
    for name in ['native','input','noise','spectra128','spectra256','learned']:(out/name).mkdir(parents=True,exist_ok=True)
    inputs=[]
    for c in cases:
        stem=f'native-{c["id"]}'
        for ext in ['.json','-per-image.npz']:shutil.copy2(run/'results/native'/(stem+ext),out/'native'/(stem+ext))
        if c['kind']=='width':
            src=run/f'results/input/{c["id"]}.json';target=out/'input'/src.name;shutil.copy2(src,target)
            r=next(r for r in json.loads(src.read_text())['records'] if r['test_images']==2000)
            inputs.append(dict(resolution=c['image_size'],prediction='Noise' if c['pred']=='eps' else 'Image',width=c['width'],
              native_clean_mse=r['native_clean_mse'],input_r2_percent=r['input']['r2_percent'],t=.5,
              source='input/'+src.name,source_sha256=digest(target),dataset_split='validation',test_images=2000))
    write_csv(out/'affine_input_recovery_200_ema1.csv',inputs)
    noises=[]
    for t in [.1,.5,.7,.8,.9,.95]:
        src=run/f'results/noise/noise-t{t}.json';target=out/'noise'/src.name;shutil.copy2(src,target)
        for r in json.loads(src.read_text())['records']:
            noises.append(dict(t=t,layer=r['layer'],noise_recovery_score=r['noise_recovery_score'],source='noise/'+src.name,
                               source_sha256=digest(target),dataset_split='validation',fit_images=8000,test_images=2000))
    write_csv(out/'noise_layerwise_200_ema1.csv',noises)
    spectral_hashes={};independent=[]
    for res in [128,256]:
        for name in ['summary.json','spectra.npz']:shutil.copy2(run/f'spectra{res}/results'/name,out/f'spectra{res}'/name)
        summary=json.loads((out/f'spectra{res}/summary.json').read_text());spectral_hashes[str(res)]=digest(run/f'spectra{res}/manifest.json')
        with np.load(out/f'spectra{res}/spectra.npz') as arrays:
            for r in summary['records']:
                if r['teacher']!=f'adm{res}' or r['patch']!=0:continue
                values=np.sort(np.maximum(arrays[f'adm{res}_t{r["t"]}_p0'],0))[::-1]
                assert np.isclose(values.mean(),r['energy_per_coordinate'],rtol=1e-9,atol=1e-12)
                assert int(np.searchsorted(np.cumsum(values),.95*values.sum())+1)==r['k95']
                independent.append(dict(resolution=res,t=r['t']))
    if a.reuse_saved_generation:
        for name in ['bottleneck_fid_200.csv','width_generation_200_ema1.csv']:shutil.copy2(ROOT/'metrics'/name,out/name)
        for name in ['learned-coefficients.csv','coefficient-provenance.json']:shutil.copy2(ROOT/'metrics/learned'/name,out/'learned'/name)
    else:
        bottleneck=[];width=[]
        for c in cases:
            row=json.loads((a.generation/c['id']/'metrics.json').read_text())
            assert row['checkpoint_sha256']==c['checkpoint_sha256'] and row['num_images']==10000
            value=row['metrics']['frechet_inception_distance']
            if c['kind']=='width':width.append(dict(resolution=c['image_size'],prediction='Noise' if c['pred']=='eps' else 'Image',width=c['width'],fid=value,epoch=200,ema='EMA1'))
            else:bottleneck.append(dict(case_id=c['id'],fid=value))
        write_csv(out/'bottleneck_fid_200.csv',bottleneck);write_csv(out/'width_generation_200_ema1.csv',width)
        for name in ['learned-coefficients.csv','coefficient-provenance.json']:shutil.copy2(a.coefficients/name,out/'learned'/name)
    save(out/'plot_cases.json',dict(cases=[{k:c[k] for k in ['id','kind','width','pred','image_size','patch','heads','head_dim','checkpoint_sha256'] if k in c} for c in cases],times=m['times']))
    save(out/'protocol.json',dict(complete=True,manifest_sha256=digest(run/'manifest.json'),
         historical_spectral_manifest_sha256=spectral_hashes,independent_cpu_adm_spectra=independent))
    save(out/'MANIFEST.json',dict(sha256={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name!='MANIFEST.json'}))
    print('Complete metrics exported to',out)

if __name__=='__main__':main()
