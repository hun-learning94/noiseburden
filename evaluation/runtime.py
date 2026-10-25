"""Materialize portable paths without modifying frozen implementations."""
from pathlib import Path
import copy,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'configs/experiments.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def configure(workspace,cache,checkpoints,accept_new=False):
    workspace=Path(workspace).resolve();cache=Path(cache).resolve()
    mapping_path=Path(checkpoints).resolve();mapping=json.loads(mapping_path.read_text())
    cases=copy.deepcopy(SPEC['cases'])
    for c in cases:
        c['source']=str(ROOT/c['source'])
        if 'loader' in c:c['loader']=str(ROOT/c['loader'])
        path=Path(mapping.get(c['id'],f'checkpoints/{c["id"]}.pth')).expanduser()
        if not path.is_absolute():path=mapping_path.parent/path
        c['checkpoint']=str(path.resolve())
        if accept_new and c['id'] in mapping:
            c['published_checkpoint_sha256']=c['checkpoint_sha256']
            c['checkpoint_sha256']=digest(path)
    run=workspace/'runs/validation-replication-20261025';run.mkdir(parents=True,exist_ok=True)
    files={'native.py':run/'native.py','input_probe.py':run/'input_probe.py','noise_probe.py':run/'noise_probe.py',
           'native_core.py':workspace/'runs/denoising-probes-20261023/worker.py',
           'affine_core.py':workspace/'runs/width-readouts-ep200-20261016/runs/i256_eps896/probe.py'}
    for name,dest in files.items():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'evaluation/frozen'/name,dest)
    hashes=json.loads((cache/'complete.json').read_text())
    manifest=dict(cases=[c for c in cases if c['kind']!='adm'],times=SPEC['times'],noise_seed=SPEC['noise_seed'],
                  split_note=SPEC['split_note'],cache_hashes=hashes,
                  historical_helpers={str(files[n]):digest(files[n]) for n in ['native_core.py','affine_core.py']})
    save(run/'manifest.json',manifest);save(workspace/'runs/denoising-probes-20261023/manifest.json',manifest)
    c=next(c for c in cases if c['id']=='bottleneck_noise')
    noise=dict(checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],source=c['source'],
               source_hashes=c['source_hashes'],cache_hashes={k:hashes[k] for k in ['images256.npy','labels.npy','samples.json']},
               worker_sha256=digest(run/'noise_probe.py'),all_plot_times=[.1,.5,.7,.8,.9,.95])
    save(run/'noise-manifest.json',noise)
    by_id={c['id']:c for c in cases}
    for res in [128,256]:
        folder=run/f'spectra{res}';(folder/'results').mkdir(parents=True,exist_ok=True);(folder/'data').mkdir(exist_ok=True)
        m=copy.deepcopy(SPEC['spectra'][str(res)])
        m['cases']=[by_id[k] for k in m.pop('case_ids')]
        m['cache_hashes']={k:hashes[k] for k in [f'images{res}.npy','labels.npy','samples.json']}
        m['worker_hashes']={}
        for name in ['extract.py','analyze.py']:
            shutil.copy2(ROOT/f'evaluation/frozen/spectra{res}'/name,folder/name);m['worker_hashes'][name]=digest(folder/name)
        for name,source in [('clean.npy',cache/f'spectral-clean{res}.npy'),('noise-eval.npy',cache/f'spectral-noise{res}.npy')]:
            dest=folder/'data'/name
            if dest.is_symlink() or dest.exists():
                if dest.resolve()!=source.resolve():raise ValueError(f'Existing data link points elsewhere: {dest}')
            else:dest.symlink_to(source)
        m['data_hashes']={f'data/{name}':digest(folder/'data'/name) for name in ['clean.npy','noise-eval.npy']}
        save(folder/'manifest.json',m)
    save(workspace/'runtime.json',dict(cache=str(cache),run=str(run),case_ids=list(by_id),
         accept_new_checkpoints=accept_new,training_base=SPEC['training_base']))
    return workspace/'runtime.json'
