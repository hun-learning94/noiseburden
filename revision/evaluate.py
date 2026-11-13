"""Prepare portable validation crops, stage workers, and evaluate one model/process."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess,sys
import numpy as np
from evaluation.runtime import ROOT,digest,save
FROZEN=ROOT/'revision/frozen'

def prepare(cache,output):
    cache=Path(cache).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    ids=np.r_[np.random.default_rng(251025).permutation(8000)[:512],np.arange(8000,10000)]
    np.save(output/'original_indices.npy',ids)
    labels=np.load(cache/'labels.npy');np.save(output/'labels.npy',labels[ids])
    samples=json.loads((ROOT/'configs/validation_samples.json').read_text());save(output/'samples.json',[samples[i] for i in ids])
    for res in [128,256]:
        source=np.load(cache/f'images{res}.npy',mmap_mode='r')
        array=np.lib.format.open_memmap(output/f'images{res}.npy',mode='w+',dtype=np.uint8,shape=(2512,3,res,res))
        for lo in range(0,2512,32):array[lo:lo+32]=source[ids[lo:lo+32]]
        array.flush()
    save(output/'complete.json',{p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='complete.json'})
    return output

def configure(workspace,cache,assets_file,checkpoints_file=None,accept_new=False):
    workspace=Path(workspace).resolve();workspace.mkdir(parents=True,exist_ok=True)
    cache=Path(cache).resolve();asset_file=Path(assets_file).resolve()
    for name,record in json.loads((ROOT/'configs/revision_data.json').read_text()).items():
        assert digest(cache/name)==record['sha256'],f'Validation cache mismatch: {name}'
    supplied=json.loads(asset_file.read_text())
    def path(key):
        p=Path(supplied[key]).expanduser();return p.resolve() if p.is_absolute() else (asset_file.parent/p).resolve()
    expected=json.loads((ROOT/'configs/public_assets.json').read_text());assets={}
    for key,value in supplied.items():
        if key not in expected:continue
        p=path(key);assert p.is_file(),p
        assert digest(p)==expected[key]['sha256'],key
        assets[key]=dict(expected[key],path=str(p))
    references={k:dict(v,path=str(path('fid'+k))) for k,v in json.loads((ROOT/'configs/fid_statistics.json').read_text()).items() if 'fid'+k in supplied}
    for r in references.values():assert digest(r['path'])==r['sha256']
    for group in ['probe256','probe128','public_fid','latent']:
        src=FROZEN/('probe' if group.startswith('probe') else group);dest=workspace/group
        shutil.copytree(src,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (dest/'results').mkdir(exist_ok=True)
        record=dict(assets)
        if group.startswith('probe'):
            if 'adm256' in assets:record['adm']=assets['adm256']
            save(dest/'assets.json',dict(assets=record))
        if group=='public_fid':
            record['vae_mse']={'files':[assets[k] for k in ['vae_mse_config','vae_mse_weights'] if k in assets]}
            save(dest/'assets.json',dict(assets=record));save(dest/'references.json',references)
        if group=='latent':
            for case in ['uvit','ldm']:
                info={k[len(case)+1:]:v for k,v in assets.items() if k.startswith(case+'_')}
                if case=='ldm':info['configs']=[str(dest/'code/ldm/configs/latent-diffusion/cin-ldm-vq-f8.yaml')]
                save(dest/f'assets-{case}.json',dict(assets=info))
            save(dest/'references.json',references)
            save(dest/'vae-mse-provenance.json',{'files':[]}) # Only the omitted MDT adapter used this unrelated VAE.
            save(dest/'data-provenance.json',{p.name:{'sha256':digest(p)} for p in cache.glob('*.npy')})
        save(dest/'source-manifest.json',{'sha256':{str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.txt','.md']}})
    cases=json.loads((ROOT/'configs/revision_jit_cases.json').read_text())
    if checkpoints_file:
        cpfile=Path(checkpoints_file).resolve();mapping=json.loads(cpfile.read_text())
        for c in cases:
            if c['id'] not in mapping:continue
            p=Path(mapping[c['id']]);p=p.resolve() if p.is_absolute() else (cpfile.parent/p).resolve()
            actual=digest(p)
            if not accept_new:assert c['checkpoint_sha256']==actual,c['id']
            c['published_checkpoint_sha256']=c['checkpoint_sha256'];c['checkpoint_sha256']=actual;c['checkpoint']=str(p)
            c['source']=str(ROOT/c['source'])
            dest=workspace/c['id'];shutil.copytree(FROZEN/'probe',dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT/'revision/jit_probe.py',dest/'jit_probe.py');(dest/'results').mkdir(exist_ok=True)
            save(dest/'case.json',c);save(dest/'assets.json',{'assets':{}})
    save(workspace/'runtime.json',dict(cache=str(cache),assets_file=str(asset_file),accept_new=accept_new,
         source_manifest={str(p.relative_to(FROZEN)):digest(p) for p in FROZEN.rglob('*.py')}))
    return workspace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','setup','input','mse','fid','validate'])
    p.add_argument('--workspace',type=Path,default=Path('work/revision'))
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--assets',type=Path);p.add_argument('--checkpoints',type=Path)
    p.add_argument('--accept-new-checkpoints',action='store_true');p.add_argument('--case')
    p.add_argument('--times',type=float,nargs='+');args=p.parse_args()
    if args.mode=='prepare':
        if not args.cache or not args.output:p.error('prepare requires the arXiv-format 10000-image --cache and an --output directory')
        print(prepare(args.cache,args.output));return
    if args.mode=='setup':
        if not args.cache or not args.assets:p.error('setup requires --cache and --assets; use an empty {} asset map for JiT-only evaluation')
        print(configure(args.workspace,args.cache,args.assets,args.checkpoints,args.accept_new_checkpoints));return
    if not args.case:p.error('choose one model with --case')
    if args.times and not args.case.startswith('i'):p.error('--times applies to JiT; public probes use the fixed nine-time protocol')
    info=json.loads((args.workspace/'runtime.json').read_text());case=args.case
    env=dict(os.environ,PUBLIC_PROBE_CACHE=info['cache'],REVISION_WORK=str(args.workspace.resolve()/'storage'),
             PYTHONPATH=str(ROOT/'evaluation/third_party')+os.pathsep+os.environ.get('PYTHONPATH',''))
    if case.startswith('i'):
        folder=args.workspace/case
        if args.mode in ['input','mse']:
            cmd=[sys.executable,str(folder/'jit_probe.py'),'--mode',args.mode]
            if args.times:cmd+=['--times',*[str(t) for t in args.times]]
        else:p.error('JiT generation uses python -m revision.generate with this workspace')
    elif case in ['adm128','adm256','dit256']:
        if args.mode in ['fid','validate']:
            folder=args.workspace/'public_fid';cmd=[sys.executable,str(folder/('validate_cpu.py' if args.mode=='validate' else 'worker.py'))]
            if args.mode=='fid':cmd+=['--case',case]
        else:
            folder=args.workspace/('probe128' if case=='adm128' else 'probe256')
            script='public_worker128.py' if case=='adm128' else 'worker.py'
            cmd=[sys.executable,str(folder/script),'--model','dit' if case=='dit256' else 'adm']
    elif case in ['uvit','ldm']:
        folder=args.workspace/'latent'
        cmd=[sys.executable,str(folder/({'input':'probe.py','fid':'fid.py','validate':'validate_cpu.py'}[args.mode]))]
        cmd+=([case] if args.mode=='validate' else ['--case',case])
    else:p.error('unknown public model or JiT case')
    subprocess.run(cmd,env=env,check=True)
if __name__=='__main__':main()
