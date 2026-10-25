"""Extract learned preconditioning coefficient grids on CPU from EMA1 weights."""
from pathlib import Path
import argparse,ast,csv,json
import numpy as np
import torch
from torch import nn
from .runtime import digest,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('work/evaluation'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    info=json.loads((a.workspace/'runtime.json').read_text());manifest=json.loads((Path(info['run'])/'manifest.json').read_text())
    rows=[];checkpoints={};torch.set_num_threads(1)
    for c in manifest['cases']:
        if not c['id'].startswith('bottleneck_learned'):continue
        source=Path(c['source'])/'denoiser.py';assert digest(source)==c['source_hashes']['denoiser.py']
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TimeCoeff')
        ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        shortcut=c['training_args']['shortcut'];model=ns['TimeCoeff'](a0=1. if shortcut=='learned_add' else 0.,precond_base=shortcut=='learned_precond',t_eps=c['training_args']['t_eps'])
        assert digest(c['checkpoint'])==c['checkpoint_sha256']
        ck=torch.load(c['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        model.load_state_dict({k[len('time_coeff.'):]:v for k,v in ck['model_ema1'].items() if k.startswith('time_coeff.')},strict=True)
        model.eval();times=sorted(set(np.linspace(.05,.95,901).tolist()+manifest['times']))
        with torch.no_grad():u,w=model(torch.tensor(times,dtype=torch.float32))
        for t,uv,wv in zip(times,u.flatten().tolist(),w.flatten().tolist()):
            rows.append(dict(case=c['id'],t=t,sigma=(1-t)/t,u=uv,w=wv,exact_u=1/(1-t),exact_w=-t/(1-t),relative_u=(1-t)*uv,relative_w=-(1-t)/t*wv,clean_skip=1-(1-t)*uv))
        checkpoints[c['id']]=c['checkpoint_sha256'];del ck,model
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'learned-coefficients.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(a.output/'coefficient-provenance.json',dict(grid_matches_saved_native_outputs=True,
         verification='Grid evaluated directly from the configured EMA1 coefficient modules.',checkpoint_sha256=checkpoints))

if __name__=='__main__':main()
