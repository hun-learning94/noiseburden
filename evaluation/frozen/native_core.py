"""Matched native denoising losses and frozen affine/nonlinear probes.
Each subprocess handles one checkpoint or one (checkpoint,time) probe group.
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
MAN=json.loads((HERE/'manifest.json').read_text())
CACHE=Path(os.environ.get('EVAL_CACHE',HERE/'cache'))
OUT=HERE/'results';OUT.mkdir(exist_ok=True)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def patchify(x,p):
    b,c,h,w=x.shape
    return x.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,-1,p*p*c)
def load(c):
    assert digest(c['checkpoint'])==c['checkpoint_sha256']
    for n,h in c['source_hashes'].items():assert digest(Path(c['source'])/n)==h,n
    if c['kind']=='width':
        assert digest(c['loader'])==c['loader_sha256']
        spec=importlib.util.spec_from_file_location('historical_probe',c['loader']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        model=mod.load_model(c)
    else:
        sys.path.insert(0,c['source'])
        from denoiser import Denoiser
        if c['kind']=='edm':
            from edm import EDMDenoiser
            Denoiser=EDMDenoiser
        ck=torch.load(c['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        assert ck['epoch']==199
        a=ck['args']
        for k,v in dict(pred_type='x',loss_type='v',shortcut='none').items():
            if not hasattr(a,k):setattr(a,k,v)
        model=Denoiser(a);model.load_state_dict(ck['model_ema1'],strict=True);model.cuda().eval().requires_grad_(False)
    assert float(c['training_args'].get('noise_scale',1))==1
    return model

def batch(c,lo,hi):
    r=c['image_size'];arr=np.load(CACHE/f'images{r}.npy',mmap_mode='r')
    a=lo if r==256 else lo-8000;b=hi if r==256 else hi-8000
    x=torch.from_numpy(np.array(arr[a:b])).cuda().float()/127.5-1
    labels=torch.from_numpy(np.array(np.load(CACHE/'labels.npy',mmap_mode='r')[lo:hi])).cuda()
    # One Gaussian field per image reused across times and checkpoints. Image-index
    # seeds make corruption independent of worker order and minibatch size.
    e=torch.stack([torch.randn((3,r,r),generator=torch.Generator().manual_seed(MAN['noise_seed']+i)) for i in range(lo,hi)]).cuda()
    return x,e,labels

@torch.no_grad()
def predict(model,c,x,e,labels,tv):
    z=tv*x+(1-tv)*e; t=torch.full((len(x),1,1,1),tv,device='cuda')
    if c['kind']=='edm':
        with torch.autocast('cuda',dtype=torch.bfloat16):clean=model.image_estimate(z,t,labels)
        skip=model.sigma_data**2/(model.sigma_data**2+((1-tv)/tv)**2)
    else:
        with torch.autocast('cuda',dtype=torch.bfloat16):raw=model.net(z,t.flatten(),labels)
        raw=raw.float()
        out=model.apply_shortcut(raw,z,t) if c['kind']=='bottleneck' else raw
        convert=getattr(model,'_out_to_v',getattr(model,'out_to_v',None));assert convert is not None
        clean=z+(1-tv)*convert(out,z,t)
        pred=c.get('pred',c['training_args'].get('pred_type'))
        direct=out if pred=='x' else (z-(1-tv)*out)/tv if pred=='eps' else z+(1-tv)*out
        assert torch.allclose(clean,direct,atol=3e-4,rtol=3e-4),'native conversion parity'
        a=0.
        if c['kind']=='bottleneck':
            s=model.shortcut
            if s=='add':a=1.
            elif s=='precond':a=1/(1-tv)
            elif s.startswith('learned'):a=float(model.time_coeff(t[:1])[0].item())
        skip=tv*a if pred=='x' else 1-(1-tv)*a if pred=='eps' else tv*(1+(1-tv)*a)
    clean=clean.float();eps=(z-tv*clean)/(1-tv)
    return clean,eps,float(skip)

def stats(a):
    a=np.asarray(a,dtype=np.float64);assert np.isfinite(a).all()
    rng=np.random.default_rng(231023)
    means=np.array([a[rng.integers(len(a),size=len(a))].mean() for _ in range(1000)])
    return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n_images=len(a))
def provenance(c):
    return dict(case=c,protocol_sha256=digest(HERE/'manifest.json'),worker_sha256=digest(__file__),cache_sha256=json.loads((CACHE/'complete.json').read_text()),torch=torch.__version__,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),precision='BF16 backbone; FP32 conversion and MLP; FP64 affine sufficient statistics',test_note=MAN['split_note'])

@torch.no_grad()
def native(c,smoke=False):
    dest=OUT/f"native-{c['id']}.json"
    if dest.exists() and not smoke:return
    model=load(c);records=[]; arrays={}
    # Architecture rank upper bound suffices for a valid (possibly conservative) floor.
    rp=min(c['width'],3*c['patch']**2) if c['kind']=='width' else 128
    for tv in ([.1,.5,.9] if smoke else MAN['times']):
        xx=[];ee=[]
        for lo in range(8000,8008 if smoke else 10000,8 if smoke else 16):
            hi=min(lo+(8 if smoke else 16),8008 if smoke else 10000)
            x,e,y=batch(c,lo,hi);clean,eps,skip=predict(model,c,x,e,y,tv)
            a=(clean-x).square().flatten(1).mean(1);b=(eps-e).square().flatten(1).mean(1)
            assert torch.allclose(a,((1-tv)/tv)**2*b,atol=5e-5,rtol=1e-4)
            xx.extend(a.cpu().tolist());ee.extend(b.cpu().tolist())
        floor=skip**2*((1-tv)/tv)**2*(1-rp/(3*c['patch']**2))
        records.append(dict(t=tv,sigma=(1-tv)/tv,clean_mse=stats(xx),noise_mse=stats(ee),clean_skip=skip,rank_upper_bound=rp,clean_mse_floor=floor))
        arrays[f'x_{tv}']=xx;arrays[f'eps_{tv}']=ee
        print(c['id'],records[-1],flush=True)
    if not smoke:
        np.savez_compressed(OUT/f"native-{c['id']}-per-image.npz",**arrays)
        save(dest,dict(provenance=provenance(c),records=records))
    else:save(OUT/f"smoke-{c['id']}.json",dict(provenance=provenance(c),records=records))

def affine_fit(f,y,nfit,ndev):
    d=f.shape[1];p=y.shape[1]
    gram=torch.zeros((d+1,d+1),device='cuda',dtype=torch.float64);cross=torch.zeros((d+1,p),device='cuda',dtype=torch.float64)
    for lo in range(0,nfit,4096):
        ff=torch.cat([f[lo:min(lo+4096,nfit)].double(),torch.ones((min(4096,nfit-lo),1),device='cuda',dtype=torch.float64)],1)
        gram+=ff.T@ff;cross+=ff.T@y[lo:min(lo+4096,nfit)].double()
    gram/=nfit;cross/=nfit
    best=None
    for ridge in [1e-8,1e-6,1e-4,1e-2,1.]:
        reg=torch.eye(d+1,device='cuda',dtype=torch.float64)*ridge*gram.diag()[:-1].mean();reg[-1,-1]=1e-12
        w=torch.linalg.solve(gram+reg,cross).float()
        err=0.
        for lo in range(nfit,nfit+ndev,8192):
            hi=min(lo+8192,nfit+ndev);err+=(f[lo:hi]@w[:-1]+w[-1]-y[lo:hi]).square().sum().item()
        if best is None or err<best[0]:best=(err,w,ridge)
    return best[1],best[2]

@torch.no_grad()
def evaluate(f,y,w,net,start,end,tokens,return_errors=False):
    errors=[]
    for lo in range(start,end,8192):
        hi=min(lo+8192,end);pred=f[lo:hi]@w[:-1]+w[-1]
        if net is not None:pred+=net(f[lo:hi])
        errors.append((pred-y[lo:hi]).square().mean(1))
    a=torch.cat(errors).reshape(-1,tokens).mean(1).cpu().numpy()
    if return_errors:return a
    return float(a.mean())

def one_probe(c,tv,tap,target,f,y,tokens,smoke=False):
    stem=f"probe-{c['id']}-t{tv}-{tap}-{target}"
    dest=OUT/(stem+'.json')
    if dest.exists() and not smoke:return
    nf,nd,nt=(16,8,8) if smoke else (6400,1600,2000)
    nfit,ndev,end=nf*tokens,nd*tokens,(nf+nd+nt)*tokens
    # Standardize using fit images only. A linear map of standardized features
    # describes the same affine class; the target remains in original units.
    mean=f[:nfit].mean(0);sd=f[:nfit].std(0).clamp_min(1e-5)
    f=(f-mean)/sd
    with torch.no_grad():w,ridge=affine_fit(f,y,nfit,ndev)
    aff=evaluate(f,y,w,None,nfit+ndev,end,tokens,True)
    ytest=y[nfit+ndev:end];var=float((ytest-ytest.mean(0)).square().mean().item())
    affine=dict(mse=stats(aff),ev=1-float(aff.mean())/var,noise_recovery_score=1-float(aff.mean()) if target=='eps' else None,ridge=ridge)
    save(OUT/(stem+'-affine.json'),affine)
    fitted=[]
    for seed in ([0] if smoke else MAN['probe']['seeds']):
        for lr in ([.001] if smoke else MAN['probe']['learning_rates']):
            fitpath=OUT/f'{stem}-seed{seed}-lr{lr}.json'
            if fitpath.exists() and not smoke:
                fitted.append(json.loads(fitpath.read_text()));continue
            torch.manual_seed(seed)
            net=nn.Sequential(nn.Linear(f.shape[1],512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,y.shape[1])).cuda()
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
            best=evaluate(f,y,w,net,nfit,nfit+ndev,tokens);bestepoch=0;beststate={k:v.detach().clone() for k,v in net.state_dict().items()};history=[dict(epoch=0,dev_mse=best)]
            for epoch in range(1,(1 if smoke else MAN['probe']['epochs'])+1):
                idx=torch.randperm(nfit,device='cuda');net.train()
                for lo in range(0,nfit,8192):
                    ii=idx[lo:lo+8192];ff=f[ii];yy=y[ii]
                    with torch.no_grad():base=ff@w[:-1]+w[-1]
                    err=(base+net(ff)-yy).square().mean();assert torch.isfinite(err)
                    opt.zero_grad(set_to_none=True);err.backward();opt.step()
                net.eval();dev=evaluate(f,y,w,net,nfit,nfit+ndev,tokens);history.append(dict(epoch=epoch,dev_mse=dev))
                print(stem,seed,lr,epoch,dev,flush=True)
                if dev<best:
                    best=dev;bestepoch=epoch;beststate={k:v.detach().clone() for k,v in net.state_dict().items()}
                if epoch-bestepoch>=MAN['probe']['patience']:break
            net.load_state_dict(beststate)
            err=evaluate(f,y,w,net,nfit+ndev,end,tokens,True)
            result=dict(seed=seed,lr=lr,best_epoch=bestepoch,dev_mse=best,mse=stats(err),ev=1-float(err.mean())/var,noise_recovery_score=1-float(err.mean()) if target=='eps' else None,history=history)
            np.savez_compressed(fitpath.with_suffix('.npz'),mlp_mse=err,affine_mse=aff)
            torch.save(dict(state={k:v.cpu() for k,v in beststate.items()},affine=w.cpu(),input_mean=mean.cpu(),input_std=sd.cpu()),fitpath.with_suffix('.pth'))
            save(fitpath,result);fitted.append(result)
            del net,opt,beststate;gc.collect()
    selected=[min([v for v in fitted if v['seed']==s],key=lambda v:v['dev_mse']) for s in sorted({v['seed'] for v in fitted})]
    save(dest,dict(case=c['id'],t=tv,tap=tap,target=target,affine=affine,selected_by_dev=selected,all_fits=fitted,noise_ceiling=1/6 if target=='eps' else None,protocol_sha256=digest(HERE/'manifest.json'),smoke=smoke))

@torch.no_grad()
def extract(c,tv,smoke):
    model=load(c);num=32 if smoke else 10000;tokens=256;p=16;captures={}
    hooks=[model.net.blocks[0].register_forward_pre_hook(lambda m,a:captures.update(embedding=a[0].detach())),model.net.blocks[-1].register_forward_hook(lambda m,a,o:captures.update(block12=o[:,-tokens:].detach()))]
    feats={};ys={k:torch.empty((num*tokens,768),device='cuda') for k in ['x','eps']}
    for lo in range(0,num,16):
        hi=min(lo+16,num);x,e,labels=batch(c,lo,hi)
        predict(model,c,x,e,labels,tv)
        for name,v in captures.items():
            assert v.shape[1:]==(tokens,768),(name,v.shape)
            if name not in feats:feats[name]=torch.empty((num*tokens,768),device='cuda')
            feats[name][lo*tokens:hi*tokens]=v.reshape(-1,768).float()
        ys['x'][lo*tokens:hi*tokens]=patchify(x,p).reshape(-1,768)
        ys['eps'][lo*tokens:hi*tokens]=patchify(e,p).reshape(-1,768)
        if lo%1024==0:print('extracted',c['id'],tv,hi,flush=True)
    for h in hooks:h.remove()
    del model,captures;gc.collect();torch.cuda.empty_cache()
    return feats,ys,tokens

def probes(c,tv,smoke=False):
    group=OUT/f"probes-{c['id']}-t{tv}.json"
    if group.exists() and not smoke:return
    feats,ys,tokens=extract(c,tv,smoke)
    for tap in ['embedding','block12']:
        for target in ['x','eps']:one_probe(c,tv,tap,target,feats[tap],ys[target],tokens,smoke)
    save(group,dict(provenance=provenance(c),t=tv,complete=True,smoke=smoke))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['native','probe']);ap.add_argument('case');ap.add_argument('--time',type=float);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.smoke:
        OUT=OUT/'smoke';OUT.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.48)
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=next(c for c in MAN['cases'] if c['id']==a.case)
    if a.mode=='native':native(c,a.smoke)
    else:probes(c,a.time,a.smoke)
