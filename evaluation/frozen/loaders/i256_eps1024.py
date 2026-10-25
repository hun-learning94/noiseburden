"""Frozen, image-disjoint affine readouts, adapted from epsilon_z_recoverability.py.

Only block12 and final-head input; three interior times; no backbone training.
"""
import argparse
import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['TORCHDYNAMO_DISABLE']='1'
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
RIDGES=(0.,1e-8,1e-6,1e-4,1e-2,1.)
TARGETS=('z','epsilon','x')
TIMES=(.1,.5,.9)
SEEDS=(99001,99003,99005)
FIT,DEV,TEST=384,128,128

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*2**20),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def patches(x,p):
    b,c,h,w=x.shape
    return x.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,(h//p)*(w//p),p*p*c)

def unpatch(x,p):
    b,n,d=x.shape;side=int(n**.5)
    assert side*side==n and d==3*p*p
    return x.reshape(b,side,side,p,p,3).permute(0,5,1,3,2,4).reshape(b,3,side*p,side*p)

def fit_probe(features,targets,names=TARGETS,n_fit=FIT,n_dev=DEV,shared_intercept=False):
    """Shared token-wise weights, fit-only position means; FP64 ridge solve.

    Ridge is selected separately for each target on development images. The
    spectral pseudoinverse discards covariance eigenvalues <= max*1e-12.
    """
    dim=targets.shape[-1]//len(names)
    assert targets.shape[-1]==dim*len(names)
    hf,yf=features[:n_fit].double(),targets[:n_fit].double()
    mh,my=(hf.mean((0,1))[None],yf.mean((0,1))[None]) if shared_intercept else (hf.mean(0),yf.mean(0))
    hf,yf=(hf-mh).flatten(0,1),(yf-my).flatten(0,1)
    gram,cross=hf.T@hf/len(hf),hf.T@yf/len(hf)
    del hf,yf
    hd=(features[n_fit:n_fit+n_dev].double()-mh).flatten(0,1)
    yd=(targets[n_fit:n_fit+n_dev].double()-my).flatten(0,1)
    gd,cd,y2=hd.T@hd/len(hd),hd.T@yd/len(hd),yd.square().mean(0)
    del hd,yd
    eig,vectors=torch.linalg.eigh(gram)
    eig=eig.clamp_min(0);scale=eig.mean().clamp_min(1e-20)
    cutoff=eig[-1].clamp_min(1e-20)*1e-12
    rotated=vectors.T@cross
    candidates,development=[],[]
    for ridge in RIDGES:
        inv=(eig+ridge*scale).clamp_min(1e-30).reciprocal()
        if ridge==0:inv=torch.where(eig>cutoff,inv,0)
        weight=vectors@(inv[:,None]*rotated)
        mse=(y2-2*(weight*cd).sum(0)+(weight*(gd@weight)).sum(0)).clamp_min(0)
        development.append(mse.reshape(len(names),dim).mean(1).tolist())
        candidates.append(weight)
    choice=torch.tensor(development).argmin(0).tolist()
    weight=torch.cat([candidates[choice[i]][:,i*dim:(i+1)*dim] for i in range(len(names))],-1)
    decoded=(features[n_fit+n_dev:].double()-mh)@weight+my
    assert torch.isfinite(decoded).all()
    return decoded,dict(target_order=list(names),ridge_candidates=list(RIDGES),
        development_mse=development,selected_ridge={n:RIDGES[choice[i]] for i,n in enumerate(names)},
        feature_dimension=features.shape[-1],target_dimension_each=dim,
        shared_matrix_parameters_each=features.shape[-1]*dim,
        position_intercept_parameters_each=(1 if shared_intercept else features.shape[1])*dim,
        shared_intercept=shared_intercept,
        covariance_rank_at_relative_1e_12=int((eig>cutoff).sum()),
        covariance_eigenvalue_min=eig[0].item(),covariance_eigenvalue_max=eig[-1].item(),
        weight_frobenius_norm={n:weight[:,i*dim:(i+1)*dim].norm().item() for i,n in enumerate(names)})

def bootstrap_weights(n,device):
    rng=np.random.default_rng(240924)
    samples=rng.integers(0,n,size=(1000,n))
    counts=np.zeros((1000,n),dtype=np.float64)
    np.add.at(counts,(np.arange(1000)[:,None],samples),1/n)
    return torch.from_numpy(counts).to(device)

def metrics(prediction,truth,boot):
    truth=truth.double();error=prediction.double()-truth
    per_image=error.square().mean((1,2))
    # Center each within-patch coordinate over test images and positions.
    means=truth.mean(1);seconds=truth.square().mean(1)
    variance=(seconds.mean(0)-means.mean(0).square()).mean()
    mse=per_image.mean();ev=1-mse/variance
    bmse=boot@per_image
    bvar=((boot@seconds)-(boot@means).square()).mean(1)
    quant=torch.tensor([.025,.975],device=truth.device,dtype=torch.float64)
    return dict(mse=mse.item(),variance=variance.item(),ev=ev.item(),r2_percent=100*ev.item(),
        mse_by_image=per_image.cpu().tolist(),
        mse_ci95=torch.quantile(bmse,quant).cpu().tolist(),
        ev_ci95=torch.quantile(1-bmse/bvar,quant).cpu().tolist())

def readout(features,targets,boot,names=TARGETS):
    predicted,info=fit_probe(features,targets,names)
    d=targets.shape[-1]//len(names)
    info['test']={name:metrics(predicted[...,i*d:(i+1)*d],
        targets[FIT+DEV:,...,i*d:(i+1)*d],boot) for i,name in enumerate(names)}
    return info

class Captures:
    def __init__(self,net,include_embed=False):
        self.values={}
        def keep(name,x):self.values[name]=x.detach().float().clone()
        self.handles=[net.blocks[-1].register_forward_hook(
            lambda m,i,o:keep('block12',o[:,net.in_context_len:])),
            net.final_layer.linear.register_forward_pre_hook(lambda m,i:keep('head_input',i[0]))]
        if include_embed:
            self.handles.append(net.x_embedder.register_forward_hook(lambda m,i,o:keep('embed',o)))
    def close(self):
        for h in self.handles:h.remove()

@torch.no_grad()
def extract(model,z,labels,tv,p,w,precision,include_embed):
    captures=Captures(model.net,include_embed)
    features={name:[] for name in (['block12','head_input','embed'] if include_embed else ['block12','head_input'])}
    outputs=[];parity=0.
    try:
        for start in range(0,len(z),32):
            zz=z[start:start+32];t=torch.full((len(zz),),tv,device=z.device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                output=model.net(zz,t,labels[start:start+32])
                replay=model.net.final_layer.linear(captures.values['head_input'])
            parity=max(parity,(unpatch(replay.float(),p)-output.float()).abs().max().item())
            for name in features:
                value=captures.values[name]
                assert value.shape==(len(zz),256,w),(name,value.shape,w)
                features[name].append(value)
            outputs.append(output.float())
    finally:captures.close()
    assert parity<1e-5,parity
    return {k:torch.cat(v) for k,v in features.items()},torch.cat(outputs),parity

def load_model(case):
    for name,h in case['source_hashes'].items():assert digest(Path(case['source'])/name)==h,name
    assert digest(case['checkpoint'])==case['checkpoint_sha256']
    sys.path.insert(0,case['source'])
    import model_jit
    from denoiser import Denoiser
    ck=torch.load(case['checkpoint'],mmap=True,map_location='cpu',weights_only=False)
    a=copy.deepcopy(ck['args']);assert ck['epoch']==case['epoch']
    model_jit.JiT_models[a.model]=partial(model_jit.JiT,depth=12,hidden_size=case['width'],
        num_heads=case['heads'],patch_size=case['patch'],bottleneck_dim=128,in_context_len=32,in_context_start=4)
    model=Denoiser(a)
    model.load_state_dict(ck['model_'+case['ema'].lower()],strict=True)
    model.cuda().eval().requires_grad_(False)
    assert len(model.net.blocks)==12 and model.net.in_context_len==32 and model.net.in_context_start==4
    if case.get('attention_backend')=='sdpa':
        from runtime_optim import configure
        configure('sdpa',False)
    return model

@torch.no_grad()
def run(case,manifest,smoke=False):
    tick=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    assert shutil.disk_usage(HERE).free>20*2**30
    model=load_model(case)
    r,p,w=case['image_size'],case['patch'],case['width']
    fixture=manifest['fixtures'][str(r)]
    assert digest(fixture['path'])==fixture['sha256']
    with np.load(fixture['path']) as f:
        images=torch.from_numpy(f['images'].copy()).cuda().float()/127.5-1
        labels=torch.from_numpy(f['labels'].copy()).cuda().long()
    assert images.shape==(640,3,r,r)
    boot=bootstrap_weights(TEST,images.device)
    report=dict(case=case,fixture=fixture,probe_sha256=digest(__file__),split=manifest['split'],
        torch_version=torch.__version__,device=torch.cuda.get_device_name(),parameters=sum(v.numel() for v in model.parameters()),
        frozen=not any(v.requires_grad for v in model.parameters()),regression_precision='float64',
        compiled=False,tf32=False,backbone_precision='bf16',
        attention_backend='sdpa' if case.get('attention_backend')=='sdpa' else 'native saved manual attention',
        times=list(TIMES),bootstrap=dict(seed=240924,samples=1000,unit='held-out probe image'),records=[],precision_check=[])
    # Historical FP32 controls are retained in evidence; do not duplicate them.
    square_epsilon=False
    for tv,seed in zip(TIMES,SEEDS):
        if tv not in case.get('times',TIMES):continue
        if smoke and tv!=.5:continue
        start=time.monotonic()
        corruption=manifest['corruptions'][str(tv)]
        assert r==256 and digest(corruption['path'])==corruption['sha256']
        noise=torch.from_numpy(np.load(corruption['path'])).cuda()
        assert noise.shape==(640,3,256,256) and noise.dtype==torch.float32
        assert hashlib.sha256(noise[0].cpu().numpy().tobytes()).hexdigest()==corruption['first_image_sha256']
        z=tv*images+(1-tv)*noise
        features,out,parity=extract(model,z,labels,tv,p,w,'bf16',square_epsilon and tv==.5)
        if smoke:
            # A full, real checkpoint hook/linear-head replay, plus a small regression.
            target=patches(images,p)
            y,info=fit_probe(features['head_input'][:24],target[:24],('x',),16,4)
            save(HERE/'results'/f"{case['case']}-smoke.json",dict(case=case['case'],parity=parity,
                 regression_finite=bool(torch.isfinite(y).all()),seconds=time.monotonic()-tick))
            return
        target=torch.cat([patches(q,p) for q in (z,noise,images)],-1)
        zpatch,xpatch=patches(z,p),patches(images,p)
        t=torch.full((640,1,1,1),tv,device='cuda')
        convert=getattr(model,'_out_to_v',getattr(model,'out_to_v',None));assert convert is not None
        clean=z+(1-tv)*convert(out,z,t)
        direct=out if case['pred']=='x' else (z-(1-tv)*out)/tv
        conversion_parity=(direct-clean).abs().max().item();assert conversion_parity<2e-5,conversion_parity
        epshat=out if case['pred']=='eps' else (z-tv*out)/(1-tv)
        row=dict(t=tv,precision='bf16',noise_seed=seed,
            noise_first_image_sha256=hashlib.sha256(noise[0].cpu().numpy().tobytes()).hexdigest(),
            head_reconstruction_max_abs=parity,native_conversion_max_abs=conversion_parity,
            native_x=metrics(patches(clean,p)[512:],xpatch[512:],boot),
            native_epsilon=metrics(patches(epshat,p)[512:],patches(noise,p)[512:],boot),readouts={})
        factor=((1-tv)/tv)**2
        identity_error=abs(row['native_x']['mse']-factor*row['native_epsilon']['mse'])
        assert identity_error < 2e-6*max(1.,row['native_x']['mse']),identity_error
        row['native_mse_conversion_identity_abs_error']=identity_error
        if case['pred']=='eps':
            # Native linear head has one bias vector shared by patch positions.
            assert model.net.final_layer.linear.bias.shape==(p*p*3,)
            refit,info=fit_probe(features['head_input'],patches(noise,p),('epsilon',),shared_intercept=True)
            refit_clean=(zpatch[512:].double()-(1-tv)*refit)/tv
            row['refitted_native_noise_head']=dict(fit=info,
                epsilon=metrics(refit,patches(noise,p)[512:],boot),
                clean=metrics(refit_clean,xpatch[512:],boot))
        for name in ('block12','head_input'):
            row['readouts'][name]=readout(features[name],target,boot)
            row['readouts']['z_plus_'+name]=readout(torch.cat([zpatch,features[name]],-1),xpatch,boot,('x',))
        row['readouts']['raw_z']=readout(zpatch,xpatch,boot,('x',))
        if square_epsilon and tv==.5:
            row['readouts']['embed_z_control']=readout(features['embed'],zpatch,boot,('z',))
        row['seconds']=time.monotonic()-start
        report['records'].append(row)
        print('CELL_COMPLETE',case['case'],case['ema'],tv,round(row['seconds'],2),flush=True)
        if square_epsilon and tv==.5:
            fp,_,parity32=extract(model,z,labels,tv,p,w,'fp32',True)
            report['precision_check'].append(dict(t=tv,precision='fp32',head_reconstruction_max_abs=parity32,
                readouts={name:readout(fp[name],zpatch,boot,('z',)) for name in ('embed','block12','head_input')}))
            del fp
        del features,target,zpatch,xpatch,noise,z,clean,out,epshat,direct
        report['seconds']=time.monotonic()-tick
        report['max_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        save(HERE/'results'/f"{case['case']}.json",report)
    print('CASE_COMPLETE',case['case'],round(report['seconds'],2),flush=True)

def self_check():
    torch.set_num_threads(2);torch.manual_seed(3821)
    z=torch.randn(96,4,8,dtype=torch.float64);x=torch.randn_like(z);eps=(z-.3*x)/.7
    target=torch.cat([z,eps,x],-1)
    h=torch.cat([z,x],-1)@torch.randn(16,24,dtype=torch.float64)+torch.randn(4,24)
    y,meta=fit_probe(h,target,n_fit=56,n_dev=20)
    assert (y-target[76:]).square().mean()<1e-15
    changed=target.clone();changed[76:]+=100
    yy,mm=fit_probe(h,changed,n_fit=56,n_dev=20)
    assert torch.equal(y,yy) and mm['selected_ridge']==meta['selected_ridge']
    for p in (8,16):
        im=torch.randn(2,3,2*p,2*p);assert torch.equal(im,unpatch(patches(im,p),p))
    q=torch.randn(128,7,4);boot=bootstrap_weights(128,'cpu')
    m=metrics(q,q,boot);assert m['mse']==0 and m['ev']==1 and m['ev_ci95']==[1.,1.]
    print('SELF_CHECK_OK: exact affine recovery; no test-target leakage; both patch orders; image bootstrap.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--smoke',action='store_true');opts=parser.parse_args()
    if opts.self_check:self_check()
    else:
        manifest=json.loads((HERE/'manifest.json').read_text())
        case=next(c for c in manifest['cases'] if c['case']==opts.case)
        run(case,manifest,opts.smoke)
