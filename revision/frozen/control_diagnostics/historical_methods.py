"""Unchanged numerical methods extracted from the frozen manuscript evaluations."""
import numpy as np
import torch
RIDGES=(0.,1e-8,1e-6,1e-4,1e-2,1.)
TARGETS=('z','epsilon','x')
FIT,DEV,TEST=384,128,128

def patchify(x,p):
    b,c,h,w=x.shape
    return x.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,-1,p*p*c)


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


