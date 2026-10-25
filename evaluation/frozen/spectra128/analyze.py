"""FP64 residual spectra, fixed-patch controls, and image-resampling stability."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def covariance(x):
    c=x-x.mean(0);return (c.T@c)/len(c)
def measures(eigen):
    e=eigen.clamp_min(0).flip(0);total=e.sum();cum=e.cumsum(0)
    if total<1e-20:return dict(energy_per_coordinate=0.,participation_rank=0.,k50=0,k90=0,k95=0,k99=0,tail64_fraction=0.,tail64_per_coordinate=0.),e
    result=dict(energy_per_coordinate=float(total/len(e)),participation_rank=float(total.square()/e.square().sum()),tail64_fraction=float(e[64:].sum()/total),tail64_per_coordinate=float(e[64:].sum()/len(e)))
    for p in [.5,.9,.95,.99]:result[f'k{int(p*100)}']=int(torch.searchsorted(cum,p*total))+1
    tails=total-cum
    for delta in [1e-5,1e-4,1e-3]:result[f'k_absolute_{delta:g}']=0 if total/len(e)<=delta else int(torch.searchsorted(-tails/len(e),torch.tensor(-delta,dtype=e.dtype,device=e.device)))+1
    return result,e
def summarize_residual(r):
    cov=covariance(r);e=torch.linalg.eigvalsh((cov+cov.T)*.5)
    assert float(e[0])>=-1e-9*max(float(e[-1]),1.)
    metrics,e=measures(e);metrics.update(mean_bias_per_coordinate=float(r.mean(0).square().mean()),second_moment_per_coordinate=float(r.square().mean()))
    assert np.isclose(metrics['energy_per_coordinate']+metrics['mean_bias_per_coordinate'],metrics['second_moment_per_coordinate'],rtol=1e-8,atol=1e-12)
    return metrics,e
def crossvalidated_tail(r):
    tails=[]
    for a,b in [(r[:2000],r[2000:]),(r[2000:],r[:2000])]:
        _,q=torch.linalg.eigh(covariance(a));q=q.flip(1)
        v=((b-b.mean(0))@q).square().mean(0);tails.append(torch.cat([v.sum()[None],v.sum()-v.cumsum(0)]))
    tail=torch.stack(tails).mean(0).clamp_min(0);fraction=tail/tail[0]
    result={f'cv_k{int(100*p)}':int(torch.nonzero(fraction<=1-p)[0]) for p in [.5,.9,.95,.99]}
    return result,fraction
def main():
    m=json.loads((HERE/'manifest.json').read_text());assert digest(__file__)==m['worker_hashes']['analyze.py']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();tick=time.time();device='cuda'
    for path,h in m['data_hashes'].items():assert digest(HERE/path)==h
    clean=np.load(HERE/'data/clean.npy',mmap_mode='r');noise=np.load(HERE/'data/noise-eval.npy',mmap_mode='r')
    x=torch.tensor(np.array(clean[6000:]),dtype=torch.float64,device=device);eps=torch.tensor(np.array(noise),dtype=torch.float64,device=device)
    baselines={}
    for n in m['baseline_fit_counts']:
        baselines[n]=[]
        for p in range(5):
            fit=torch.tensor(np.array(clean[:n,p]),dtype=torch.float64,device=device);mu=fit.mean(0);e,q=torch.linalg.eigh(covariance(fit));assert e[0]>-1e-9
            baselines[n].append((mu,e.clamp_min(0),q))
    def affine(tv,p,n=6000):
        mu,e,q=baselines[n][p];s=(1-tv)/tv;y=x[:,p]+s*eps[:,p]
        return mu+(((y-mu)@q)*(e/(e+s*s)))@q.T
    rng=np.random.default_rng(m['bootstrap_seed']);counts=torch.tensor(rng.multinomial(4000,np.full(4000,1/4000),size=m['bootstrap_samples']),dtype=torch.float64,device=device)
    subset=np.random.default_rng(241025).permutation(4000);records=[];controls=[];negative=[];spectra={};boot_output={};central={}
    # Clean-target residuals are a negative control: they also contain Bayes uncertainty.
    for tv in m['times']:
        for p in range(5):
            metric,e=summarize_residual(x[:,p]-affine(tv,p));negative.append(dict(t=tv,patch=p,**metric));spectra[f'clean-target_t{tv}_p{p}']=e.cpu().numpy()
    for case in m['cases']:
        key=case['id'];complete=json.loads((OUT/f'{key}-complete.json').read_text());assert complete['complete'] and complete['checkpoint_sha256']==case['checkpoint_sha256']
        assert complete['manifest_sha256']==digest(HERE/'manifest.json')
        preds=np.load(OUT/f'{key}-predictions.npy',mmap_mode='r');mse=np.load(OUT/f'{key}-mse.npy',mmap_mode='r')
        assert preds.shape==(7,4000,5,192) and np.isfinite(preds).all() and np.isfinite(mse).all()
        for ti,tv in enumerate(m['times']):
            for p in range(5):
                d=torch.tensor(np.array(preds[ti,:,p]),dtype=torch.float64,device=device);ell=affine(tv,p);r=d-ell;truth_error=x[:,p]-d
                metric,e=summarize_residual(r);cross=(r*truth_error).mean(1)
                metric.update(teacher_patch_mse=float(truth_error.square().mean()),teacher_full_image_mse=float(mse[ti].mean()),native_fixture_fp32_mse=float(mse[ti,2000:].mean()),affine_clean_mse=float((x[:,p]-ell).square().mean()),bayes_orthogonality_cross=float(cross.mean()),cross_over_residual_second_moment=float(cross.mean()/r.square().mean()))
                assert np.isclose(metric['affine_clean_mse'],metric['teacher_patch_mse']+metric['second_moment_per_coordinate']+2*metric['bayes_orthogonality_cross'],rtol=1e-8)
                records.append(dict(teacher=key,t=tv,sigma=(1-tv)/tv,patch=p,**metric));spectra[f'{key}_t{tv}_p{p}']=e.cpu().numpy()
                if p!=0:continue
                central[key,tv]=r.detach().cpu().numpy().astype(np.float32)
                cv,tail=crossvalidated_tail(r);controls.append(dict(teacher=key,t=tv,control='two_fold_heldout_principal_directions',**cv));spectra[f'{key}_t{tv}_cv_fraction']=tail.cpu().numpy()
                # Fit-clean covariance sensitivity and image-count sensitivity.
                metric_half,_=summarize_residual(d-affine(tv,p,3000));controls.append(dict(teacher=key,t=tv,control='baseline_fit_3000',**metric_half))
                for n in [1000,2000]:
                    metric_sub,_=summarize_residual(r[torch.tensor(subset[:n],device=device)]);controls.append(dict(teacher=key,t=tv,control=f'evaluation_images_{n}',**metric_sub))
                # A held-out, teacher-specific affine projection diagnoses linear mismatch.
                yy=x[:,p]+((1-tv)/tv)*eps[:,p];yf=yy[:2000];df=d[:2000];my=yf.mean(0);md=df.mean(0);yc=yf-my;dc=df-md;gram=yc.T@yc/2000
                ridge=1e-4*gram.diag().mean();w=torch.linalg.solve(gram+ridge*torch.eye(192,dtype=torch.float64,device=device),yc.T@dc/2000)
                projected=d[2000:]-((yy[2000:]-my)@w+md)
                metric_projected,_=summarize_residual(projected);controls.append(dict(teacher=key,t=tv,control='teacher_specific_affine_2000fit_2000test_ridge1e-4',**metric_projected))
                metric_primary_half,_=summarize_residual(r[2000:]);controls.append(dict(teacher=key,t=tv,control='primary_baseline_same_2000test',**metric_primary_half))
                # Conditional image-resampling stability; not a calibrated posterior-spectrum CI.
                be=[];bn=[];bp=[];eigen_boot=[]
                for bi,weights in enumerate(counts):
                    mu=(weights@r)/4000;cov=r.T@(r*weights[:,None])/4000-torch.outer(mu,mu);ev=torch.linalg.eigvalsh((cov+cov.T)*.5);bm,bev=measures(ev)
                    be.append(bm['energy_per_coordinate']);bn.append(bm['k95']);bp.append(bm['participation_rank']);eigen_boot.append(bev.cpu().numpy())
                    if bi%25==0:
                        progress=dict(phase='spectral-resampling',teacher=key,t=tv,replicate=bi,total=m['bootstrap_samples'],seconds=time.time()-tick,job=os.environ.get('SLURM_JOB_ID'));save(HERE/'progress.json',progress);print(json.dumps(progress),flush=True)
                b=np.array([be,bn,bp]).T;boot_output[key,tv]=b
                np.savez_compressed(OUT/f'bootstrap-{key}-t{tv}.npz',energy_per_coordinate=b[:,0],k95=b[:,1],participation_rank=b[:,2],eigenvalues=np.array(eigen_boot),orthogonality_cross=(counts@cross/4000).cpu().numpy())
            save(OUT/'spectral-progress.json',dict(records=records,controls=controls,negative_control=negative));np.savez_compressed(OUT/'spectra.npz',**spectra)
    # Pairwise teacher disagreement on exactly the same held-out inputs.
    agreement=[];keys=[c['id'] for c in m['cases']]
    for i,key in enumerate(keys):
        for other in keys[i+1:]:
            for tv in m['times']:
                ra=central[key,tv].astype(float);rb=central[other,tv].astype(float);delta=np.mean((ra-rb)**2)
                ea=next(r['energy_per_coordinate'] for r in records if r['teacher']==key and r['t']==tv and r['patch']==0);eb=next(r['energy_per_coordinate'] for r in records if r['teacher']==other and r['t']==tv and r['patch']==0)
                agreement.append(dict(teacher_a=key,teacher_b=other,t=tv,disagreement_mse=float(delta),disagreement_over_average_residual_energy=float(delta/((ea+eb)/2))))
    contrasts=[]
    for key in keys:
        high=next(r for r in records if r['teacher']==key and r['t']==.1 and r['patch']==0);low=next(r for r in records if r['teacher']==key and r['t']==.9 and r['patch']==0)
        bh=boot_output[key,.1];bl=boot_output[key,.9]
        contrasts.append(dict(teacher=key,high_noise_t=.1,low_noise_t=.9,energy_ratio_low_over_high=low['energy_per_coordinate']/high['energy_per_coordinate'],k95_difference_low_minus_high=low['k95']-high['k95'],energy_ratio_resampling_range95=np.quantile(bl[:,0]/bh[:,0],[.025,.975]).tolist(),k95_difference_resampling_range95=np.quantile(bl[:,1]-bh[:,1],[.025,.975]).tolist(),participation_difference_low_minus_high=low['participation_rank']-high['participation_rank']))
    save(OUT/'summary.json',dict(complete=True,records=records,controls=controls,negative_control=negative,teacher_agreement=agreement,primary_contrasts=contrasts,bootstrap=dict(samples=m['bootstrap_samples'],seed=m['bootstrap_seed'],interpretation='Descriptive image-resampling ranges conditional on the teacher and fitted baseline; not calibrated confidence intervals for a population Bayes spectrum.'),seconds=time.time()-tick,job=os.environ.get('SLURM_JOB_ID'),manifest_sha256=digest(HERE/'manifest.json'),worker_sha256=digest(__file__)))
    save(HERE/'complete.json',dict(complete=True,records=len(records),teachers=keys,times=m['times'],patches=m['positions'],job=os.environ.get('SLURM_JOB_ID'),analysis_seconds=time.time()-tick));print('SPECTRAL ANALYSIS COMPLETE',time.time()-tick,flush=True)
if __name__=='__main__':main()
