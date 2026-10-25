"""One frozen teacher per process, preserving each historical JiT module namespace."""
import argparse,gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;M=json.loads((HERE/'manifest.json').read_text());OUT=HERE/'results'
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def patchify(x):
    return torch.stack([x[:,:,r*16:(r+1)*16,c*16:(c+1)*16].permute(0,2,3,1).reshape(len(x),768) for r,c in M['positions']],1)
def load(case):
    assert digest(case['checkpoint'])==case['checkpoint_sha256']
    for name,value in case['source_hashes'].items():assert digest(Path(case['source'])/name)==value,name
    if case['kind']=='adm':
        sys.path.insert(0,case['source']);from guided_diffusion.script_util import create_model
        model=create_model(image_size=256,num_channels=256,num_res_blocks=2,learn_sigma=True,class_cond=True,attention_resolutions='32,16,8',num_heads=4,num_head_channels=64,resblock_updown=True,use_fp16=False,use_scale_shift_norm=True)
        ck=torch.load(case['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck,strict=True);del ck
        model.float().cuda().eval().requires_grad_(False)
    else:
        assert digest(case['loader'])==case['loader_sha256']
        spec=importlib.util.spec_from_file_location('historical_spectral_teacher',case['loader']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        model=mod.load_model(case);assert model.net.x_embedder.__class__.__name__=='LinearPatchEmbed'
        model.float().eval().requires_grad_(False)
    return model
@torch.no_grad()
def predict(model,case,x,eps,labels,tv,bf16=False):
    if case['kind']=='adm':
        s=(1-tv)/tv;alpha=M['adm_mapping'][str(tv)]['alpha_bar'];tau=M['adm_mapping'][str(tv)]['fractional_timestep']
        vp=(x+s*eps)*np.sqrt(alpha);time_input=torch.full((len(x),),tau,device='cuda')
        raw=model(vp,time_input,y=labels)[:,:3].float();clean=x+s*eps-s*raw
    else:
        tt=torch.full((len(x),),tv,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):raw=model.net(tv*x+(1-tv)*eps,tt,labels)
        raw=raw.float();clean=raw if case['pred']=='x' else (tv*x+(1-tv)*eps-(1-tv)*raw)/tv
    assert torch.isfinite(clean).all()
    return clean
def main():
    args=argparse.ArgumentParser();args.add_argument('--teacher',required=True);args.add_argument('--smoke',action='store_true');a=args.parse_args()
    assert torch.cuda.is_available() and digest(__file__)==M['worker_hashes']['extract.py']
    case=next(c for c in M['cases'] if c['id']==a.teacher);key=case['id'];done=OUT/f'{key}-complete.json'
    if done.exists() and not a.smoke:print('Already complete',key);return
    tick=time.time();model=load(case);cache=Path(os.environ['SPECTRAL_CACHE'])
    images=np.load(cache/'images256.npy',mmap_mode='r');labels=np.load(cache/'labels.npy',mmap_mode='r')
    for name,value in M['cache_hashes'].items():
        if name!='samples.json':assert digest(cache/name)==value,name
    if a.smoke:
        lo,hi=8000,8002;x=torch.from_numpy(np.array(images[lo:hi])).cuda().float()/127.5-1;y=torch.from_numpy(np.array(labels[lo:hi])).cuda()
        eps=torch.stack([torch.randn((3,256,256),generator=torch.Generator().manual_seed(M['noise_seed']+i)) for i in range(lo,hi)]).cuda()
        expected_noise=np.load(HERE/'data/noise-eval.npy',mmap_mode='r')[lo-6000:hi-6000]
        np.testing.assert_array_equal(patchify(eps).cpu().numpy(),expected_noise)
        expected_clean=np.load(HERE/'data/clean.npy',mmap_mode='r')[lo:hi]
        np.testing.assert_allclose(patchify(x).cpu().numpy(),expected_clean,rtol=0,atol=2e-7)
        for tv in [.05,.5,.95]:
            pred=predict(model,case,x,eps,y,tv);print('SMOKE',key,tv,float((pred-x).square().mean()),tuple(patchify(pred).shape),flush=True)
        save(OUT/f'{key}-smoke.json',dict(complete=True,gpu=torch.cuda.get_device_name(),seconds=time.time()-tick,job=os.environ.get('SLURM_JOB_ID')));return
    state=OUT/f'{key}-progress.json';start=6000
    if state.exists():
        old=json.loads(state.read_text());assert old['worker_sha256']==M['worker_hashes']['extract.py'];start=6000+old['images_done']
    mode='r+' if start>6000 else 'w+'
    preds=np.lib.format.open_memmap(OUT/f'{key}-predictions.npy',mode=mode,dtype=np.float32,shape=(len(M['times']),4000,5,768))
    errors=np.lib.format.open_memmap(OUT/f'{key}-mse.npy',mode=mode,dtype=np.float64,shape=(len(M['times']),4000))
    precision=[]
    for lo in range(start,10000,M['batch_size']):
        hi=min(lo+M['batch_size'],10000);x=torch.from_numpy(np.array(images[lo:hi])).cuda().float()/127.5-1;y=torch.from_numpy(np.array(labels[lo:hi])).cuda()
        eps=torch.stack([torch.randn((3,256,256),generator=torch.Generator().manual_seed(M['noise_seed']+i)) for i in range(lo,hi)]).cuda()
        if lo==6000:
            expected_noise=np.load(HERE/'data/noise-eval.npy',mmap_mode='r')[:hi-lo]
            np.testing.assert_array_equal(patchify(eps).cpu().numpy(),expected_noise)
            expected_clean=np.load(HERE/'data/clean.npy',mmap_mode='r')[lo:hi]
            np.testing.assert_allclose(patchify(x).cpu().numpy(),expected_clean,rtol=0,atol=2e-7)
        for ti,tv in enumerate(M['times']):
            pred=predict(model,case,x,eps,y,tv)
            preds[ti,lo-6000:hi-6000]=patchify(pred).cpu().numpy();errors[ti,lo-6000:hi-6000]=(pred-x).square().mean((1,2,3)).cpu().numpy()
            if lo==6000 and case['kind']!='adm':
                other=predict(model,case,x,eps,y,tv,bf16=True)
                precision.append(dict(t=tv,fp32_mse=float((pred-x).square().mean()),bf16_mse=float((other-x).square().mean()),bf16_fp32_disagreement=float((other-pred).square().mean())))
        if (hi-6000)%128==0 or hi==10000:
            preds.flush();errors.flush();torch.cuda.synchronize()
            progress=dict(teacher=key,images_done=hi-6000,images_total=4000,seconds=time.time()-tick,worker_sha256=M['worker_hashes']['extract.py'],gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,job=os.environ.get('SLURM_JOB_ID'))
            save(state,progress);save(HERE/'progress.json',progress);print(json.dumps(progress),flush=True)
            if precision:save(OUT/f'{key}-precision.json',precision)
    preds.flush();errors.flush()
    save(done,dict(complete=True,teacher=key,checkpoint_sha256=case['checkpoint_sha256'],manifest_sha256=digest(HERE/'manifest.json'),worker_sha256=M['worker_hashes']['extract.py'],predictions_sha256=digest(OUT/f'{key}-predictions.npy'),mse_sha256=digest(OUT/f'{key}-mse.npy'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,precision='FP32; TF32 disabled; no clipping/guidance',seconds=time.time()-tick,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,job=os.environ.get('SLURM_JOB_ID')))
    print('EXTRACTION COMPLETE',key,time.time()-tick,flush=True)
if __name__=='__main__':main()
