"""Held-out final-head affine input recovery, with matched-128 and expanded-2000 tests."""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
M=json.loads((HERE/'manifest.json').read_text());OUT=HERE/'results/input';OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path(os.environ.get('VALIDATION_CACHE',HERE/'cache'))
HELPER=ROOT/'runs/width-readouts-ep200-20261016/runs/i256_eps896/probe.py'
spec=importlib.util.spec_from_file_location('frozen_affine_methods',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    case=next(c for c in M['cases'] if c['id']==a.case);assert case['kind']=='width'
    target=OUT/f'{case["id"]}.json'
    if target.exists() and not a.smoke:return
    tick=time.time();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.48)
    assert h.digest(HELPER)==M['historical_helpers'][str(HELPER)]
    loader_spec=importlib.util.spec_from_file_location('frozen_model_loader',case['loader']);loader=importlib.util.module_from_spec(loader_spec);loader_spec.loader.exec_module(loader)
    model=loader.load_model(case);r,p,width=case['image_size'],case['patch'],case['width']
    ids=np.load(CACHE/'input-probe-indices.npy');images=np.load(CACHE/f'images{r}.npy',mmap_mode='r');labels=np.load(CACHE/'labels.npy',mmap_mode='r');noise=np.load(CACHE/f'input-noise{r}.npy',mmap_mode='r')
    assert h.digest(CACHE/'input-probe-indices.npy')==M['cache_hashes']['input-probe-indices.npy']
    values={};handle=model.net.final_layer.linear.register_forward_pre_hook(lambda mod,args:values.__setitem__('h',args[0].detach().float()))
    features=[];targets=[];native=[];parity=0.
    count=24 if a.smoke else len(ids)
    with torch.no_grad():
        for lo in range(0,count,32):
            hi=min(lo+32,count);ii=ids[lo:hi]
            x=torch.from_numpy(np.array(images[ii])).cuda().float()/127.5-1
            e=torch.from_numpy(np.array(noise[lo:hi])).cuda();y=torch.from_numpy(np.array(labels[ii])).cuda()
            z=.5*x+.5*e;t=torch.full((len(x),),.5,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw=model.net(z,t,y);replay=model.net.final_layer.linear(values['h'])
            parity=max(parity,float((h.unpatch(replay.float(),p)-raw.float()).abs().max()))
            assert values['h'].shape==(len(x),256,width)
            features.append(values['h']);targets.append(h.patches(z,p))
            raw=raw.float();clean=raw if case['pred']=='x' else 2*z-raw
            native.append((clean.double()-x.double()).square().mean((1,2,3)))
    handle.remove();assert parity<1e-5
    f=torch.cat(features);z=torch.cat(targets);errors=torch.cat(native);del features,targets,native
    with torch.no_grad():
        decoded,fit=h.fit_probe(f,z,('z',),16 if a.smoke else 384,4 if a.smoke else 128)
        start=20 if a.smoke else 512
        records=[]
        for n in ([4] if a.smoke else [128,2000]):
            boot=h.bootstrap_weights(n,'cuda');score=h.metrics(decoded[:n],z[start:start+n],boot)
            m=errors[start:start+n].double();quant=torch.tensor([.025,.975],device='cuda',dtype=torch.float64)
            records.append(dict(test_images=n,input=score,native_clean_mse=float(m.mean()),native_clean_mse_ci95=torch.quantile(boot@m,quant).cpu().tolist(),native_clean_mse_by_image=m.cpu().tolist()))
    h.save(OUT/f'{case["id"]}{"-smoke" if a.smoke else ""}.json',dict(complete=True,case=case,protocol_sha256=h.digest(HERE/'manifest.json'),worker_sha256=h.digest(__file__),historical_affine_helper_sha256=h.digest(HELPER),fit=fit,fit_images=16 if a.smoke else 384,dev_images=4 if a.smoke else 128,t=.5,head_reconstruction_max_abs=parity,records=records,seconds=time.time()-tick,job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,precision='BF16 backbone and FP64 affine solve/metrics; TF32 disabled',split_note=M['split_note']))
    print('INPUT COMPLETE',case['id'],[(v['test_images'],v['input']['r2_percent']) for v in records],time.time()-tick,flush=True)

if __name__=='__main__':main()
