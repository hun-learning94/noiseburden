"""Adapter around the frozen historical denoising evaluator; validation arrays at both sizes."""
import argparse,importlib.util,json,os
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'runs/denoising-probes-20261023/worker.py'
spec=importlib.util.spec_from_file_location('frozen_native_evaluator',OLD)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
w.HERE=HERE;w.MAN=json.loads((HERE/'manifest.json').read_text());w.CACHE=Path(os.environ.get('VALIDATION_CACHE',HERE/'cache'))
w.OUT=HERE/'results/native';w.OUT.mkdir(parents=True,exist_ok=True)

def batch(c,lo,hi):
    r=c['image_size'];arr=np.load(w.CACHE/f'images{r}.npy',mmap_mode='r')
    x=torch.from_numpy(np.array(arr[lo:hi])).cuda().float()/127.5-1
    y=torch.from_numpy(np.array(np.load(w.CACHE/'labels.npy',mmap_mode='r')[lo:hi])).cuda()
    e=torch.stack([torch.randn((3,r,r),generator=torch.Generator().manual_seed(w.MAN['noise_seed']+i)) for i in range(lo,hi)]).cuda()
    return x,e,y

def provenance(c):
    return dict(case=c,protocol_sha256=w.digest(HERE/'manifest.json'),worker_sha256=w.digest(__file__),historical_worker=str(OLD),historical_worker_sha256=w.digest(OLD),cache_sha256=w.MAN['cache_hashes'],torch=torch.__version__,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),precision='BF16 backbone; FP32 conversion; TF32 disabled',test_note=w.MAN['split_note'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.48)
    assert w.digest(OLD)==w.MAN['historical_helpers'][str(OLD)]
    w.batch=batch;w.provenance=provenance
    c=next(c for c in w.MAN['cases'] if c['id']==a.case)
    w.native(c,a.smoke)
