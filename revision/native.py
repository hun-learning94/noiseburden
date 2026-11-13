"""Historical BF16 native-loss grid, paired with the arXiv width baseline."""
from pathlib import Path
import argparse,importlib.util,json,os
os.environ['TORCHDYNAMO_DISABLE']='1'
import numpy as np
import torch
from evaluation.runtime import ROOT,save
from .loaders import load

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('work/revision'))
    p.add_argument('--case',required=True);p.add_argument('--cache',type=Path,required=True,help='Full 10000-image validation cache')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_capability()[0]<8:raise RuntimeError('The archived BF16 loss protocol requires a GPU with native BF16 support')
    model,args,c=load(a.workspace/a.case)
    m=importlib.util.spec_from_file_location('historical',ROOT/'revision/frozen/control_diagnostics/historical_methods.py')
    h=importlib.util.module_from_spec(m);m.loader.exec_module(h)
    images=np.load(a.cache/f'images{c["image_size"]}.npy',mmap_mode='r');labels=np.load(a.cache/'labels.npy')
    records=[];arrays={};seed=json.loads((ROOT/'configs/experiments.json').read_text())['noise_seed']
    for t in [.05,.1,.2,.5,.8,.9,.95]:
        xx=[];ee=[]
        with torch.no_grad():
            for lo in range(8000,10000,16):
                hi=min(lo+16,10000);res=c['image_size']
                x=torch.from_numpy(np.array(images[lo:hi])).cuda().float()/127.5-1
                eps=torch.stack([torch.randn((3,res,res),generator=torch.Generator().manual_seed(seed+i)) for i in range(lo,hi)]).cuda()
                clean,pred,skip=h.predict(model,c,x,eps,torch.from_numpy(labels[lo:hi]).cuda(),t)
                xloss=(clean-x).square().flatten(1).mean(1);eloss=(pred-eps).square().flatten(1).mean(1)
                assert torch.allclose(xloss,((1-t)/t)**2*eloss,atol=5e-5,rtol=1e-4)
                xx.extend(xloss.cpu().tolist());ee.extend(eloss.cpu().tolist())
        arrays[f'x_{t}']=xx;arrays[f'eps_{t}']=ee
        records.append(dict(t=t,sigma=(1-t)/t,clean_mse=h.stats(xx),noise_mse=h.stats(ee)))
    a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'native-per-image.npz',**arrays)
    save(a.output/'native.json',dict(complete=True,records=records,checkpoint_sha256=c['checkpoint_sha256'],completed_epochs=c['completed_epochs'],
         precision='BF16 backbone; FP32 native conversion and MSE; shared CPU per-image noise seeds',ema='EMA1'))
if __name__=='__main__':main()
