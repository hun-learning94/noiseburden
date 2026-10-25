"""Generate class-balanced FID samples from a configured EMA1 checkpoint."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
import numpy as np
import torch
from PIL import Image
from .runtime import ROOT,digest,save

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('work/evaluation'))
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fid-stats',type=Path);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--num-images',type=int,default=10000);a=p.parse_args()
    if a.num_images%1000:p.error('use an equal number of samples for each of the 1000 classes')
    info=json.loads((a.workspace/'runtime.json').read_text());run=Path(info['run'])
    os.environ['VALIDATION_CACHE']=info['cache'];sys.path.insert(0,str(ROOT/'evaluation/third_party'))
    spec=importlib.util.spec_from_file_location('native_generation_loader',run/'native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=next(c for c in module.w.MAN['cases'] if c['id']==a.case)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=module.w.load(case)
    model.cfg_scale=1. if case['kind']=='width' else 2.9
    model.cfg_interval=(.1,1.);model.method='heun';model.steps=50
    labels=np.arange(1000).repeat(a.num_images//1000);a.output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for lo in range(0,a.num_images,a.batch_size):
            y=torch.tensor(labels[lo:lo+a.batch_size],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):sample=model.generate(y)
            pixels=np.round(np.clip((sample.float().cpu().numpy()+1)/2,0,1)*255).astype(np.uint8)
            for i,x in enumerate(pixels):Image.fromarray(x.transpose(1,2,0)).save(a.output/f'{lo+i:05d}.png')
            if lo%1000==0:print('Generated',lo,'/',a.num_images,flush=True)
    result=dict(case=a.case,checkpoint_sha256=case['checkpoint_sha256'],num_images=a.num_images,
                sampling_method='heun',sampling_steps=50,guidance=model.cfg_scale,seed=0,batch_size=a.batch_size)
    if a.fid_stats:
        import torch_fidelity
        result['reference_statistics_sha256']=digest(a.fid_stats)
        result['metrics']=torch_fidelity.calculate_metrics(input1=str(a.output),input2=None,
            fid_statistics_file=str(a.fid_stats),cuda=True,isc=True,fid=True,kid=False,prc=False,verbose=False)
    save(a.output/'metrics.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
