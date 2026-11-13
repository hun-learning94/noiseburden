"""Historical single-rank batch-eight JiT FID-10K with EMA1 and Heun50."""
from pathlib import Path
import argparse,copy,json,os,random,socket,sys,time
import numpy as np
import torch
import torch.distributed as dist
from evaluation.runtime import ROOT,digest,save
from .loaders import load

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('work/revision'))
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fid-stats',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,args,c=load(a.workspace/a.case)
    args.output_dir=str(a.output.resolve());args.num_images=10000;args.class_num=1000
    args.cfg=1.;args.sampling_method='heun';args.num_sampling_steps=50;args.gen_bsz=8;args.interval_min=.1;args.interval_max=1.
    model.cfg_scale=1.;model.cfg_interval=(.1,1.);model.method='heun';model.steps=50;model.ema_params1=list(model.parameters())
    a.output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=0,world_size=1)
    import engine_jit
    original=engine_jit.torch_fidelity.calculate_metrics
    def measure(**kwargs):
        images=Path(kwargs['input1']);assert {p.name for p in images.glob('*.png')}=={f'{i:05d}.png' for i in range(10000)}
        kwargs['fid_statistics_file']=str(a.fid_stats)
        result=original(**kwargs)
        save(a.output/'fid.json',dict(complete=True,case=c['id'],fid=float(result['frechet_inception_distance']),
             num_images=10000,verified_image_count=10000,cfg=1.,sampling_method='heun',sampling_steps=50,seed=0,
             generation_batch_per_rank=8,world_size=1,ema='EMA1',completed_epochs=c['completed_epochs'],
             checkpoint_sha256=c['checkpoint_sha256'],reference_sha256=digest(a.fid_stats)))
        return result
    engine_jit.torch_fidelity.calculate_metrics=measure
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(a.output/'tensorboard'))
    try:
        # Reset after model construction, preserving the historical batch-eight draws.
        with torch.random.fork_rng(),torch.no_grad():
            torch.manual_seed(0);np.random.seed(0);random.seed(0)
            engine_jit.evaluate(model,args,c['completed_epochs']-1,batch_size=8,log_writer=writer)
    finally:
        writer.close();engine_jit.torch_fidelity.calculate_metrics=original;dist.destroy_process_group()
if __name__=='__main__':main()
