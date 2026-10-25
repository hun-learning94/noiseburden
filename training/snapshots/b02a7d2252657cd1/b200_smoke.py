"""Two real optimizer updates and finite sampling, without changing production state."""
from pathlib import Path
import json,os,time
import torch
import torch.distributed as dist
import continue_width as width
import main_jit

root=Path(os.environ['OUTPUT_ROOT']);case=width.CASE
source=root/'train/checkpoint-last.pth';before_stat=source.stat()
checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
info=width.validate_checkpoint(checkpoint);width.register()
a=main_jit.get_args_parser().parse_args([])
for k,v in vars(checkpoint['args']).items():setattr(a,k,v)
a.epochs=checkpoint['epoch']+2;a.batch_size=case['microbatch'];a.grad_accum_steps=case['accumulation']
a.world_size=1;a.dist_url='env://';a.dist_on_itp=False
a.data_path=os.environ['IMAGENET_PATH'];a.num_workers=8
a.output_dir=str(root/'b200-smoke');a.resume=str(source.parent)
a.max_steps=2*a.grad_accum_steps;a.log_freq=1;a.bench_warmup=0
a.online_eval=False;a.evaluate_gen=False;a.eval_completed_epochs=[]
a.save_rng_state=True;a.ema_snapshot_epochs=[]
a.cfg=1.;a.num_sampling_steps=2;a.sampling_method='heun'
a.output_shortcut='none';a.exclude_image_manifest=''
assert a.batch_size*a.grad_accum_steps==1024
del checkpoint
original_train=main_jit.train_one_epoch
result={};begin=time.monotonic()
def train(*args,**kwargs):
    model=args[1];optimizer=args[3]
    before={int(s['step']) for s in optimizer.state.values()}
    original_train(*args,**kwargs)
    after={int(s['step']) for s in optimizer.state.values()}
    assert len(before)==1 and after=={next(iter(before))+2},(before,after)
    model.eval()
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        samples=model.generate(torch.arange(16,device='cuda'))
    assert samples.shape==(16,3,case['image_size'],case['image_size']) and torch.isfinite(samples).all()
    result.update(info,updates_before=list(before),updates_after=list(after),
                  finite_samples=16,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,global_batch=1024)
main_jit.train_one_epoch=train
main_jit.main(a)
assert (source.stat().st_size,source.stat().st_mtime_ns)==(before_stat.st_size,before_stat.st_mtime_ns)
result['seconds']=time.monotonic()-begin
(root/'b200-smoke/PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
print('B200_WIDTH_SMOKE_PASSED',json.dumps(result),flush=True)
if dist.is_initialized():dist.destroy_process_group()
