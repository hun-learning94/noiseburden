"""Continue audited width checkpoints, keeping model/loss/Adam/EMA unchanged."""
import argparse,copy,fcntl,hashlib,json,math,os,random,shutil,time
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import model_jit,main_jit,engine_jit
import util.misc as misc
from torch.utils.tensorboard import SummaryWriter

ROOT=Path(os.environ['OUTPUT_ROOT'])
CASE=json.loads((ROOT/'case.json').read_text())

def publish(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)

def validate_checkpoint(c):
    a=c['args'];w=CASE['width'];p=CASE['patch'];h=CASE['head_dim']
    assert a.model==CASE['model'] and a.img_size==CASE['image_size']
    assert a.pred_space==CASE['pred'] and a.loss_space=='x' and a.loss_impl=='weighted_v'
    assert a.patch_embed=='linear' and getattr(a,'output_shortcut','none')=='none'
    assert a.batch_size*getattr(a,'grad_accum_steps',1)*a.world_size==1024
    assert a.lr==.0002 and a.lr_schedule=='constant' and a.weight_decay==0
    assert a.ema_decay1==.9999 and a.ema_decay2==.9996
    assert a.seed==0 and not getattr(a,'exclude_image_manifest','')
    assert 39<=c['epoch']<=79
    steps={int(v['step']) for v in c['optimizer']['state'].values()}
    assert steps=={(c['epoch']+1)*1251},steps
    for key in ['model','model_ema1','model_ema2']:
        s=c[key];assert s['net.x_embedder.proj.weight'].shape==(w,3,p,p)
        assert s['net.final_layer.linear.weight'].shape==(p*p*3,w)
        assert s['net.blocks.0.attn.q_norm.weight'].numel()==h
        assert all(s[f'net.blocks.{i}.attn.qkv.weight'].shape==(3*w,w) and
                   s[f'net.blocks.{i}.attn.proj.weight'].shape==(w,w) for i in range(12))
        assert 'net.blocks.12.attn.qkv.weight' not in s
    return dict(completed_epochs=c['epoch']+1,optimizer_updates=next(iter(steps)),
                full_attention=True,head_dim=h,extra_patch_bottleneck=False,
                old_world_size=a.world_size,rng_states=len(c.get('rng_states') or []))

def register():
    model_jit.JiT_models[CASE['model']]=partial(model_jit.JiT,
        depth=12,hidden_size=CASE['width'],num_heads=CASE['width']//CASE['head_dim'],
        bottleneck_dim=128,in_context_len=32,in_context_start=4,patch_size=CASE['patch'])

original_save=misc.save_model
def save_checkpoint(*args,**kwargs):
    # Serialize large rolling saves across this batch, without serializing training.
    lock=None
    if misc.is_main_process():
        lock=(ROOT.parent/'.width80-save.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX)
        needed=CASE['checkpoint_bytes']+5*2**30
        assert shutil.disk_usage(ROOT).free>needed, 'Insufficient atomic-save headroom'
    try:
        original_save(*args,**kwargs)
        if misc.is_main_process():
            publish(ROOT/'progress.json',dict(completed_epochs=kwargs['epoch']+1,
                updates=(kwargs['epoch']+1)*1251,target_epochs=80,
                saved_at=time.time(),free_gib=shutil.disk_usage(ROOT).free/2**30))
    finally:
        if lock is not None:lock.close()

original_train=main_jit.train_one_epoch
def train_epoch(model,model_without_ddp,data_loader,optimizer,device,epoch,log_writer=None,args=None):
    assert len(data_loader)//args.grad_accum_steps==1251
    before={int(v['step']) for v in optimizer.state.values()}
    assert before=={epoch*1251},before
    original_train(model,model_without_ddp,data_loader,optimizer,device,epoch,log_writer,args)
    after={int(v['step']) for v in optimizer.state.values()}
    assert after=={(epoch+1)*1251},after

original_metrics=engine_jit.torch_fidelity.calculate_metrics
def evaluate(model,args,epoch,batch_size=128,log_writer=None):
    completed=epoch+1
    names=['EMA1','EMA2'] if completed==80 else ['EMA1']
    original_params=model.ema_params1
    base=args.output_dir
    for name in names:
        out=ROOT/'eval'/f'ep{completed:03d}'/name
        if (out/'metrics.json').exists():
            result=json.loads((out/'metrics.json').read_text())
            assert result['completed_epochs']==completed and result['ema']==name and result['verified_image_count']==10000
            continue
        out.mkdir(parents=True,exist_ok=True)
        assert shutil.disk_usage(ROOT).free>5*2**30
        args.output_dir=str(out)
        model.ema_params1=original_params if name=='EMA1' else model.ema_params2
        writer=SummaryWriter(str(out)) if misc.is_main_process() else None
        def metrics(**kw):
            folder=Path(kw['input1'])
            assert {p.name for p in folder.glob('*.png')}=={f'{i:05d}.png' for i in range(10000)}
            s=args.img_size;grid=Image.new('RGB',(8*s,8*s))
            for k,label in enumerate(np.linspace(0,999,64,dtype=int)):
                with Image.open(folder/f'{label*10:05d}.png') as im:grid.paste(im.convert('RGB'),(k%8*s,k//8*s))
            grid.save(out/'samples.png')
            m=original_metrics(**kw)
            assert math.isfinite(float(m['frechet_inception_distance']))
            row=dict(case=CASE['key'],width=CASE['width'],image_size=args.img_size,
                     patch_size=CASE['patch'],pred=args.pred_space,loss='x',global_batch=1024,
                     completed_epochs=completed,ema=name,ema_decay=.9999 if name=='EMA1' else .9996,
                     fid=float(m['frechet_inception_distance']),inception_score=float(m['inception_score_mean']),
                     cfg=1.,num_images=10000,verified_image_count=10000,sampling_method='heun',sampling_steps=50,
                     world_size=misc.get_world_size(),generation_batch_per_rank=batch_size,seed=0,
                     reference_sha256=hashlib.sha256(Path(f'fid_stats/jit_in{args.img_size}_stats.npz').read_bytes()).hexdigest())
            publish(out/'metrics.json',row);print('VERIFIED_WIDTH80_FID',json.dumps(row),flush=True)
            return m
        engine_jit.torch_fidelity.calculate_metrics=metrics
        numpy_state=np.random.get_state();python_state=random.getstate()
        try:
            with torch.random.fork_rng(),torch.no_grad():
                torch.manual_seed(misc.get_rank());np.random.seed(misc.get_rank());random.seed(misc.get_rank())
                engine_jit.evaluate(model,args,epoch,batch_size=batch_size,log_writer=writer)
        finally:
            np.random.set_state(numpy_state);random.setstate(python_state)
            if writer:writer.close()
            model.ema_params1=original_params;args.output_dir=base
            engine_jit.torch_fidelity.calculate_metrics=original_metrics
        # Restart after metric publication can leave sample PNGs. Only this verified output.
        if misc.is_main_process():
            for p in out.glob('heun-steps*'):
                if p.is_dir():shutil.rmtree(p)
        dist.barrier()

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--verify-only',action='store_true');options=cli.parse_args()
    register()
    cp=ROOT/'train/checkpoint-last.pth'
    c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    info=validate_checkpoint(c)
    a=main_jit.get_args_parser().parse_args([])
    for k,v in vars(c['args']).items():setattr(a,k,v)
    a.epochs=80;a.batch_size=CASE['microbatch'];a.grad_accum_steps=CASE['accumulation']
    a.data_path=os.environ['IMAGENET_PATH'];a.num_workers=12 if CASE['gpus']==2 else 16
    a.output_dir=str(ROOT/'train');a.resume=a.output_dir
    a.save_last_freq=1;a.save_freq=1000000;a.save_rng_state=True;a.ema_snapshot_epochs=[]
    a.online_eval=True;a.eval_completed_epochs=[60,80];a.evaluate_gen=False;a.max_steps=0
    a.cfg=1.;a.num_images=10000;a.sampling_method='heun';a.num_sampling_steps=50
    a.interval_min=.1;a.interval_max=1.;a.gen_bsz=64 if CASE['width']>=1536 else 128
    if CASE['gpus']==1:a.gen_bsz=256
    a.output_shortcut='none';a.exclude_image_manifest=''
    a.world_size=int(os.environ.get('WORLD_SIZE',CASE['gpus']));a.dist_url='env://';a.dist_on_itp=False
    assert a.world_size==CASE['gpus'] and a.batch_size*a.grad_accum_steps*a.world_size==1024
    info.update(case=CASE['key'],target_epochs=80,global_batch=1024,world_size=a.world_size,
                microbatch=a.batch_size,accumulation=a.grad_accum_steps,torch_version=torch.__version__)
    if int(os.environ.get('RANK','0'))==0:publish(ROOT/'startup-audit.json',info);print('WIDTH80_RESUME_AUDIT',json.dumps(info),flush=True)
    del c
    if options.verify_only:return
    main_jit.train_one_epoch=train_epoch;main_jit.evaluate=evaluate;misc.save_model=save_checkpoint
    main_jit.main(a)
    if misc.is_main_process():
        final=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
        result=validate_checkpoint(final);assert result['completed_epochs']==80
        assert len(final['rng_states'])==CASE['gpus']
        for name in ['EMA1','EMA2']:assert (ROOT/'eval/ep080'/name/'metrics.json').exists()
        publish(ROOT/'complete.json',dict(result,metrics={name:json.loads((ROOT/'eval/ep080'/name/'metrics.json').read_text()) for name in ['EMA1','EMA2']}))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
