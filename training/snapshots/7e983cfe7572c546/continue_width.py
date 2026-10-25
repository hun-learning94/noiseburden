"""Continue audited width checkpoints, keeping model/loss/Adam/EMA unchanged."""
import argparse,copy,fcntl,hashlib,json,math,os,random,shutil,time,sys,subprocess
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
ROOT=Path(os.environ['OUTPUT_ROOT'])
TASK=ROOT.parent.parent
CASE=json.loads((ROOT/'case.json').read_text())
sys.path.insert(0,str(ROOT/'code'))
import model_jit,main_jit,engine_jit
import util.misc as misc
from torch.utils.tensorboard import SummaryWriter

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
    assert 79<=c['epoch']<=159
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
        lock=(ROOT.parent/'.width160-save.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX)
        needed=CASE['checkpoint_bytes']+5*2**30
        assert shutil.disk_usage(ROOT).free>needed, 'Insufficient atomic-save headroom'
    try:
        original_save(*args,**kwargs)
        if kwargs['epoch']+1 in (120,160) and kwargs.get('epoch_name')=='last':
            archived=dict(kwargs,epoch_name=f"ep{kwargs['epoch']+1:03d}")
            original_save(*args,**archived)
        if misc.is_main_process():
            publish(ROOT/'progress.json',dict(completed_epochs=kwargs['epoch']+1,
                updates=(kwargs['epoch']+1)*1251,target_epochs=160,
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
    assert after=={epoch*1251+(args.max_steps//args.grad_accum_steps if args.max_steps else 1251)},after

original_metrics=engine_jit.torch_fidelity.calculate_metrics
def evaluate(model,args,epoch,batch_size=128,log_writer=None):
    completed=epoch+1
    retained=ROOT/'train'/f'checkpoint-ep{completed:03d}.pth'
    if not retained.exists():
        # Resume may follow a preemption between the rolling and retained saves.
        latest=ROOT/'train/checkpoint-last.pth'
        check=torch.load(latest,map_location='cpu',mmap=True,weights_only=False)
        assert check['epoch']+1==completed
        del check
        temp=retained.with_suffix('.tmp');shutil.copy2(latest,temp);temp.replace(retained)
    names=['EMA1','EMA2']
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
            publish(out/'metrics.json',row);print('VERIFIED_WIDTH160_FID',json.dumps(row),flush=True)
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
        (dist.barrier() if dist.is_initialized() else None)
    if misc.is_main_process():
        subprocess.run([sys.executable,str(TASK/'evaluate_saved.py'),'--run',str(ROOT),'--epoch',str(completed)],check=True)
    (dist.barrier() if dist.is_initialized() else None)

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--verify-only',action='store_true');cli.add_argument('--smoke',action='store_true');options=cli.parse_args()
    if not options.verify_only:
        gate=json.loads((TASK/'validation.json').read_text())
        assert gate['complete'], 'Matched checkpoint evaluations must pass first'
    for name,h in CASE['continuation_source_hashes'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==h,name
    register()
    cp=ROOT/'train/checkpoint-last.pth'
    c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    info=validate_checkpoint(c)
    if options.verify_only:
        print('STRUCTURAL_CHECK_OK',CASE['key'],json.dumps(info),flush=True)
        return
    a=main_jit.get_args_parser().parse_args([])
    for k,v in vars(c['args']).items():setattr(a,k,v)
    a.epochs=160
    memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30
    a.batch_size=512 if memory_gib>=100 else 256 if memory_gib>=40 else 128 if memory_gib>=23 else 64
    a.grad_accum_steps=1024//a.batch_size
    a.data_path=os.environ['IMAGENET_PATH'];a.num_workers=6
    a.output_dir=str(ROOT/'train');a.resume=a.output_dir
    a.save_last_freq=1;a.save_freq=1000000;a.save_rng_state=True;a.ema_snapshot_epochs=[]
    a.online_eval=True;a.eval_completed_epochs=[120,160];a.evaluate_gen=False;a.max_steps=0
    a.cfg=1.;a.num_images=10000;a.sampling_method='heun';a.num_sampling_steps=50
    a.interval_min=.1;a.interval_max=1.;a.gen_bsz=64 if CASE['width']>=1536 else 128
    if CASE['gpus']==1:a.gen_bsz=min(a.batch_size,256)
    a.output_shortcut='none';a.exclude_image_manifest=''
    a.world_size=int(os.environ.get('WORLD_SIZE',CASE['gpus']));a.dist_url='env://';a.dist_on_itp=False
    assert a.world_size==CASE['gpus'] and a.batch_size*a.grad_accum_steps*a.world_size==1024
    info.update(case=CASE['key'],target_epochs=160,global_batch=1024,world_size=a.world_size,
                microbatch=a.batch_size,accumulation=a.grad_accum_steps,torch_version=torch.__version__,
                device=torch.cuda.get_device_name(),memory_gib=memory_gib,command=sys.argv,
                initial_checkpoint_sha256=CASE['checkpoint_sha256'],source_hashes=CASE['continuation_source_hashes'])
    if int(os.environ.get('RANK','0'))==0:publish(ROOT/'startup-audit.json',info);print('WIDTH160_RESUME_AUDIT',json.dumps(info),flush=True)
    del c
    if options.verify_only:return
    main_jit.train_one_epoch=train_epoch;main_jit.evaluate=evaluate;misc.save_model=save_checkpoint
    if options.smoke:
        a.max_steps=2*a.grad_accum_steps;a.online_eval=False
    main_jit.main(a)
    if options.smoke:
        publish(ROOT/'smoke.json',dict(info,passed=True,optimizer_updates_tested=2,checkpoint_unchanged=True))
        (dist.barrier() if dist.is_initialized() else None);(dist.destroy_process_group() if dist.is_initialized() else None);return
    if misc.is_main_process():
        final=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
        result=validate_checkpoint(final);assert result['completed_epochs']==160
        assert len(final['rng_states'])==CASE['gpus']
        for name in ['EMA1','EMA2']:assert (ROOT/'eval/ep160'/name/'metrics.json').exists()
        publish(ROOT/'complete.json',dict(result,metrics={name:json.loads((ROOT/'eval/ep160'/name/'metrics.json').read_text()) for name in ['EMA1','EMA2']}))
    (dist.barrier() if dist.is_initialized() else None);(dist.destroy_process_group() if dist.is_initialized() else None)

if __name__=='__main__':main()
