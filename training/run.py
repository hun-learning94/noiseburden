"""Train a configured checkpoint from scratch, with local crop caching by default."""
from pathlib import Path
import argparse,importlib.util,itertools,json,os,sys
from functools import partial
from evaluation.runtime import ROOT,SPEC,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--microbatch',type=int,default=256)
    p.add_argument('--accumulation',type=int);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--resume',default='');p.add_argument('--no-cache',action='store_true');p.add_argument('--dry-run',action='store_true')
    opt=p.parse_args();case=next((c for c in SPEC['cases'] if c['id']==opt.case),None)
    if case is None or case['kind']=='adm':p.error('choose one of the 26 JiT configurations in configs/experiments.json')
    world=int(os.environ.get('WORLD_SIZE',1));accum=opt.accumulation or 1024//(opt.microbatch*world)
    if opt.microbatch*world*accum!=1024:p.error('microbatch × world size × accumulation must equal 1024')
    values=dict(case['training_args']);values.update(data_path=str(opt.data_root.resolve()),output_dir=str(opt.output.resolve()),
        resume=opt.resume,start_epoch=0,epochs=opt.epochs,batch_size=opt.microbatch,grad_accum_steps=accum,
        lr=.0002,weight_decay=0.,warmup_epochs=5,lr_schedule='constant',seed=0,
        online_eval=False,evaluate_gen=False,save_last_freq=1,save_freq=40,
        device='cuda',world_size=world,rank=0,gpu=0,local_rank=-1,dist_url='env://')
    if opt.dry_run:
        print(json.dumps(dict(case=case['id'],source=case['source'],configuration=values,
                         node_local_cache=not opt.no_cache,training_base=SPEC['training_base']),indent=2));return
    if not (opt.data_root/'train').is_dir():p.error('--data-root must contain an ImageNet train/ ImageFolder')
    source=ROOT/case['source'];sys.path.insert(0,str(ROOT/'evaluation/third_party'));sys.path.insert(0,str(source))
    import main_jit,model_jit
    if case['kind']=='width':
        model_jit.JiT_models[values['model']]=partial(model_jit.JiT,depth=12,hidden_size=case['width'],
            num_heads=case['heads'],patch_size=case['patch'],bottleneck_dim=128,in_context_len=32,in_context_start=4)
    elif case['kind']=='edm':
        from edm import EDMDenoiser
        main_jit.Denoiser=EDMDenoiser
    args=main_jit.get_args_parser().parse_args([])
    for key,value in values.items():setattr(args,key,value)
    # Use the frozen accumulation/EMA implementation; discard incomplete final
    # effective batches while keeping the original sampler and augmentation RNG.
    from training.accum_train import train_one_epoch
    class CompleteBatches:
        def __init__(self,loader):self.loader=loader
        def __len__(self):return len(self.loader)//accum*accum
        def __iter__(self):return itertools.islice(iter(self.loader),len(self))
        def __getattr__(self,name):return getattr(self.loader,name)
    def epoch(model,bare,loader,optimizer,device,epoch,log_writer=None,args=None):
        return train_one_epoch(model,bare,CompleteBatches(loader),optimizer,device,epoch,log_writer,args)
    main_jit.train_one_epoch=epoch
    enabled=not opt.no_cache and os.environ.get('JIT_LOCAL_CACHE','1').lower() not in ['0','false','off']
    opt.output.mkdir(parents=True,exist_ok=True)
    if enabled:
        from training_cache.cache import build,cache_directory
        from training_cache.launch import install
        import getpass
        local=Path(os.environ.get('JIT_CACHE_ROOT',str((Path('/scratch') if Path('/scratch').is_dir() else Path('/tmp'))/getpass.getuser()/'noiseburden')))
        crop=source/'util/crop.py';cache=cache_directory(local,opt.data_root.resolve(),case['image_size'],crop)
        record=build(opt.data_root.resolve(),cache,int(os.environ.get('JIT_CACHE_WORKERS','16')),crop,case['image_size'])
        install(opt.data_root.resolve(),cache,crop,case['image_size'])
        if int(os.environ.get('RANK','0'))==0:save(opt.output/'cache-provenance.json',dict(path=str(cache),details=record))
    if int(os.environ.get('RANK','0'))==0:save(opt.output/'reproduction-configuration.json',dict(case=case['id'],args=values,source_hashes=case['source_hashes']))
    main_jit.main(args)

if __name__=='__main__':main()
