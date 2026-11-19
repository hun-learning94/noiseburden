"""Resumable frozen-backbone fitting, held-out MSE, and native P-JiT sampling."""
import argparse
import contextlib
import fcntl
import gc
import hashlib
import io
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import *

sys.path.insert(0, str(SOURCE))
from pyramid_jit.config import SamplerConfig
from pyramid_jit.sampler import MomentumBuffer, adaptive_projected_guidance
from pyramid_jit.noise import StackedRandomGenerator

STOP = False
RUN_STARTED = time.monotonic()
STOP_AFTER_SECONDS = 3 * 60 * 60 + 45 * 60

def stop(*_):
    global STOP
    STOP = True

def checkpoint_stop():
    elapsed = time.monotonic() - RUN_STARTED
    if STOP or elapsed >= STOP_AFTER_SECONDS:
        # All callers are outside atomic output commits. Avoid interpreter/library
        # teardown delaying the resumable exit past the allocation's hard timeout.
        receipt = {'reason': 'signal' if STOP else 'runtime_limit',
                   'elapsed_seconds': elapsed, 'exit_code': 75,
                   'job_id': os.environ.get('PYRAMID_RUN_ID'), 'pid': os.getpid(),
                   'time': time.time()}
        try:
            save(STORE / 'execution' / f"stop-{receipt['job_id']}-{receipt['pid']}.json", receipt)
        except OSError as error:
            receipt['receipt_write_error'] = str(error)
        print('RESUMABLE_STOP', json.dumps(receipt), flush=True)
        sys.stderr.flush()
        os._exit(75)

def atomic_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp)
    tmp.replace(path)

def patches(x):
    b,c,f,h,w=x.shape
    assert (c,f,h,w)==(3,1,512,512)
    return x.reshape(b,3,16,32,16,32).permute(0,2,4,1,3,5).reshape(b,256,3072)

def unpatch(p):
    return p.reshape(len(p),16,16,3,32,32).permute(0,3,1,4,2,5).reshape(len(p),3,1,512,512)

def project(x, rows):
    p=patches(x)
    return unpatch(p-(p@rows.T)@rows)

def source_time(t):
    # Use the exact released float32 Euler grid, including its integer time embedding.
    if abs(t*50-round(t*50))<1e-6:
        return torch.linspace(1.,0.,51,device='cuda')[round(t*50)]
    return torch.tensor(1-t,device='cuda',dtype=torch.float32)

def strength(t,method):
    if method=='baseline':return 0.
    if method=='late_half' and round(t,6)>=.9:return .5
    return 1.

class TextCache:
    def __init__(self):
        self.encoder=None
        self.folder=STORE/'text-features';self.folder.mkdir(exist_ok=True)
    def path(self,caption):
        return self.folder/(hashlib.sha256(caption.encode()).hexdigest()+'.pt')
    def get(self,captions):
        missing=list(dict.fromkeys(c for c in captions if not self.path(c).exists()))
        if missing and self.encoder is None:
            from pyramid_jit.text_encoder import QwenTextEncoder
            self.encoder=QwenTextEncoder(str(STORE/'assets/text'),device=torch.device('cuda'))
            # Prefix masks are assumed by the released feature extraction implementation.
            assert self.encoder.tokenizer.padding_side=='right', 'Check text extraction for left padding'
        for start in range(0,len(missing),TEXT_BATCH):
            checkpoint_stop()
            group=missing[start:start+TEXT_BATCH]
            features=self.encoder(group)
            for caption,feature in zip(group,features):
                assert feature.shape[1]==7680 and feature.shape[0]<=256
                assert torch.isfinite(feature).all()
                dest=self.path(caption)
                tmp=dest.with_name(dest.name+f'.{os.getpid()}.tmp')
                torch.save({'caption':caption,'revision':TEXT_REV,'features':feature.cpu()},tmp)
                # A generation worker may concurrently cache the same fixed caption.
                try:os.link(tmp,dest)
                except FileExistsError:pass
                finally:tmp.unlink(missing_ok=True)
        result=[]
        for c in captions:
            item=torch.load(self.path(c),map_location='cpu',weights_only=True)
            assert item['caption']==c and item['revision']==TEXT_REV
            result.append(item['features'].cuda())
        return result
    def release(self):
        self.encoder=None;gc.collect();torch.cuda.empty_cache()

def text_batch(features):
    return {'text':torch.stack([torch.cat([x,x.new_zeros(256-len(x),7680)])for x in features]).contiguous(),
            'text_lens':torch.tensor([len(x)for x in features],device='cuda',dtype=torch.int32)}

def load_model():
    from pyramid_jit.model import PyramidJiT
    from pyramid_jit.attention import set_attention_backend
    set_attention_backend('sdpa')
    model=PyramidJiT.from_pretrained(str(STORE/'assets/model')).requires_grad_(False)
    rows=torch.from_numpy(np.load(STORE/'assets/encoder_rows.npy')).cuda()
    return model,rows

@torch.inference_mode()
def raw_predictions(model,x,s,cond,negative):
    t_int=(s*1000).long().repeat(len(x))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return (model(x_t=x,t=t_int,**cond),model(x_t=x,t=t_int,**negative))

def corrected(x,pred,s,rows,vbar,mult=1.):
    t=1-s
    gain=vbar*t/(vbar*t*t+4*s*s)
    delta=mult*gain*project(x.float()-t*pred.float(),rows.float())
    return (pred.float()+delta).to(pred.dtype)

@torch.inference_mode()
def sample(model,rows,features,negative_features,seeds,method,coefficients,return_state=False):
    conf=SamplerConfig()
    gen=StackedRandomGenerator(device='cuda',seeds=seeds,sm_count=conf.noise_sm_count)
    x=conf.noise_scale*gen.randn((len(seeds),3,1,512,512),dtype=torch.bfloat16,device='cuda')
    cond=text_batch(features);neg=text_batch([negative_features]*len(seeds))
    ts=torch.linspace(1.,0.,51,device='cuda')
    momentum=MomentumBuffer(momentum=conf.apg_momentum)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(50):
            checkpoint_stop()
            s,sn=ts[i],ts[i+1]
            pc,pu=raw_predictions(model,x,s,cond,neg)
            if method!='baseline' and i>0:
                c=coefficients[key(i/50)]
                # Disable autocast for the small calibration projection only.
                with torch.autocast('cuda',enabled=False):
                    pc=corrected(x,pc,s,rows,c['conditional']['vbar'],strength(i/50,method))
                    pu=corrected(x,pu,s,rows,c['negative']['vbar'],strength(i/50,method))
            vc=(x-pc)/s;vu=(x-pu)/s
            velocity=adaptive_projected_guidance(vc,vu,conf.guidance_scale,momentum,
                eta=conf.apg_eta,rescale=conf.apg_rescale)
            x=x+(sn-s)*velocity
            if not torch.isfinite(x).all():raise FloatingPointError(f'nonfinite step {i}')
    if return_state:return x
    return ((x+1)*127.5).clamp(0,255).to(torch.uint8)

@torch.inference_mode()
def preflight(model,rows,cache):
    dest=STORE/'preflight'/f'{os.environ.get("PYRAMID_RUN_ID","local")}.json'
    caption='A photograph of a dog sitting on grass.'
    features,negative=cache.get([caption,NEGATIVE])
    from pyramid_jit.sampler import generate
    class CachedEncoder:
        def __call__(self,text):
            return [features if c==caption else negative if c==NEGATIVE else None for c in text]
    expected=generate(model,CachedEncoder(),caption,[19181,19182],SamplerConfig(),quiet=True)
    actual=sample(model,rows,[features,features],negative,[19181,19182],'baseline',{})
    expected=torch.stack(expected)
    assert torch.equal(expected,actual),'Native baseline PNG parity failed'
    x=torch.randn(2,3,1,512,512,device='cuda')
    assert torch.equal(unpatch(patches(x)),x)
    q=project(x.double(),rows)
    retained=(patches(q)@rows.T).abs().max().item()
    assert retained<1e-10
    s=source_time(.5)
    pc,pu=raw_predictions(model,x,s,text_batch([features]*2),text_batch([negative]*2))
    assert torch.isfinite(pc).all()and torch.isfinite(pu).all()
    save(dest,{'complete':True,'native_png_exact':True,'retained_component_max':retained,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'backend':'sdpa',
        'host':socket.gethostname(),'time':time.time(),'manifest_sha256':digest(ROOT/'manifest.json')})
    print('PREFLIGHT',dest,flush=True)

def images(bank,ids):
    array=np.asarray(bank[ids]).copy()
    return torch.from_numpy(array).permute(0,3,1,2).unsqueeze(2).cuda().float()/127.5-1

def noises(records):
    return torch.stack([torch.randn((3,1,512,512),generator=torch.Generator().manual_seed(r['noise_seed']))for r in records]).cuda()

@torch.inference_mode()
def fit(model,rows,cache,cohort):
    bank=np.load(STORE/'banks/fit.npy',mmap_mode='r')
    neg=cache.get([NEGATIVE])[0]
    priority=[.5,.9,.95,.98,.1]
    for t in priority+[t for t in TIMES if t not in priority]:
        k=key(t);out=STORE/'fit'/f'{k}.json'
        if out.exists():
            if t in TEST_TIMES:denoise(model,rows,cache,cohort,selected_times=[t])
            continue
        s=source_time(t);a=1-s
        for start in range(0,FIT_N,FIT_BATCH):
            checkpoint_stop()
            tick=time.monotonic()
            end=min(start+FIT_BATCH,FIT_N)
            dest=STORE/'fit'/k/f'{start:05d}.npz'
            if dest.exists():continue
            records=cohort['fit'][start:end]
            clean=images(bank,list(range(start,end)))
            x=(a*clean+2*s*noises(records)).to(torch.bfloat16)
            cond=text_batch(cache.get([r['caption']for r in records]));negative=text_batch([neg]*len(records))
            pred=raw_predictions(model,x,s,cond,negative)
            errors={}
            for name,p in zip(['conditional','negative'],pred):
                e=project(p.double()-clean.double(),rows)
                errors[name]=e.square().flatten(1).sum(1).cpu().numpy()/(256*2048)
            atomic_npz(dest,ids=np.arange(start,end),**errors)
            save(STORE/'fit/progress.json',{'t':t,'completed_images':end,'images':FIT_N,
                'batch_seconds':time.monotonic()-tick,'time':time.time()})
            if start%256==0:print('FIT',k,start,FIT_N,flush=True)
        arrays=[np.load(STORE/'fit'/k/f'{i:05d}.npz')for i in range(0,FIT_N,FIT_BATCH)]
        result={'complete':True,'t':t,'source_time':float(s),'images':FIT_N}
        for name in ['conditional','negative']:
            values=np.concatenate([a[name]for a in arrays]);v=float(values.mean())
            sigma=float(2*s/(1-s))
            result[name]={'vbar':v,'coefficient':v/(v+sigma*sigma),'effective_sigma':sigma}
        for ar in arrays:ar.close()
        save(out,result)
        print('FITTED',json.dumps(result),flush=True)
        if t in TEST_TIMES:denoise(model,rows,cache,cohort,selected_times=[t])

def coefficients(times=None):
    return {key(t):read(STORE/'fit'/f'{key(t)}.json')for t in (TIMES if times is None else times)}

@torch.inference_mode()
def denoise(model,rows,cache,cohort,selected_times=None):
    bank=np.load(STORE/'banks/test.npy',mmap_mode='r');neg=cache.get([NEGATIVE])[0]
    times=TEST_TIMES if selected_times is None else selected_times
    fits=coefficients(times)
    for t in times:
        k=key(t);out=STORE/'denoising'/f'{k}.json'
        if out.exists():continue
        s=source_time(t);a=1-s
        for start in range(0,TEST_N,FIT_BATCH):
            checkpoint_stop();end=min(start+FIT_BATCH,TEST_N)
            dest=STORE/'denoising'/k/f'{start:05d}.npz'
            if dest.exists():continue
            rec=cohort['test'][start:end];clean=images(bank,list(range(start,end)))
            x=(a*clean+2*s*noises(rec)).to(torch.bfloat16)
            pc,pu=raw_predictions(model,x,s,text_batch(cache.get([r['caption']for r in rec])),text_batch([neg]*len(rec)))
            values={}
            for name,p in [('conditional',pc),('negative',pu)]:
                for method in ['baseline','calibrated','late_half']:
                    pp=p if method=='baseline' else corrected(x,p,s,rows,fits[k][name]['vbar'],strength(t,method))
                    values[f'{name}_{method}']=(pp.double()-clean.double()).square().flatten(1).mean(1).cpu().numpy()
            atomic_npz(dest,ids=np.arange(start,end),**values)
        arrays=[np.load(STORE/'denoising'/k/f'{i:05d}.npz')for i in range(0,TEST_N,FIT_BATCH)]
        vals={name:np.concatenate([a[name]for a in arrays])for name in arrays[0].files if name!='ids'}
        result={'complete':True,'t':t,'images':TEST_N,'raw_image_predictions_before_APG':True,
                'mse':{name:float(v.mean())for name,v in vals.items()},'paired_improvement':{}}
        for branch in ['conditional','negative']:
            base=vals[f'{branch}_baseline']
            for method in ['calibrated','late_half']:
                difference=base-vals[f'{branch}_{method}']
                result['paired_improvement'][f'{branch}_{method}']={
                    'mean':float(difference.mean()),'standard_error':float(difference.std(ddof=1)/math.sqrt(TEST_N)),
                    'relative_percent':float(100*difference.mean()/base.mean())}
        for ar in arrays:ar.close()
        save(out,result)
        print('DENOISING',json.dumps(result),flush=True)
        build_report()
    if all((STORE/'denoising'/f'{key(t)}.json').exists()for t in TEST_TIMES):
        save(ROOT/'DENOISING_COMPLETE.json',{'complete':True,'time':time.time(),'times':TEST_TIMES,
             'fit_images':FIT_N,'test_images':TEST_N,'generation_not_yet_started':True})

def feature_extractor():
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3('inception-v3-compat',['2048']).cuda().eval()

@torch.inference_mode()
def reference_features():
    out=STORE/'fid/reference-statistics.npz'
    if out.exists():return
    net=feature_extractor();bank=np.load(STORE/'banks/reference.npy',mmap_mode='r')
    for start in range(0,len(bank),64):
        checkpoint_stop();end=min(start+64,len(bank));dest=STORE/'fid/reference-features'/f'{start:05d}.npz'
        if dest.exists():continue
        x=torch.from_numpy(np.array(bank[start:end])).permute(0,3,1,2).cuda()
        f=net(x)[0].cpu().numpy()
        atomic_npz(dest,ids=np.arange(start,end),features=f)
    features=np.concatenate([np.load(p)['features']for p in sorted((STORE/'fid/reference-features').glob('*.npz'))])
    assert features.shape==(40504,2048)
    atomic_npz(out,mu=features.astype(np.float64).mean(0),sigma=np.cov(features.astype(np.float64),rowvar=False),count=np.array(40504))
    del net;torch.cuda.empty_cache()

def score(method,count):
    dest=STORE/'fid'/f'{method}-{count}.json'
    if dest.exists():return read(dest)
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    files=sorted((STORE/'generated'/method/'features').glob('*.npz'))
    selected=[];allids=[]
    for p in files:
        with np.load(p) as d:
            mask=d['ids']<count
            if mask.any():selected.append(d['features'][mask]);allids.extend(d['ids'][mask].tolist())
    assert allids==list(range(count)),(len(allids),count)
    features=np.concatenate(selected).astype(np.float64)
    stat={'mu':features.mean(0),'sigma':np.cov(features,rowvar=False)}
    with np.load(STORE/'fid/reference-statistics.npz')as ref:
        metric=fid_statistics_to_metric(stat,{'mu':ref['mu'],'sigma':ref['sigma']},False)
    fid=float(metric['frechet_inception_distance']);assert math.isfinite(fid)and fid>=0
    result={'complete':True,'variant':method,'generated_images':count,'reference_images':40504,'fid':fid,
        'cohorts_sha256':digest(ROOT/'cohorts.json'),'reference_sha256':digest(STORE/'fid/reference-statistics.npz'),
        'model_revision':MODEL_REV,'native_unclamped_sampler':True,'time':time.time(),
        'nested_prefix':True,'generation_batch':SAMPLE_BATCH}
    save(dest,result);build_report();print('FID',json.dumps(result),flush=True)
    return result

def save_png(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    Image.fromarray(x).save(tmp,format='PNG');tmp.replace(path)

@torch.inference_mode()
def generation(model,rows,cache,cohort,method,count):
    out=STORE/'generated'/method
    fits=coefficients();neg=cache.get([NEGATIVE])[0];net=feature_extractor()
    for start in range(0,count,SAMPLE_BATCH):
        checkpoint_stop();end=min(start+SAMPLE_BATCH,count)
        dest=out/'features'/f'{start:05d}.npz'
        if dest.exists():continue
        pngs=[out/'png'/f'{i:05d}.png'for i in range(start,end)]
        if all(p.exists()for p in pngs):
            pixels=torch.from_numpy(np.stack([np.array(Image.open(p).convert('RGB'))for p in pngs])).permute(0,3,1,2).cuda()
        else:
            records=cohort['generation'][start:end]
            features=cache.get([r['caption']for r in records])
            pixels=sample(model,rows,features,neg,[r['noise_seed']for r in records],method,fits).squeeze(2)
            arrays=pixels.permute(0,2,3,1).cpu().numpy()
            for p,x in zip(pngs,arrays):save_png(p,x)
        f=net(pixels)[0].cpu().numpy()
        atomic_npz(dest,ids=np.arange(start,end),features=f)
        save(out/'progress.json',{'completed_images':end,'target':count,'variant':method,'time':time.time()})
        if start%128==0:print('GENERATE',method,end,count,flush=True)
    del net;torch.cuda.empty_cache();score(method,count)

def build_report():
    from .report import render_workspace
    render_workspace(ROOT)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit','generate'],required=True)
    p.add_argument('--variant',choices=['baseline','calibrated','late_half']);p.add_argument('--count',type=int)
    args=p.parse_args()
    for s in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    manifest=verify_source();assert read(ROOT/'PREPARED.json')['complete']
    assert read(ROOT/'PROJECTOR_READY.json')['complete']
    if (ROOT/'execution-sha256.json').exists():
        for name,sha in read(ROOT/'execution-sha256.json').items():
            assert digest(ROOT/name)==sha,name
    prepared=read(ROOT/'PREPARED.json')
    assert digest(ROOT/'cohorts.json')==prepared['data']['cohorts_sha256']
    assert digest(STORE/'assets/model/model.safetensors')==prepared['models']['model']['sha256']['model.safetensors']
    assert digest(STORE/'assets/encoder_rows.npy')==read(STORE/'assets/encoder.json')['rows_sha256']
    lock=(STORE/f'{args.stage}-{args.variant or "fit"}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cohort=read(ROOT/'cohorts.json');cache=TextCache()
    if args.stage=='fit':
        captions=[NEGATIVE]+[r['caption']for split in ['fit','test']for r in cohort[split]]
        for i in range(0,len(captions),TEXT_BATCH):
            checkpoint_stop();cache.get(captions[i:i+TEXT_BATCH])
            if i%128==0:print('TEXT',i,len(captions),flush=True)
        cache.release()
    model,rows=load_model();preflight(model,rows,cache)
    if args.stage=='fit':
        cache.release();fit(model,rows,cache,cohort);denoise(model,rows,cache,cohort)
        del model,rows;cache.release();reference_features()
        save(ROOT/'FIT_COMPLETE.json',{'complete':True,'time':time.time(),'denoising_complete':True,'reference_complete':True})
    else:
        assert args.count in [2000,10000,20000,30000,50000]
        decision=read(ROOT/'GENERATION_DECISION.json')
        assert decision['approved_variants'][args.variant]
        generation(model,rows,cache,cohort,args.variant,args.count)
    return 0

if __name__=='__main__':raise SystemExit(main())
