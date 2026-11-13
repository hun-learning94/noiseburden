import argparse,gc,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from native import ROOT,WORK,Native,digest,save,log,verify_sources
from probe_math import Moments,fit,metrics,patchify,neighborhoods,sample_pixels
STOP=False
def stop(*_):
 global STOP
 STOP=True
def check_stop():
 if STOP:raise SystemExit(75)
DATA=Path(os.environ['PUBLIC_PROBE_CACHE'])

def retry(fn,batch):
 while True:
  oom=False
  try:return fn(batch),batch
  except torch.cuda.OutOfMemoryError:
   if batch==1:raise
   oom=True
  if oom:
   gc.collect();torch.cuda.empty_cache();batch=max(1,batch//2);log('OOM_RETRY',batch=batch)

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);args=ap.parse_args()
 for s in [signal.SIGUSR1,signal.SIGTERM]:signal.signal(s,stop)
 verify_sources();
 for record in json.loads((ROOT/'vae-mse-provenance.json').read_text())['files']:
  assert digest(record['path'])==record['sha256']
 validation=json.loads((ROOT/f'validation-cpu-{args.case}.json').read_text());assert validation['passed'] and validation['native_sha256']==digest(ROOT/'native.py')
 torch.set_num_threads(4);torch.manual_seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.94)
 result=ROOT/'results'/args.case;result.mkdir(parents=True,exist_ok=True)
 native=Native(args.case,'cuda');save(result/'model-audit.json',native.audit)
 for filename,record in json.loads((ROOT/'data-provenance.json').read_text()).items():
  if filename!='samples.json':assert digest(DATA/filename)==record['sha256'],filename
 ids=np.load(DATA/'original_indices.npy');labels=np.load(DATA/'labels.npy');assert len(ids)==len(labels)==2512
 images=np.load(DATA/'images256.npy',mmap_mode='r')
 latent_path=WORK/args.case/'clean-latents.npy';latent_path.parent.mkdir(parents=True,exist_ok=True)
 latent_meta=latent_path.with_suffix('.json')
 if latent_meta.exists():
  lm=json.loads(latent_meta.read_text());assert lm['checkpoint_sha256']==native.checkpoint_sha256 and lm['sha256']==digest(latent_path)
 else:
  out=np.lib.format.open_memmap(str(latent_path)+'.tmp',mode='w+',dtype=np.float32,shape=(2512,native.channels,native.side,native.side))
  lo,batch=0,32
  while lo<2512:
   check_stop()
   def encode(b):
    hi=min(lo+b,2512);x=torch.from_numpy(np.array(images[lo:hi])).cuda().float()/127.5-1
    noise=torch.stack([torch.randn((native.channels,native.side,native.side),generator=torch.Generator().manual_seed(7200000+int(i))) for i in ids[lo:hi]]).cuda()
    z=native.encode(x,noise);assert torch.isfinite(z).all();return z.cpu().numpy()
   z,batch=retry(encode,batch);out[lo:lo+len(z)]=z;lo+=len(z)
   if lo%256<batch:log('ENCODE_PROGRESS',case=args.case,images=lo,batch=batch)
  out.flush();del out;Path(str(latent_path)+'.tmp').replace(latent_path)
  save(latent_meta,dict(sha256=digest(latent_path),checkpoint_sha256=native.checkpoint_sha256,posterior_seed='7200000+original image index',images_sha256=digest(DATA/'images256.npy')))
 native.vae.cpu();gc.collect();torch.cuda.empty_cache();clean=np.load(latent_path,mmap_mode='r');native.capture()
 positions=torch.from_numpy(np.stack([np.random.default_rng(9300000+int(i)).choice(native.side**2,size=min(512,native.side**2),replace=False) for i in ids])).cuda()
 def extract(lo,hi,mp):
  x=torch.from_numpy(np.array(clean[lo:hi])).cuda();eps=torch.stack([torch.randn(tuple(x.shape[1:]),generator=torch.Generator().manual_seed(8100000+int(i))) for i in ids[lo:hi]]).cuda()
  a=mp['alpha_bar']**.5;b=(1-mp['alpha_bar'])**.5;z=a*x+b*eps
  ts=torch.full((len(x),),mp['timestep'],device='cuda',dtype=torch.float32);y=torch.from_numpy(labels[lo:hi]).cuda()
  raw=native.forward(z,ts,y);h=native.feature;native.feature=None
  replay=native.replay(h);err=float((raw-replay).abs().max());assert err<1e-5,err
  mse=(((z-b*raw[:,:native.channels])/a).double()-x.double()).square().mean((1,2,3))
  if args.case=='ldm':features=neighborhoods(h,positions[lo:hi]);target=sample_pixels(z,positions[lo:hi])
  else:features=h[:,native.model.extras:,:] if args.case=='uvit' else h;target=patchify(z)
  assert features.shape[:2]==target.shape[:2] and torch.isfinite(features).all() and torch.isfinite(raw).all()
  return features,target,mse,err
 # Choose the largest tested batch with headroom; all probe math remains FP64.
 batch=16
 for candidate in [16,32,64,128,256]:
  torch.cuda.reset_peak_memory_stats()
  oom=False
  try:
   sample=extract(0,candidate,native.schedule.mapping(.5));del sample;torch.cuda.synchronize()
  except torch.cuda.OutOfMemoryError:oom=True;native.feature=None
  gc.collect();torch.cuda.empty_cache()
  if oom:break
  if torch.cuda.max_memory_allocated()>.80*torch.cuda.get_device_properties(0).total_memory:break
  batch=candidate
 save(result/'batch.json',dict(batch=batch,gpu=torch.cuda.get_device_name(),precision='FP32 model; FP64 probes'))
 for t in np.arange(1,10)/10:
  check_stop();dest=result/f't{t:g}.json'
  if dest.exists():
   old=json.loads(dest.read_text());assert old['complete'] and old['source_manifest_sha256']==digest(ROOT/'source-manifest.json') and old['checkpoint_sha256']==native.checkpoint_sha256;continue
  mp=native.schedule.mapping(float(t));q=native.channels if args.case=='ldm' else native.channels*4
  train=Moments(native.feature_dim,q,'cuda');dev=Moments(native.feature_dim,q,'cuda')
  errors,means,seconds,losses=[],[],[],[];replay_error=0.;start=time.time()
  for phase,first,end in [('fit',0,384),('tune',384,512),('test',512,2512)]:
   lo=first
   while lo<end:
    check_stop()
    def step(b):return extract(lo,min(lo+b,end),mp)
    (h,y,loss,err),batch=retry(step,batch);replay_error=max(replay_error,err);count=len(h)
    if phase!='test':(train if phase=='fit' else dev).add(h,y)
    else:
     yd=y.double();errors.extend(((h.double()@w+bias)-yd).square().mean((1,2)).cpu().tolist());means.extend(yd.mean(1).cpu().tolist());seconds.extend(yd.square().mean(1).cpu().tolist());losses.extend(loss.cpu().tolist());del yd
    del h,y,loss;lo+=count
    save(result/'progress.json',dict(case=args.case,t=float(t),phase=phase,images=lo-first,total=end-first,batch=batch,updated=time.time(),job=os.environ.get('SLURM_JOB_ID')))
   if phase=='tune':
    w,bias,info=fit(train,dev);del train,dev
    tmp=dest.with_suffix('.pt.tmp');torch.save(dict(weight=w.cpu(),bias=bias.cpu()),tmp);tmp.replace(dest.with_suffix('.pt'))
  score=metrics(errors,means,seconds)
  save(dest,dict(complete=True,model=args.case,mapping=mp,probe=score,fit=info,fit_images=384,tune_images=128,test_images=2000,observations_per_image=len(positions[0]) if args.case=='ldm' else 256,head_reconstruction_max_abs=replay_error,native_latent_clean_mse=float(np.mean(losses)),input_space='native latent',recovery_target='actual noisy model input',checkpoint_sha256=native.checkpoint_sha256,protocol_sha256=digest(ROOT/'PROTOCOL.md'),source_manifest_sha256=digest(ROOT/'source-manifest.json'),batch=batch,seconds=time.time()-start,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),precision='FP32 model and inputs; FP64 fit and metrics; TF32 off'))
  log('TIME_COMPLETE',case=args.case,t=float(t),r2=score['r2_percent'],seconds=time.time()-start)
 save(result/'probe-complete.json',dict(complete=True,case=args.case,times=[float(i/10) for i in range(1,10)],checkpoint_sha256=native.checkpoint_sha256,source_manifest_sha256=digest(ROOT/'source-manifest.json')))
if __name__=='__main__':main()
