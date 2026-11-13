import argparse,contextlib,gc,json,math,os,signal,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from native import ROOT,WORK,Native,digest,save,log,verify_sources
from sampler import integrate
STOP=False
def stop(*_):
 global STOP
 STOP=True

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);case=ap.parse_args().case
 for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 verify_sources();
 for record in json.loads((ROOT/'vae-mse-provenance.json').read_text())['files']:
  assert digest(record['path'])==record['sha256']
 out=ROOT/'results'/case
 probe=json.loads((out/'probe-complete.json').read_text());assert probe['complete']
 assert json.loads((ROOT/f'validation-cpu-{case}.json').read_text())['passed']
 if (out/'fid.json').exists():assert json.loads((out/'fid.json').read_text())['complete'];return 0
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.94)
 native=Native(case,'cuda');assert native.checkpoint_sha256==probe['checkpoint_sha256']
 reference=json.loads((ROOT/'references.json').read_text())['256'];assert digest(reference['path'])==reference['sha256']
 use_bf16=False;precision='FP32 network/VAE and ODE (stable across heterogeneous generic GPUs)'
 def autocast():return torch.autocast('cuda',dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
 def epsilon(state,t,labels):
  sigma=(1-t)/t;ts=torch.full((len(state),),native.schedule.native_time(sigma),device='cuda',dtype=torch.float32)
  with autocast():raw=native.forward(state/math.sqrt(t*t+(1-t)**2),ts,labels)
  return raw[:,:native.channels].float()
 # Validate native epsilon-to-clean conversion on the actual model before generation.
 mp=native.schedule.mapping(.5);a=mp['alpha_bar']**.5;b=(1-mp['alpha_bar'])**.5;t=1/(1+mp['sigma'])
 z=torch.randn((1,native.channels,native.side,native.side),device='cuda');labels=torch.zeros(1,device='cuda',dtype=torch.long)
 ts=torch.full((1,),mp['timestep'],device='cuda',dtype=torch.float32)
 with autocast():raw=native.forward(z,ts,labels)
 state=z*math.sqrt(t*t+(1-t)**2);direct=(z-b*raw[:,:native.channels].float())/a;converted=(state-(1-t)*raw[:,:native.channels].float())/t
 torch.testing.assert_close(direct,converted,rtol=2e-4,atol=2e-4)
 save(out/'fid-validation-gpu.json',dict(passed=True,conversion_max_abs=float((direct-converted).abs().max()),precision=precision,gpu=torch.cuda.get_device_name()))
 del z,labels,ts,raw,state,direct,converted
 images=WORK/case/'fid-images';images.mkdir(parents=True,exist_ok=True)
 done={int(p.stem) for p in images.glob('*.png')};assert all(0<=i<10000 for i in done)
 provenance=images.parent/'fid-generation.json'
 if provenance.exists():
  prior=json.loads(provenance.read_text());assert prior['source_manifest_sha256']==digest(ROOT/'source-manifest.json')
  # Keep precision stable across resume on heterogeneous generic GPUs.
  if prior['precision'].startswith('FP32'):use_bf16=False;precision=prior['precision']
  elif not use_bf16:raise RuntimeError('Existing BF16 samples require BF16-capable GPU; do not mix precision.')
 else:save(provenance,dict(source_manifest_sha256=digest(ROOT/'source-manifest.json'),checkpoint_sha256=native.checkpoint_sha256,precision=precision))
 def generate(ids):
  shape=(native.channels,native.side,native.side)
  z=torch.stack([torch.randn(shape,generator=torch.Generator(device='cuda').manual_seed(i),device='cuda') for i in ids])
  y=torch.tensor([i//10 for i in ids],device='cuda',dtype=torch.long)
  samples=integrate(z,y,epsilon,native.schedule.t_start)
  with autocast():rgb=native.decode(samples)
  assert rgb.shape==(len(ids),3,256,256) and torch.isfinite(rgb).all()
  return np.round(np.clip((rgb.float().cpu().numpy().transpose(0,2,3,1)+1)*127.5,0,255)).astype(np.uint8)
 batch=8;start=time.time();log('FID_MODEL_READY',case=case,existing=len(done),precision=precision,gpu=torch.cuda.get_device_name())
 while len(done)<10000:
  if STOP:return 75
  lo=next(i for i in range(10000) if i not in done);ids=[i for i in range(lo,min(lo+batch,10000)) if i not in done];oom=False
  try:pixels=generate(ids)
  except torch.cuda.OutOfMemoryError:
   if batch==1:raise
   oom=True
  if oom:gc.collect();torch.cuda.empty_cache();batch=max(1,batch//2);log('OOM_RETRY',batch=batch);continue
  for i,pix in zip(ids,pixels):
   p=images/f'{i:05d}.png';tmp=p.with_suffix('.tmp.png');Image.fromarray(pix).save(tmp);tmp.replace(p);done.add(i)
  del pixels
  save(out/'fid-progress.json',dict(images=len(done),total=10000,batch=batch,seconds_this_attempt=time.time()-start,job=os.environ.get('SLURM_JOB_ID'),updated=time.time()))
  if len(done)%100<batch:log('FID_GENERATION',case=case,images=len(done),seconds=time.time()-start)
 if STOP:return 75
 metadata=dict(checkpoint_sha256=native.checkpoint_sha256,t_start=native.schedule.t_start,native_audit=native.audit)
 del native,epsilon,generate;gc.collect();torch.cuda.empty_cache()
 assert {p.name for p in images.glob('*.png')}=={f'{i:05d}.png' for i in range(10000)}
 import torch_fidelity
 result=torch_fidelity.calculate_metrics(input1=str(images),input2=None,fid_statistics_file=reference['path'],cuda=True,isc=True,fid=True,kid=False,prc=False,verbose=False,batch_size=8)
 fid=float(result['frechet_inception_distance']);assert math.isfinite(fid)
 save(out/'fid.json',dict(complete=True,case=case,resolution=256,fid=fid,inception_score=float(result['inception_score_mean']),num_images=10000,verified_image_count=10000,classes=1000,images_per_class=10,guidance=1.,sampling_method='adapted-heun',sampling_steps=50,network_evaluations=99,t_end=1.,initialization='(1-t_start)*standard Gaussian',x0_clipping=False,reference_sha256=reference['sha256'],source_manifest_sha256=digest(ROOT/'source-manifest.json'),protocol_sha256=digest(ROOT/'PROTOCOL.md'),precision=precision,batch=batch,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),**metadata))
 log('FID_COMPLETE',case=case,fid=fid);return 0
if __name__=='__main__':raise SystemExit(main())
