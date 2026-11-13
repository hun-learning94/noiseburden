"""One public model's matched FID-10K with adapted Heun50 on a generic GPU."""
import argparse,contextlib,gc,hashlib,json,math,os,signal,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sampler import integrate,native_t,T_START,ALPHAS,SIGMAS
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
STOP=False

def stop(*_):
 global STOP
 STOP=True

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def log(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['adm128','adm256','dit256'],required=True);a=ap.parse_args()
 for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 for f,h in json.loads((HERE/'source-manifest.json').read_text())['sha256'].items():assert digest(HERE/f)==h,f
 assert json.loads((HERE/'validation-cpu.json').read_text())['passed']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
 torch.cuda.set_per_process_memory_fraction(.92)
 assets=json.loads((HERE/'assets.json').read_text());size=128 if a.case=='adm128' else 256
 out=HERE/'results'/a.case;out.mkdir(parents=True,exist_ok=True)
 protocol_sha=digest(HERE/'PROTOCOL.md');source_sha=digest(HERE/'source-manifest.json')
 if (out/'fid.json').exists():
  d=json.loads((out/'fid.json').read_text());assert d['complete'] and d['protocol_sha256']==protocol_sha and d['source_manifest_sha256']==source_sha;return 0
 images=Path(os.environ['REVISION_WORK'])/a.case/'images';images.mkdir(parents=True,exist_ok=True)
 info=assets['assets']['dit' if a.case=='dit256' else a.case];assert digest(info['path'])==info['sha256']
 reference=json.loads((HERE/'references.json').read_text())[str(size)];assert digest(reference['path'])==reference['sha256']
 if a.case=='dit256':
  from dit_models import DiT_XL_2
  model=DiT_XL_2(input_size=32);channels,side=4,32
 else:
  from guided_diffusion.script_util import create_model
  model=create_model(image_size=size,num_channels=256,num_res_blocks=2,learn_sigma=True,class_cond=True,
    attention_resolutions='32,16,8',num_heads=4,num_head_channels=-1 if size==128 else 64,
    resblock_updown=True,use_fp16=False,use_scale_shift_norm=True)
  channels,side=3,size
 model.load_state_dict(torch.load(info['path'],map_location='cpu',weights_only=True,mmap=True),strict=True)
 model=model.float().cuda().eval().requires_grad_(False)
 vae=None
 if a.case=='dit256':
  from diffusers import AutoencoderKL
  vf=next(f for f in assets['vae_mse']['files'] if f['path'].endswith('.safetensors'))
  assert digest(vf['path'])==vf['sha256']
  vae=AutoencoderKL.from_pretrained(str(Path(vf['path']).parent),local_files_only=True,use_safetensors=True).float().cuda().eval().requires_grad_(False)
 use_bf16=(torch.cuda.get_device_capability()[0]>=8 and torch.cuda.is_bf16_supported());precision='BF16 backbone/VAE with FP32 ODE states' if use_bf16 else 'FP32 backbone/VAE and ODE states (BF16 unsupported)'
 def autocast():return torch.autocast('cuda',dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
 def epsilon(state,t,labels):
  scale=math.sqrt(t*t+(1-t)*(1-t));native=state/scale
  steps=torch.full((len(state),),native_t(t),device='cuda',dtype=torch.float32)
  with autocast():raw=model(native,steps,labels)
  assert raw.shape==(len(state),channels*2,side,side)
  return raw[:,:channels].float()
 # Check the adapter's input normalization and conversion against official DDPM parsing at a native integer step.
 from diffusion import create_diffusion
 diffusion=create_diffusion('');k=500;t=float(1/(1+SIGMAS[k]));alpha=float(ALPHAS[k])**.5;beta=float(1-ALPHAS[k])**.5
 u=torch.randn((1,channels,side,side),device='cuda');y=torch.zeros(1,device='cuda',dtype=torch.long)
 native=u/math.sqrt(t*t+(1-t)*(1-t));ts=torch.tensor([k],device='cuda')
 with autocast():raw=model(native,ts,y)
 official=diffusion.p_mean_variance(lambda *args,**kw:raw.float(),native,ts,clip_denoised=False)['pred_xstart']
 estimate=(u-(1-t)*raw[:,:channels].float())/t
 torch.testing.assert_close(estimate,official,rtol=2e-4,atol=2e-4)
 assert abs(native_t(t)-k)<1e-8
 save(out/'validation-gpu.json',dict(passed=True,conversion_max_abs=float((estimate-official).abs().max()),native_step=k,precision=precision,gpu=torch.cuda.get_device_name()))
 del u,y,native,ts,raw,official,estimate
 def generate(ids):
  initial=torch.stack([torch.randn((channels,side,side),generator=torch.Generator(device='cuda').manual_seed(i),device='cuda') for i in ids])
  labels=torch.tensor([i//10 for i in ids],device='cuda',dtype=torch.long)
  sampled=integrate(initial,labels,epsilon)
  if vae is not None:
   with autocast():sampled=vae.decode(sampled/.18215).sample
  assert sampled.shape==(len(ids),3,size,size) and torch.isfinite(sampled).all()
  return np.round(np.clip((sampled.float().cpu().numpy().transpose(0,2,3,1)+1)*127.5,0,255)).astype(np.uint8)
 batch=8;done={int(p.stem) for p in images.glob('*.png')};assert all(0<=i<10000 for i in done)
 start=time.time();log('MODEL_READY',case=a.case,precision=precision,existing=len(done),gpu=torch.cuda.get_device_name())
 while len(done)<10000:
  if STOP:return 75
  lo=next(i for i in range(10000) if i not in done);hi=min(lo+batch,10000)
  ids=[i for i in range(lo,hi) if i not in done]
  oom=False
  try:pixels=generate(ids)
  except torch.cuda.OutOfMemoryError:
   if batch==1:raise
   oom=True
  if oom:
   gc.collect();torch.cuda.empty_cache();batch=max(1,batch//2);log('OOM_RETRY',batch=batch);continue
  for i,arr in zip(ids,pixels):
   path=images/f'{i:05d}.png';tmp=path.with_suffix('.tmp.png');Image.fromarray(arr).save(tmp);tmp.replace(path);done.add(i)
  del pixels
  save(out/'progress.json',dict(case=a.case,images=len(done),total=10000,batch=batch,job=os.environ.get('SLURM_JOB_ID','local'),updated=time.time(),seconds_this_attempt=time.time()-start))
  if len(done)%100<batch:log('GENERATION_PROGRESS',case=a.case,images=len(done),batch=batch,seconds=time.time()-start)
 if STOP:return 75
 # Free the large denoiser/VAE before Inception extraction on 8GB generic GPUs.
 del model,vae,epsilon,generate;gc.collect();torch.cuda.empty_cache()
 assert {p.name for p in images.glob('*.png')}=={f'{i:05d}.png' for i in range(10000)}
 import torch_fidelity
 log('FID_START',case=a.case)
 metrics=torch_fidelity.calculate_metrics(input1=str(images),input2=None,fid_statistics_file=reference['path'],cuda=True,
   isc=True,fid=True,kid=False,prc=False,verbose=False,batch_size=8)
 fid=float(metrics['frechet_inception_distance']);assert math.isfinite(fid)
 save(out/'fid.json',dict(complete=True,case=a.case,resolution=size,fid=fid,inception_score=float(metrics['inception_score_mean']),
   num_images=10000,verified_image_count=10000,classes=1000,images_per_class=10,guidance=1.,sampling_method='adapted-heun',
   sampling_steps=50,network_evaluations=99,t_start=T_START,t_end=1.,native_timestep_mapping='linear interpolation in log sigma',
   ode='linear corruption X_t=t*X_1+(1-t)*epsilon',initialization='(1-t_start)*N(0,I); finite native maximum sigma',
   x0_clipping=False,png_quantization='round(clip((x+1)*127.5,0,255))',seed='per-image CUDA generator seed=image index, starting at0',
   precision=precision,batch=batch,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID','local'),checkpoint_sha256=info['sha256'],
   reference_sha256=reference['sha256'],protocol_sha256=protocol_sha,source_manifest_sha256=source_sha,
   vae='ft-MSE, divide by0.18215' if a.case=='dit256' else None,seconds_this_attempt=time.time()-start))
 log('FID_COMPLETE',case=a.case,fid=fid)
 return 0
if __name__=='__main__':raise SystemExit(main())
