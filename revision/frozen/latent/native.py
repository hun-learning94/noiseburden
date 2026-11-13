"""Pinned native epsilon predictors; inference-only wrappers, no model-source edits."""
import hashlib,json,math,sys,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
WORK=Path(os.environ['REVISION_WORK'])

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def log(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
def verify_sources():
 for f,h in json.loads((ROOT/'source-manifest.json').read_text())['sha256'].items():assert digest(ROOT/f)==h,f

def load_weights(path):
 # Official research checkpoints may include optimizer metadata; do not load arbitrary user inputs.
 return torch.load(path,map_location='cpu',weights_only=False)

class Schedule:
 def __init__(self,case,alphas=None):
  self.case=case
  if case=='uvit':
   self.sigma_max=math.sqrt(math.expm1(.1+.5*19.9));self.name='continuous VP beta(s)=0.1+19.9*s; network timestep=999*s'
  else:
   self.alphas=np.asarray(alphas,dtype=np.float64);self.sigmas=np.sqrt(1/self.alphas-1)
   self.sigma_max=float(self.sigmas[-1]);self.name='native discrete VP; nearest log sigma for probes, interpolated log sigma for sampling'
  self.t_start=1/(1+self.sigma_max)
 def native_time(self,sigma):
  if self.case=='uvit':
   # Solve .1*s + 9.95*s^2 = log(1+sigma^2), avoiding cancellation.
   c=math.log1p(sigma*sigma);s=2*c/(.1+math.sqrt(.01+39.8*c));return 999*s
  return float(np.interp(math.log(sigma),np.log(self.sigmas),np.arange(len(self.sigmas))))
 def mapping(self,t):
  sigma=(1-t)/t;nt=self.native_time(sigma)
  if self.case=='uvit':ab=1/(1+sigma*sigma)
  else:
   nt=int(np.argmin(abs(np.log(self.sigmas)-math.log(sigma))));ab=float(self.alphas[nt]);sigma=float(self.sigmas[nt])
  return dict(jit_reference_t=t,requested_sigma=(1-t)/t,sigma=sigma,timestep=nt,alpha_bar=ab,schedule=self.name)

class VQInference(nn.Module):
 """VQModelInterface encode/decode, using the original component implementations."""
 def __init__(self,p):
  super().__init__()
  from ldm.modules.diffusionmodules.model import Encoder,Decoder
  from taming.modules.vqvae.quantize import VectorQuantizer2
  d=p['ddconfig'];e=p['embed_dim']
  self.encoder=Encoder(**d);self.decoder=Decoder(**d)
  self.quantize=VectorQuantizer2(p['n_embed'],e,beta=.25,remap=p.get('remap'),sane_index_shape=p.get('sane_index_shape',False))
  self.quant_conv=nn.Conv2d(d['z_channels'],e,1);self.post_quant_conv=nn.Conv2d(e,d['z_channels'],1)
 def encode(self,x):return self.quant_conv(self.encoder(x))
 def decode(self,z):return self.decoder(self.post_quant_conv(self.quantize(z)[0]))

class Native:
 def __init__(self,case,device='cpu'):
  self.case=case;self.assets=json.loads((ROOT/f'assets-{case}.json').read_text())['assets']
  ck=self.assets['checkpoint'];assert digest(ck['path'])==ck['sha256']
  self.checkpoint_sha256=ck['sha256'];self.feature=None;self.handle=None
  sys.path.insert(0,str(ROOT/'code'/case))
  state=load_weights(ck['path'])
  self.audit={}
  if case=='uvit':
   from libs.uvit import UViT
   from libs.autoencoder import get_model
   self.model=UViT(img_size=32,patch_size=2,in_chans=4,embed_dim=1152,depth=28,num_heads=16,mlp_ratio=4,qkv_bias=False,mlp_time_embed=False,num_classes=1001,use_checkpoint=False,conv=False)
   self.model.load_state_dict(state,strict=True)
   ae=self.assets['autoencoder'];assert digest(ae['path'])==ae['sha256'];self.vae=get_model(ae['path'])
   self.head=self.model.decoder_pred;self.schedule=Schedule(case);self.channels,self.side=4,32;self.feature_dim=1152;self.scale=.18215
   self.audit['vae']='official autoencoder_kl_ema.pth; posterior sample; scale0.18215'
  elif case=='ldm':
   import yaml
   sys.path.insert(0,str(ROOT/'code/taming'))
   from ldm.modules.diffusionmodules.openaimodel import UNetModel
   from ldm.modules.diffusionmodules.util import make_beta_schedule
   configs=[(p,yaml.safe_load(Path(p).read_text())) for p in self.assets['configs']]
   configs=[(p,c) for p,c in configs if 'model' in c];assert len(configs)==1,configs
   config_path,config=configs[0];p=config['model']['params'];self.config=p
   assert p.get('parameterization','eps')=='eps'
   assert p.get('conditioning_key','crossattn')=='crossattn'
   self.model=UNetModel(**p['unet_config']['params'])
   sd=state['state_dict'];raw={k[len('model.diffusion_model.'):]:v for k,v in sd.items() if k.startswith('model.diffusion_model.')}
   self.model.load_state_dict(raw,strict=True)
   ema=p.get('use_ema',True)
   if ema:
    for name,param in self.model.named_parameters():
     key='model_ema.'+('diffusion_model.'+name).replace('.','');assert key in sd,key;param.data.copy_(sd[key])
   cp=p['cond_stage_config']['params'];self.condition=nn.Embedding(cp.get('n_classes',1000),cp['embed_dim'])
   self.condition.load_state_dict({k[len('cond_stage_model.embedding.'):]:v for k,v in sd.items() if k.startswith('cond_stage_model.embedding.')},strict=True)
   fp=p['first_stage_config']['params'];assert p['first_stage_config']['target'].endswith('VQModelInterface')
   self.vae=VQInference(fp)
   vs={k[len('first_stage_model.'):]:v for k,v in sd.items() if k.startswith('first_stage_model.')}
   self.vae.load_state_dict(vs,strict=True)
   self.scale=float(sd.get('scale_factor',p.get('scale_factor',1.)))
   ab=sd['alphas_cumprod'].double().numpy();betas=make_beta_schedule(p.get('beta_schedule','linear'),p.get('timesteps',1000),p.get('linear_start',1e-4),p.get('linear_end',.02))
   np.testing.assert_allclose(ab,np.cumprod(1-betas),rtol=1e-6,atol=1e-8)
   self.schedule=Schedule(case,ab);self.head=self.model.out[-1]
   self.channels=p['channels'];self.side=256//2**(len(fp['ddconfig']['ch_mult'])-1);self.feature_dim=self.head.in_channels*9
   self.audit.update(config_path=config_path,config_sha256=digest(config_path),ema_applied=ema,vae='native continuous pre-quantization VQ embedding; native quantization on decode',scale=self.scale)
   del sd,raw,vs
  else:raise ValueError(case)
  del state
  self.model.float().to(device).eval().requires_grad_(False);self.vae.float().to(device).eval().requires_grad_(False)
  if case=='ldm':self.condition.float().to(device).eval().requires_grad_(False)
  self.device=device
  self.audit.update(case=case,parameters=sum(p.numel() for p in self.model.parameters()),checkpoint_sha256=self.checkpoint_sha256,latent_shape=[self.channels,self.side,self.side],feature_dimension=self.feature_dim,schedule=self.schedule.name,t_start=self.schedule.t_start,strict_load=True)
 def capture(self):
  self.handle=self.head.register_forward_pre_hook(lambda m,args:setattr(self,'feature',args[0].detach()))
 def forward(self,x,ts,y):
  if self.case=='uvit':return self.model(x,ts,y=y)
  if self.case=='mdt':return self.model(x,ts,y,enable_mask=False)
  return self.model(x,ts,context=self.condition(y[:,None]))
 def replay(self,h):
  z=self.head(h)
  if self.case=='uvit':
   from libs.uvit import unpatchify
   return unpatchify(z[:,self.model.extras:,:],self.channels)
  if self.case=='mdt':return self.model.unpatchify(z)
  return z
 def encode(self,x,noise):
  if self.case=='uvit':
   mean,logvar=self.vae.encode_moments(x).chunk(2,1);return (mean+torch.exp(.5*logvar.clamp(-30,20))*noise)*self.scale
  if self.case=='mdt':
   q=self.vae.encode(x).latent_dist;return (q.mean+q.std*noise)*self.scale
  return self.vae.encode(x)*self.scale
 def decode(self,z):
  if self.case=='uvit':return self.vae.decode(z)
  if self.case=='mdt':return self.vae.decode(z/self.scale).sample
  return self.vae.decode(z/self.scale)
