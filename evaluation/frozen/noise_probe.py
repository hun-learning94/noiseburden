"""Noise-only affine probes on backbone-held-out validation images; historical settings.
FP64 sufficient statistics and solves run on the billed GPU. All six times
are evaluated together with common corruptions; old t=.5 scores are audited.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;M=json.loads((HERE/'noise-manifest.json').read_text());OUT=HERE/'results/noise';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
CACHE=Path(os.environ['PROBE_CACHE']);START=time.time()
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(p)
def patchify(x):
 b,c,h,w=x.shape;p=16
 return x.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,3,5,1).reshape(b,-1,p*p*c)
def main():
 assert torch.cuda.is_available();assert digest(__file__)==M['worker_sha256']
 assert digest(M['checkpoint'])==M['checkpoint_sha256']
 for name,h in M['source_hashes'].items():assert digest(Path(M['source'])/name)==h,name
 for name in ['images256.npy','labels.npy','samples.json']:assert digest(CACHE/name)==M['cache_hashes'][name],name
 sys.path.insert(0,M['source']);from denoiser import Denoiser
 ck=torch.load(M['checkpoint'],map_location='cpu',mmap=True,weights_only=False);assert ck['epoch']==199
 assert ck['args'].pred_type=='eps' and ck['args'].shortcut=='none'
 model=Denoiser(ck['args']);model.load_state_dict(ck['model_ema1'],strict=True);net=model.net.float().cuda().eval().requires_grad_(False);del ck
 images=np.load(CACHE/'images256.npy',mmap_mode='r');labels=np.load(CACHE/'labels.npy',mmap_mode='r');assert images.shape==(10000,3,256,256)
 store={};hooks=[];nctx=net.in_context_len;start=net.in_context_start
 hooks.append(net.blocks[0].register_forward_pre_hook(lambda mod,args:store.__setitem__(0,args[0].detach())))
 def hook(i):
  def keep(mod,args,out):store[i+1]=(out[:,nctx:] if nctx>0 and i>=start else out).detach()
  return keep
 for i,block in enumerate(net.blocks):hooks.append(block.register_forward_hook(hook(i)))
 def run(times,tag):
  accum={};weights={};errors={};per_image={}
  for tv in times:
   for li in range(13):
    accum[tv,li]=[torch.zeros((769,769),dtype=torch.float64,device='cuda'),torch.zeros((769,768),dtype=torch.float64,device='cuda')]
    errors[tv,li]=torch.zeros((),dtype=torch.float64,device='cuda');per_image[tv,li]=torch.empty(2000,dtype=torch.float64,device='cuda')
  target_sum=torch.zeros(768,dtype=torch.float64,device='cuda');target_sq=torch.zeros((),dtype=torch.float64,device='cuda');eval_count=0
  for phase,lo,hi,seed in [('fit',0,8000,117),('eval',8000,10000,991)]:
   generator=torch.Generator(device='cuda').manual_seed(seed)
   phase_start=time.time()
   with torch.no_grad():
    for bi,a in enumerate(range(lo,hi,8)):
     b=min(a+8,hi)
     # Match the original CPU pixel normalization and CUDA draw shape exactly.
     x=torch.from_numpy(np.array(images[a:b])).float().div_(255).mul_(2).sub_(1).cuda()
     y=torch.from_numpy(np.array(labels[a:b])).cuda()
     eps=torch.randn(x.shape,generator=generator,device='cuda',dtype=torch.float32)
     target=patchify(eps).reshape(-1,768).double()
     if phase=='eval':target_sum+=target.sum(0);target_sq+=(target**2).sum();eval_count+=target.shape[0]
     for tv in times:
      t=torch.full((len(x),),tv,device='cuda',dtype=torch.float32);net(tv*x+(1-tv)*eps,t,y)
      for li in range(13):
       feat=store[li].reshape(-1,768).double();feat=torch.cat((feat,torch.ones((len(feat),1),device='cuda',dtype=torch.float64)),1)
       if phase=='fit':
        xx,xy=accum[tv,li];xx.addmm_(feat.T,feat);xy.addmm_(feat.T,target)
       else:
        sq=(feat@weights[tv,li]-target).square();errors[tv,li]+=sq.sum();per_image[tv,li][a-8000:b-8000]=sq.reshape(len(x),256,768).mean((1,2))
     if bi%50==0 or b==hi:
      torch.cuda.synchronize();progress=dict(group=tag,times=times,phase=phase,images_done=b-lo,images_total=hi-lo,seconds=time.time()-START,phase_seconds=time.time()-phase_start,job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
      save(HERE/'noise-progress.json',progress);print(json.dumps(progress),flush=True)
   if phase=='fit':
    for tv in times:
     for li in range(13):
      xx,xy=accum.pop((tv,li));reg=torch.eye(769,dtype=torch.float64,device='cuda')*(1e-3*xx.diag()[:-1].mean());reg[-1,-1]=1e-8;weights[tv,li]=torch.linalg.solve(xx+reg,xy)
  variance_total=target_sq-(target_sum**2).sum()/eval_count
  for tv in times:
   records=[]
   for li in range(13):
    mse=float(errors[tv,li].item()/(eval_count*768));score=float((1-errors[tv,li]/variance_total).item())
    records.append(dict(layer=li,noise_recovery_score=score,noise_mse=mse,unit_variance_score=1-mse))
   result=dict(t=tv,layers=list(range(13)),records=records,checkpoint_sha256=M['checkpoint_sha256'],source_hashes=M['source_hashes'],worker_sha256=M['worker_sha256'],manifest_sha256=digest(HERE/'noise-manifest.json'),ema='EMA1',completed_epochs=200,fit_images=8000,test_images=2000,batch_size=8,image_seed=20261025,noise_seed_fit=117,noise_seed_eval=991,noise_variance=float(variance_total.item()/(eval_count*768)),precision='FP32 backbone; FP64 GPU affine statistics and solve; TF32 off',gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-START)
   save(OUT/f'noise-t{tv}.json',result)
   np.savez(OUT/f'noise-t{tv}-per-image.npz',**{f'layer{li}':per_image[tv,li].cpu().numpy() for li in range(13)})
  del accum,weights,errors,per_image;torch.cuda.empty_cache()
 run(M['all_plot_times'],'six-time-cohort')
 for h in hooks:h.remove()
 save(HERE/'noise-complete.json',dict(complete=True,times=M['all_plot_times'],layers=13,probe_scores=78,job=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.time()-START))
if __name__=='__main__':main()
