"""FP32 JiT clean loss and shared-intercept input recovery on common validation data."""
import argparse,copy,json,sys
from pathlib import Path
from functools import partial
import numpy as np
import torch
import worker
from worker import DATA,DEVICE,HERE,Extractor,digest,run_time,save,log
from probe_math import patchify

class JiTExtractor(Extractor):
    def __init__(self):
        self.case=json.loads((HERE/'case.json').read_text())
        self.name='jit_'+self.case['id'];self.feature=None;self.replay_error=0.
        c=self.case
        for name,value in c['source_hashes'].items():assert digest(Path(c['source'])/name)==value,name
        checkpoint=Path(c['checkpoint']);self.checkpoint_sha256=digest(checkpoint)
        assert self.checkpoint_sha256==c['checkpoint_sha256']
        worker.ASSETS[self.name]={'sha256':self.checkpoint_sha256}
        sys.path.insert(0,c['source'])
        import model_jit
        from denoiser import Denoiser
        state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert state['epoch']+1==c['completed_epochs']
        args=copy.deepcopy(state['args'])
        assert args.patch_embed=='linear' and args.pred_space==c['pred']
        if c['kind']=='width':
            model_jit.JiT_models[args.model]=partial(model_jit.JiT,depth=c['depth'],hidden_size=c['width'],num_heads=c['heads'],
                patch_size=c['patch'],bottleneck_dim=128,in_context_len=32,in_context_start=4)
        wrapper=Denoiser(args);wrapper.load_state_dict(state['model_ema1'],strict=True);del state
        self.model=wrapper.net.float().to(DEVICE).eval().requires_grad_(False)
        self.head=self.model.final_layer.linear;self.keys={'token':(c['width'],3*c['patch']**2)}
        self.labels=np.load(DATA/'labels.npy');self.ids=np.load(DATA/'original_indices.npy')
        self.clean=np.load(DATA/f'images{c["image_size"]}.npy',mmap_mode='r')
        self.handle=self.head.register_forward_pre_hook(lambda mod,args:setattr(self,'feature',args[0].detach()))
        assert len(self.model.blocks)==c['depth']
        self.parameters=sum(p.numel() for p in self.model.parameters())-self.model.pos_embed.numel()
        if c.get('expected_parameters'):assert self.parameters==c['expected_parameters']

    @torch.inference_mode()
    def extract(self,lo,hi,mp,replay=False):
        c=self.case;t=mp['jit_reference_t'];res=c['image_size']
        x=torch.from_numpy(np.array(self.clean[lo:hi])).to(DEVICE).float().div_(127.5).sub_(1)
        eps=torch.stack([torch.randn((3,res,res),generator=torch.Generator().manual_seed(8100000+int(i))) for i in self.ids[lo:hi]]).to(DEVICE)
        z=t*x+(1-t)*eps
        raw=self.model(z,torch.full((len(x),),t,device=DEVICE),torch.from_numpy(self.labels[lo:hi]).to(DEVICE))
        h=self.feature
        if replay:
            error=float((self.model.unpatchify(self.head(h),c['patch'])-raw).abs().max());assert error<1e-5
            self.replay_error=max(self.replay_error,error)
        estimate=(z-(1-t)*raw)/t if c['pred']=='eps' else raw
        mse=(estimate.double()-x.double()).square().mean((1,2,3))
        assert torch.isfinite(mse).all();self.feature=None
        return {'token':h},patchify(z,c['patch']),mse

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['input','mse'],default='input')
    p.add_argument('--times',type=float,nargs='+',default=[.1,.2,.3,.4,.5,.6,.7,.8,.9]);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    x=JiTExtractor();batch=x.tune()
    if args.mode=='input':assert x.case['pred']=='eps','Image predictors only need --mode mse'
    for t in args.times:
        assert 0<t<1
        if args.mode=='input':batch=run_time(x,t,batch)
        else:
            errors=[]
            for lo in range(512,2512,batch):errors.extend(x.extract(lo,min(lo+batch,2512),{'jit_reference_t':t},replay=(lo==512))[2].cpu().tolist())
            save(HERE/f'results/{x.name}-t{t:g}.json',dict(complete=True,model=x.name,mapping=dict(native_time=t,sigma=(1-t)/t),
                 checkpoint_sha256=x.checkpoint_sha256,completed_epochs=x.case['completed_epochs'],ema='EMA1',
                 native_clean_mse=float(np.mean(errors)),native_clean_mse_by_image=errors,test_images=2000,precision='FP32 model; FP64 metrics; TF32 off'))
    log('COMPLETE',case=x.case['id'],parameters=x.parameters)
if __name__=='__main__':main()
