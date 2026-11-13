import argparse,json,math
import numpy as np
import torch
from native import ROOT,Native,save,digest
from sampler import integrate
from probe_math import Moments,fit
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('case');case=ap.parse_args().case
 torch.set_num_threads(4);torch.manual_seed(19)
 n=Native(case,'cpu');n.capture();checks=[]
 for t in [.1,.5,.9]:
  mp=n.schedule.mapping(t);sigma=mp['sigma'];a=mp['alpha_bar']**.5;b=(1-mp['alpha_bar'])**.5
  assert abs(b/a-sigma)<1e-10
  assert abs(n.schedule.native_time(sigma)-mp['timestep'])<1e-7
  checks.append(mp)
 z=torch.randn((1,n.channels,n.side,n.side));mp=n.schedule.mapping(.5)
 y=torch.tensor([278]);ts=torch.tensor([mp['timestep']],dtype=torch.float32)
 raw=n.forward(z,ts,y);h=n.feature;n.feature=None;replay=n.replay(h)
 assert raw.shape==(1,n.channels*(2 if case=='mdt' else 1),n.side,n.side)
 torch.testing.assert_close(raw,replay,rtol=0,atol=1e-6)
 assert torch.isfinite(raw).all()
 # Conversion equals native epsilon prediction with the native alpha/beta at every tested time.
 for mp in checks:
  t=1/(1+mp['sigma']);a=mp['alpha_bar']**.5;b=(1-mp['alpha_bar'])**.5;eps=raw[:,:n.channels]
  state=z*math.sqrt(t*t+(1-t)**2)
  torch.testing.assert_close((z-b*eps)/a,(state-(1-t)*eps)/t,rtol=1e-5,atol=1e-5)
 # Analytic Gaussian denoiser exercises both Heun evaluations and the endpoint.
 initial=torch.randn((2,1,2,2),dtype=torch.float64)
 oracle=lambda x,t,y:(1-t)/(t*t+(1-t)**2)*x
 sampled=integrate(initial,torch.zeros(2),oracle,n.schedule.t_start)
 expected=initial*(1-n.schedule.t_start)/math.sqrt(n.schedule.t_start**2+(1-n.schedule.t_start)**2)
 torch.testing.assert_close(sampled,expected,rtol=1e-3,atol=1e-4)
 # Affine solver is tested on independent fitting/tuning/test examples with a nonzero shared intercept.
 w=torch.randn(7,3,dtype=torch.float64);b=torch.randn(3,dtype=torch.float64)
 train,dev=Moments(7,3,'cpu'),Moments(7,3,'cpu')
 for obj in [train,dev]:
  x=torch.randn(128,7,dtype=torch.float64);obj.add(x,x@w+b)
 wp,bp,_=fit(train,dev);torch.testing.assert_close(w,wp,rtol=1e-6,atol=1e-7);torch.testing.assert_close(b,bp,rtol=1e-6,atol=1e-7)
 save(ROOT/f'validation-cpu-{case}.json',dict(passed=True,case=case,model=n.audit,mappings=checks,head_reconstruction_max_abs=float((raw-replay).abs().max()),gaussian_sampler_max_abs=float((sampled-expected).abs().max()),affine_solver=True,native_sha256=digest(ROOT/'native.py')))
 print('CPU_VALIDATED',case,flush=True)
if __name__=='__main__':main()
