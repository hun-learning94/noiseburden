"""Check VP-to-linear-path geometry, time mapping and the Heun oracle solution."""
import json,sys,math
from pathlib import Path
import numpy as np
import torch
from sampler import ALPHAS,SIGMAS,T_START,native_t,integrate
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))

def main():
 torch.set_num_threads(2)
 from diffusion import create_diffusion
 d=create_diffusion('');np.testing.assert_array_equal(d.alphas_cumprod,ALPHAS)
 for k in [0,10,100,500,900,999]:
  t=float(1/(1+SIGMAS[k]));assert abs(native_t(t)-k)<1e-7
  np.testing.assert_allclose([t/math.sqrt(t*t+(1-t)**2),(1-t)/math.sqrt(t*t+(1-t)**2)],
                           [ALPHAS[k]**.5,(1-ALPHAS[k])**.5],rtol=1e-12)
 # Standard Gaussian clean data has an exact analytic marginal at every time.
 initial=torch.linspace(-3,3,100,dtype=torch.float64).reshape(1,1,10,10)
 labels=torch.zeros(1,dtype=torch.long)
 def oracle(state,t,labels):return (1-t)*state/(t*t+(1-t)**2)
 actual=integrate(initial,labels,oracle)
 expected=initial*(1-T_START)/math.sqrt(T_START*T_START+(1-T_START)**2)
 difference=float((actual-expected).abs().max())
 torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-5)
 # Learned variance channels are ignored just as the native clean-estimate conversion requires.
 x=torch.randn(2,4,8,8);eps=torch.randn_like(x);k=500;alpha=float(ALPHAS[k])**.5;beta=float(1-ALPHAS[k])**.5
 raw=torch.cat([eps,torch.randn_like(eps)],1);t=torch.full((2,),k,dtype=torch.long)
 parsed=d.p_mean_variance(lambda *args,**kwargs:raw,x,t,clip_denoised=False)['pred_xstart']
 manual=(x-beta*eps)/alpha;torch.testing.assert_close(parsed,manual,rtol=1e-5,atol=1e-5)
 result=dict(passed=True,t_start=T_START,maximum_sigma=float(SIGMAS[-1]),heun_steps=50,network_evaluations=99,
             gaussian_oracle_max_abs=difference,native_schedule_exact=True,normalization_and_time_mapping=True,
             official_epsilon_conversion=True,validation_scope='CPU mathematics; each job validates actual model conversion on GPU before generating')
 (HERE/'validation-cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
