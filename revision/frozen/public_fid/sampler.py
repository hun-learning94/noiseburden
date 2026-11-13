"""A native VP epsilon predictor on the normalized JiT linear corruption path."""
import math
import numpy as np
import torch

ALPHAS=np.cumprod(1-np.linspace(.0001,.02,1000,dtype=np.float64))
SIGMAS=np.sqrt((1-ALPHAS)/ALPHAS)
T_START=float(1/(1+SIGMAS[-1]))
STEPS=50

def native_t(t):
    return float(np.interp(math.log((1-t)/t),np.log(SIGMAS),np.arange(1000)))

def integrate(initial,labels,epsilon,steps=STEPS):
    """Initial is standard Gaussian; omit the negligible clean term at finite sigma_max."""
    x=initial*(1-T_START)
    grid=np.linspace(T_START,1,steps+1)
    evaluations=0
    def velocity(state,t):
        nonlocal evaluations
        eps=epsilon(state,t,labels);evaluations+=1
        return (state-eps.float())/t
    for i in range(steps):
        t=float(grid[i]);tn=float(grid[i+1]);dt=tn-t
        v=velocity(x,t);pred=x+dt*v
        x=pred if i==steps-1 else x+dt*.5*(v+velocity(pred,tn))
        if not torch.isfinite(x).all():raise FloatingPointError(f'Non-finite Heun state at step {i}')
    assert evaluations==2*steps-1
    return x
