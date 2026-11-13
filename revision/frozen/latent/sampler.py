"""Heun50 on u(t)=t*z_clean+(1-t)*epsilon; native finite maximum noise."""
import numpy as np
import torch

def integrate(initial,labels,epsilon,t_start,steps=50):
 x=initial*(1-t_start);grid=np.linspace(t_start,1,steps+1);nfe=0
 def velocity(state,t):
  nonlocal nfe
  nfe+=1;return (state-epsilon(state,t,labels).float())/t
 for i in range(steps):
  t,tn=float(grid[i]),float(grid[i+1]);dt=tn-t;v=velocity(x,t);pred=x+dt*v
  x=pred if i==steps-1 else x+.5*dt*(v+velocity(pred,tn))
  if not torch.isfinite(x).all():raise FloatingPointError(f'Nonfinite state at step{i}')
 assert nfe==2*steps-1
 return x
