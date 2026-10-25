"""Compare learned a(t), b(t) against the analytic precond coefficients.
Usage: python plot_coeff.py output/xloss_predeps_sclearned/checkpoint-40.pth
Prints a table and (if matplotlib is available) saves learned_coeff.png.
"""
import sys, torch, numpy as np
from denoiser import Denoiser

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
args = ckpt['args']
den = Denoiser(args)
key = 'model_ema1' if 'model_ema1' in ckpt else 'model'
den.load_state_dict(ckpt[key])
den.eval()
assert den.time_coeff is not None, "this checkpoint has no learned coefficients (shortcut != learned*)"

t = torch.linspace(0.02, 0.98, 49).view(-1, 1, 1, 1)
with torch.no_grad():
    a, b = den.time_coeff(t)
t = t.flatten().numpy(); a = a.flatten().numpy(); b = b.flatten().numpy()
a_star = 1.0 / (1.0 - t)
b_star = -t / (1.0 - t)

print(f"loaded {sys.argv[1]}  (epoch {ckpt.get('epoch','?')}, weights={key}, shortcut={args.shortcut})")
print(f"{'t':>5} {'a_learn':>9} {'a*':>9} {'da':>8}   {'b_learn':>9} {'b*':>9} {'db':>8}")
for i in range(0, len(t), 3):
    print(f"{t[i]:5.2f} {a[i]:9.3f} {a_star[i]:9.3f} {a[i]-a_star[i]:8.3f}   "
          f"{b[i]:9.3f} {b_star[i]:9.3f} {b[i]-b_star[i]:8.3f}")

# summary distances (weight by dt; skip t->1 where analytic blows up)
m = t < 0.9
print(f"\nmean |a - a*| over t<0.9: {np.abs(a-a_star)[m].mean():.3f}")
print(f"mean |b - b*| over t<0.9: {np.abs(b-b_star)[m].mean():.3f}")

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(t, a, 'o-', ms=3, label='learned a(t)')
    ax[0].plot(t, a_star, '--', label=r'analytic $1/(1-t)$')
    ax[1].plot(t, b, 's-', ms=3, label='learned b(t)')
    ax[1].plot(t, b_star, '--', label=r'analytic $-t/(1-t)$')
    for x in ax:
        x.set_xlabel('t'); x.legend(); x.set_ylim(-12, 12); x.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('learned_coeff1.png', dpi=200)
    print("\nsaved -> learned_coeff.png")
except Exception as e:
    print(f"\n(plot skipped: {e})")
