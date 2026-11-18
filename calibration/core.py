"""Scalar correction and the frozen FP32 JiT sampling arithmetic.

Clean data is at t=1: X_t = t X_1 + (1-t) epsilon. The public worker
supports the released ImageNet256 JiT-B/16 checkpoint recorded in the protocol.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/calibration/protocol.json'
METRICS = ROOT / 'metrics/calibration'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def key(t):
    return f'{float(t):.6f}'


def patches(x):
    b, c, h, w = x.shape
    if (c, h, w) != (3, 256, 256):
        raise ValueError('This reproduction uses 3x256x256 images and 16x16 patches.')
    return x.reshape(b, c, h//16, 16, w//16, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, 768)


def unpatch(p):
    return p.reshape(len(p), 16, 16, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(len(p), 3, 256, 256)


def project_discarded(x, rows):
    p = patches(x)
    return unpatch(p - (p @ rows.T) @ rows)


def guidance(t):
    return 3. if np.float32(.1) < np.float32(t) < np.float32(1.) else 1.


@torch.inference_mode()
def predictions(model, z, tv, labels):
    # Preserve the original operation order, including guidance in velocity space.
    t = torch.full((len(z), 1, 1, 1), float(tv), device=z.device, dtype=torch.float32)
    cond = model.net(z, t.flatten(), labels)
    uncond = model.net(z, t.flatten(), torch.full_like(labels, 1000))
    vc = (cond-z)/(1-t).clamp_min(model.t_eps)
    vu = (uncond-z)/(1-t).clamp_min(model.t_eps)
    return {mode: (uncond+w*(cond-uncond), vu+w*(vc-vu))
            for mode, w in [('unguided', 1.), ('guided', guidance(tv))]}


def multiplier(t, method):
    if method == 'baseline':
        return 0.
    if method == 'full':
        return 1.
    if method == 'late_half':
        return .5 if round(float(t), 6) >= .9 else 1.
    raise ValueError(method)


def load_coefficients(folder=METRICS/'fit'):
    return {key(d['t']): d['coefficients'] for p in sorted(Path(folder).glob('*.json'))
            if (d := read(p)).get('complete') and 'coefficients' in d}


@torch.inference_mode()
def velocity(model, rows, x, t, labels, method, coefficients):
    clean, v = predictions(model, x, float(t), labels)['guided']
    if method != 'baseline' and float(t) > 0:
        vbar = coefficients[key(t)]['guided']['vbar']
        # Stable equivalent of c_t * Q(X_t/t - f_t); do not divide by t.
        gain = vbar*t/(vbar*t*t + (1-t)**2)
        v = v + (gain*multiplier(t, method))*project_discarded(x-t*clean, rows)/(1-t).clamp_min(model.t_eps)
    return v


@torch.inference_mode()
def sample(model, rows, z, labels, method, coefficients):
    ts = torch.linspace(0., 1., 51, device=z.device)
    for i in range(49):
        t, tn = ts[i], ts[i+1]
        v = velocity(model, rows, z, t, labels, method, coefficients)
        euler = z + (tn-t)*v
        vn = velocity(model, rows, euler, tn, labels, method, coefficients)
        z = z + (tn-t)*.5*(v+vn)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite sample at step {i}')
    z = z + (ts[-1]-ts[-2])*velocity(model, rows, z, ts[-2], labels, method, coefficients)
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite terminal sample')
    return z


def load_model(checkpoint, floor=.05):
    protocol = read(CONFIG)
    if digest(checkpoint) != protocol['checkpoint']['sha256']:
        raise ValueError('Checkpoint does not match the released EMA1 experiment.')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference requires CUDA. The report and verification commands run on CPU.')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(ROOT/'calibration/frozen/jit'))
    from denoiser import Denoiser
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    state = ck['model_ema1']
    rows = np.load(METRICS/'encoder_rows.npy')
    a = state['net.x_embedder.proj1.weight'].detach().double().reshape(128, 768).numpy()
    b = state['net.x_embedder.proj2.weight'].detach().double().reshape(768, 128).numpy()
    if np.linalg.matrix_rank(a) != 128 or np.linalg.matrix_rank(b) != 128:
        raise ValueError('Unexpected patch-embedding rank')
    np.testing.assert_allclose(rows@rows.T, np.eye(128), rtol=0, atol=1e-10)
    np.testing.assert_allclose(a-(a@rows.T)@rows, 0., rtol=0, atol=1e-10)
    args = SimpleNamespace(model='JiT-B/16', img_size=256, class_num=1000,
        attn_dropout=0., proj_dropout=0., label_drop_prob=.1, P_mean=-.8, P_std=.8,
        t_eps=floor, noise_scale=1., ema_decay1=.9999, ema_decay2=.9996,
        cfg=3., interval_min=.1, interval_max=1., sampling_method='heun', num_sampling_steps=50)
    model = Denoiser(args)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval().requires_grad_(False), torch.from_numpy(rows).cuda()


@torch.inference_mode()
def preflight(model, rows):
    # Also runs for unclamped generation; no network is evaluated at t=1.
    z = torch.randn((1, 3, 256, 256), device='cuda',
        generator=torch.Generator(device='cuda').manual_seed(1711))
    y = torch.zeros(1, device='cuda', dtype=torch.long)
    assert torch.equal(unpatch(patches(z)), z)
    for tv in [.02, .1, .5, .9, .98]:
        t = torch.full((1, 1, 1, 1), tv, device='cuda')
        v = predictions(model, z, tv, y)['guided'][1]
        torch.testing.assert_close(v, model._forward_sample(z, t, y), atol=1e-6, rtol=1e-6)
    torch.manual_seed(1711)
    expected = model.generate(y)
    torch.manual_seed(1711)
    initial = torch.randn((1, 3, 256, 256), device='cuda')
    actual = sample(model, rows.float(), initial, y, 'baseline', {})
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    return {'baseline_trajectory_max_difference': float((actual-expected).abs().max()),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'precision': 'FP32; TF32 disabled', 'endpoint_clamp': model.t_eps}
