"""Probe intermediate activations of a trained JiT denoiser for linear
reconstruction of the input z (and its components x and eps).

For each (layer, t) cell we fit token-wise linear probes W: R^{D+1} -> R^{p*p*c}
by streaming closed-form ridge regression (no activations are stored), then
report explained variance (EV) on held-out images for three targets:
    z   -- the network input,
    x   -- the clean-image component,
    eps -- the noise component.

Usage (single GPU is enough):
    python probe_z.py --ckpt /path/to/checkpoint-last.pth \
        --data_path /path/to/imagenet \
        --num_fit 8000 --num_eval 2000 --batch_size 64 \
        --t_grid 0.1,0.3,0.5,0.7,0.9 --out probe_results.npz
"""
import argparse
import numpy as np
import torch

from denoiser import Denoiser


def patchify(img, p):
    """(N, C, H, W) -> (N, T, p*p*c), inverse layout of JiT.unpatchify."""
    n, c, hh, ww = img.shape
    h, w = hh // p, ww // p
    x = img.reshape(n, c, h, p, w, p)
    x = torch.einsum('nchpwq->nhwpqc', x)
    return x.reshape(n, h * w, p * p * c)


class Accum:
    """Streaming sufficient statistics for one (layer, t) cell."""
    def __init__(self, d_feat, d_tgt, targets):
        self.sxx = torch.zeros(d_feat + 1, d_feat + 1, dtype=torch.float64)
        self.sxy = {k: torch.zeros(d_feat + 1, d_tgt, dtype=torch.float64) for k in targets}
        self.W = None
        self.err = {k: 0.0 for k in targets}
        self.ysum = {k: torch.zeros(d_tgt, dtype=torch.float64) for k in targets}
        self.ysq = {k: 0.0 for k in targets}
        self.n_eval = 0

    def fit_batch(self, feats, tgts):
        f = torch.cat([feats, torch.ones(feats.shape[0], 1, dtype=torch.float64)], 1)
        self.sxx += f.T @ f
        for k, y in tgts.items():
            self.sxy[k] += f.T @ y

    def solve(self, lam):
        d = self.sxx.shape[0]
        reg = lam * (torch.diagonal(self.sxx)[:-1].mean()) * torch.eye(d, dtype=torch.float64)
        reg[-1, -1] = 1e-8  # do not regularize the bias
        self.W = {k: torch.linalg.solve(self.sxx + reg, sxy) for k, sxy in self.sxy.items()}

    def eval_batch(self, feats, tgts):
        f = torch.cat([feats, torch.ones(feats.shape[0], 1, dtype=torch.float64)], 1)
        for k, y in tgts.items():
            pred = f @ self.W[k]
            self.err[k] += ((y - pred) ** 2).sum().item()
            self.ysum[k] += y.sum(0)
            self.ysq[k] += (y ** 2).sum().item()
        self.n_eval += f.shape[0]

    def ev(self):
        out = {}
        for k in self.err:
            mean = self.ysum[k] / max(self.n_eval, 1)
            var = self.ysq[k] - self.n_eval * (mean ** 2).sum().item()
            out[k] = 1.0 - self.err[k] / max(var, 1e-12)
        return out


class TimeFourier(torch.nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        self.n_freq = n_freq

    def forward(self, t):
        k = torch.arange(1, self.n_freq + 1, device=t.device, dtype=t.dtype)
        tf = t[:, None]
        return torch.cat([torch.sin(2 * torch.pi * tf * k), torch.cos(2 * torch.pi * tf * k)], dim=1)


class MLPProbe(torch.nn.Module):
    """Token-wise 2-layer MLP probe with additive time conditioning."""
    def __init__(self, d_feat, d_tgt, hidden=1024, temb_dim=16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_feat, elementwise_affine=False)
        self.fc1 = torch.nn.Linear(d_feat, hidden)
        self.ft = torch.nn.Linear(temb_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, d_tgt)

    def forward(self, x, temb):
        x = self.norm(x)
        h = torch.nn.functional.gelu(self.fc1(x) + self.ft(temb)[:, None, :])
        return self.fc2(h)


class BlockProbe(torch.nn.Module):
    """One pre-LN transformer block (cross-token attention + MLP) with
    adaLN-zero time conditioning, followed by a linear head."""
    def __init__(self, d_feat, d_tgt, heads=8, temb_dim=16):
        super().__init__()
        self.ln_in = torch.nn.LayerNorm(d_feat, elementwise_affine=False)
        self.ln_out = torch.nn.LayerNorm(d_feat, elementwise_affine=False)
        self.ln1 = torch.nn.LayerNorm(d_feat, elementwise_affine=False)
        self.ln2 = torch.nn.LayerNorm(d_feat, elementwise_affine=False)
        self.attn = torch.nn.MultiheadAttention(d_feat, heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_feat, 4 * d_feat), torch.nn.GELU(), torch.nn.Linear(4 * d_feat, d_feat))
        self.ada = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(temb_dim, 6 * d_feat))
        torch.nn.init.zeros_(self.ada[-1].weight)
        torch.nn.init.zeros_(self.ada[-1].bias)
        self.head = torch.nn.Linear(d_feat, d_tgt)

    def forward(self, x, temb):
        x = self.ln_in(x)
        s1, c1, g1, s2, c2, g2 = self.ada(temb)[:, None, :].chunk(6, dim=-1)
        h = self.ln1(x) * (1 + c1) + s1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.ln2(x) * (1 + c2) + s2
        x = x + g2 * self.mlp(h)
        return self.head(self.ln_out(x))


def get_batches(args, n_images, seed):
    g = torch.Generator().manual_seed(seed)
    if args.fake_data:
        done = 0
        while done < n_images:
            b = min(args.batch_size, n_images - done)
            yield (torch.randint(0, 256, (b, 3, args.img_size, args.img_size), generator=g).float() / 255.0,
                   torch.randint(0, args.class_num, (b,), generator=g))
            done += b
        return
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from util.crop import center_crop_arr
    tf_ = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.PILToTensor(),
    ])
    ds = datasets.ImageFolder(args.data_path + '/train', transform=tf_)
    idx = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    sub = torch.utils.data.Subset(ds, idx)
    dl = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, drop_last=False)
    for x, y in dl:
        yield x.float() / 255.0, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--which', default='model_ema1', choices=['model', 'model_ema1', 'model_ema2'])
    ap.add_argument('--data_path', default='')
    ap.add_argument('--fake_data', action='store_true')
    ap.add_argument('--num_fit', type=int, default=8000)
    ap.add_argument('--num_eval', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--t_grid', default='0.1,0.3,0.5,0.7,0.9')
    ap.add_argument('--layers', default='')  # e.g. "0,2,4,6,8,10,12"; empty = all
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--probe', default='linear', choices=['linear', 'mlp', 'block'],
                    help='linear: closed-form ridge per (layer,t); mlp/block: gradient-trained, one probe per (layer,target), t-conditioned')
    ap.add_argument('--probe_hidden', type=int, default=1024)
    ap.add_argument('--probe_heads', type=int, default=8)
    ap.add_argument('--probe_epochs', type=int, default=4)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--out', default='probe_results.npz')
    cli = ap.parse_args()

    ckpt = torch.load(cli.ckpt, map_location='cpu', weights_only=False)
    args = ckpt['args']
    for k in ['pred_type', 'loss_type', 'shortcut']:
        if not hasattr(args, k):
            setattr(args, k, {'pred_type': 'x', 'loss_type': 'v', 'shortcut': 'none'}[k])
    cli.img_size, cli.class_num = args.img_size, args.class_num

    den = Denoiser(args)
    key = cli.which if cli.which in ckpt else 'model'
    den.load_state_dict(ckpt[key])
    net = den.net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()
    print(f"loaded {cli.ckpt} (epoch {ckpt.get('epoch', '?')}), probing weights={key}, "
          f"pred={args.pred_type} shortcut={args.shortcut}")

    n_blocks = len(net.blocks)
    layer_ids = list(range(n_blocks + 1)) if not cli.layers else [int(s) for s in cli.layers.split(',')]
    t_grid = [float(s) for s in cli.t_grid.split(',')]
    p = net.patch_size
    d_feat = net.blocks[0].adaLN_modulation[-1].out_features // 6  # hidden size
    d_tgt = p * p * 3
    ctx_len = net.in_context_len
    ctx_start = net.in_context_start

    store = {}
    hooks = []

    def keep(li, x):
        store[li] = x.detach().float()   # stays on device; linear path moves to cpu below

    def make_hook(i):
        def hook(m, inp, out):
            o = out
            if ctx_len > 0 and i >= ctx_start:
                o = o[:, ctx_len:]
            keep(i + 1, o)
        return hook

    hooks.append(net.blocks[0].register_forward_pre_hook(lambda m, inp: keep(0, inp[0])))
    for i, blk in enumerate(net.blocks):
        hooks.append(blk.register_forward_hook(make_hook(i)))

    acc = {(li, ti): Accum(d_feat, d_tgt, ('z', 'x', 'eps'))
           for li in layer_ids for ti in range(len(t_grid))}

    def run_pass(n_images, seed, mode):
        for x, y in get_batches(cli, n_images, seed):
            x, y = x.to(device), y.to(device)
            for ti, tv in enumerate(t_grid):
                eps = torch.randn_like(x) * args.noise_scale
                t = torch.full((x.shape[0],), tv, device=device)
                z = tv * x + (1 - tv) * eps
                with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                                    enabled=(device == 'cuda')):
                    net(z, t, y)
                tgts_full = {'z': patchify(z, p), 'x': patchify(x, p), 'eps': patchify(eps, p)}
                for li in layer_ids:
                    feats = store[li].reshape(-1, d_feat).double().cpu()
                    tgts = {k: v.reshape(-1, d_tgt).double().cpu() for k, v in tgts_full.items()}
                    if mode == 'fit':
                        acc[(li, ti)].fit_batch(feats, tgts)
                    else:
                        acc[(li, ti)].eval_batch(feats, tgts)

    if cli.probe == 'linear':
        print("pass 1/2: fitting probes ...")
        run_pass(cli.num_fit, seed=0, mode='fit')
        for a in acc.values():
            a.solve(cli.lam)
        print("pass 2/2: evaluating ...")
        run_pass(cli.num_eval, seed=1, mode='eval')
        ev = {k: np.zeros((len(layer_ids), len(t_grid))) for k in ('z', 'x', 'eps')}
        for a_i, li in enumerate(layer_ids):
            for ti in range(len(t_grid)):
                for k, v in acc[(li, ti)].ev().items():
                    ev[k][a_i, ti] = v
    else:
        targets = ('z', 'x', 'eps')
        temb_f = TimeFourier()
        mk = (lambda: MLPProbe(d_feat, d_tgt, cli.probe_hidden)) if cli.probe == 'mlp' \
            else (lambda: BlockProbe(d_feat, d_tgt, cli.probe_heads))
        probes = {(li, k): mk().to(device) for li in layer_ids for k in targets}
        opt = torch.optim.AdamW([p for m in probes.values() for p in m.parameters()], lr=cli.probe_lr)

        for ep in range(cli.probe_epochs):
            print(f"fit epoch {ep + 1}/{cli.probe_epochs} ...")
            for xb, yb in get_batches(cli, cli.num_fit, seed=0):
                xb, yb = xb.to(device), yb.to(device)
                for tv in t_grid:
                    eps = torch.randn_like(xb) * args.noise_scale
                    t = torch.full((xb.shape[0],), tv, device=device)
                    z = tv * xb + (1 - tv) * eps
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                                         enabled=(device == 'cuda')):
                        net(z, t, yb)
                    temb = temb_f(t)
                    tg = {'z': patchify(z, p), 'x': patchify(xb, p), 'eps': patchify(eps, p)}
                    loss = 0.
                    for li in layer_ids:
                        f = store[li]
                        for k in targets:
                            loss = loss + ((probes[(li, k)](f, temb) - tg[k]) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for m in probes.values() for p in m.parameters()], 1.0)
                    opt.step()

        print("evaluating ...")
        stat = {(li, ti, k): [0.0, torch.zeros(d_tgt, dtype=torch.float64, device=device), 0.0, 0]
                for li in layer_ids for ti in range(len(t_grid)) for k in targets}
        with torch.no_grad():
            for xb, yb in get_batches(cli, cli.num_eval, seed=1):
                xb, yb = xb.to(device), yb.to(device)
                for ti, tv in enumerate(t_grid):
                    eps = torch.randn_like(xb) * args.noise_scale
                    t = torch.full((xb.shape[0],), tv, device=device)
                    z = tv * xb + (1 - tv) * eps
                    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                        enabled=(device == 'cuda')):
                        net(z, t, yb)
                    temb = temb_f(t)
                    tg = {'z': patchify(z, p), 'x': patchify(xb, p), 'eps': patchify(eps, p)}
                    for li in layer_ids:
                        f = store[li]
                        for k in targets:
                            y = tg[k].reshape(-1, d_tgt).double()
                            pr = probes[(li, k)](f, temb).reshape(-1, d_tgt).double()
                            st = stat[(li, ti, k)]
                            st[0] += ((y - pr) ** 2).sum().item()
                            st[1] += y.sum(0)
                            st[2] += (y ** 2).sum().item()
                            st[3] += y.shape[0]
        ev = {k: np.zeros((len(layer_ids), len(t_grid))) for k in targets}
        for a_i, li in enumerate(layer_ids):
            for ti in range(len(t_grid)):
                for k in targets:
                    err, ysum, ysq, n = stat[(li, ti, k)]
                    mean = ysum / max(n, 1)
                    var = ysq - n * (mean ** 2).sum().item()
                    ev[k][a_i, ti] = 1.0 - err / max(var, 1e-12)
    for h in hooks:
        h.remove()
    np.savez(cli.out, layers=np.array(layer_ids), t_grid=np.array(t_grid), **ev)

    for k in ('z', 'x', 'eps'):
        print(f"\nEV[{k}]  (rows: layer, cols: t = {t_grid})")
        for a_i, li in enumerate(layer_ids):
            row = "  ".join(f"{ev[k][a_i, ti]:6.3f}" for ti in range(len(t_grid)))
            print(f"  layer {li:2d}: {row}")
    print(f"\nsaved -> {cli.out}")


if __name__ == '__main__':
    main()