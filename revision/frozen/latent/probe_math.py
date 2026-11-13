"""Streaming affine ridge probes, with image-level uncertainty estimates."""
import numpy as np
import torch

RIDGES = [0., 1e-8, 1e-6, 1e-4, 1e-2, 1.]


class Moments:
    def __init__(self, d, q, device):
        self.n = 0
        self.sx = torch.zeros(d, dtype=torch.float64, device=device)
        self.sy = torch.zeros(q, dtype=torch.float64, device=device)
        self.xx = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.xy = torch.zeros(d, q, dtype=torch.float64, device=device)
        self.yy = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, x, y):
        x = x.reshape(-1, x.shape[-1]).double()
        y = y.reshape(-1, y.shape[-1]).double()
        assert len(x) == len(y) and torch.isfinite(x).all() and torch.isfinite(y).all()
        self.n += len(x)
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx.addmm_(x.T, x)
        self.xy.addmm_(x.T, y)
        self.yy += y.square().sum()

    def loss(self, w, b):
        sse = self.yy + (w * (self.xx @ w)).sum() - 2 * (w * self.xy).sum()
        sse += 2 * (b * (self.sx @ w - self.sy)).sum() + self.n * b.square().sum()
        return float(sse / (self.n * len(b)))


def fit(train, dev):
    mx, my = train.sx / train.n, train.sy / train.n
    xx = train.xx / train.n - mx[:, None] * mx[None, :]
    xy = train.xy / train.n - mx[:, None] * my[None, :]
    xx = (xx + xx.T) * .5
    eigenvalues, vectors = torch.linalg.eigh(xx)
    eigenvalues = eigenvalues.clamp_min(0)
    projected = vectors.T @ xy
    scale = float(eigenvalues.mean())
    cutoff = float(eigenvalues[-1]) * 1e-12
    candidates, best = [], None
    for ridge in RIDGES:
        denominator = eigenvalues + ridge * scale
        inverse = torch.where(denominator > cutoff, denominator.reciprocal(), 0.)
        weight = vectors @ (inverse[:, None] * projected)
        bias = my - mx @ weight
        loss = dev.loss(weight, bias)
        candidates.append(dict(relative_ridge=ridge, dev_mse=loss))
        if best is None or loss < best[0]:
            best = (loss, weight, bias, ridge)
    return best[1], best[2], dict(relative_ridge=best[3], dev_mse=best[0], candidates=candidates,
                                fit_rows=train.n, covariance_rank=int((eigenvalues > cutoff).sum()),
                                feature_dimension=len(mx), target_dimension=len(my),
                                intercept='one shared intercept across images and spatial positions')


def metrics(mse, means, seconds):
    mse, means, seconds = [np.asarray(a, dtype=np.float64) for a in (mse, means, seconds)]
    variance = np.mean(seconds.mean(0) - means.mean(0)**2)
    ev = 1 - mse.mean() / variance
    rng = np.random.default_rng(240924)
    boot_mse, boot_ev = [], []
    for _ in range(1000):
        ids = rng.integers(len(mse), size=len(mse))
        bm = mse[ids].mean()
        bv = np.mean(seconds[ids].mean(0) - means[ids].mean(0)**2)
        boot_mse.append(bm)
        boot_ev.append(1 - bm / bv)
    return dict(mse=float(mse.mean()), variance=float(variance), ev=float(ev), r2_percent=float(ev*100),
                mse_ci95=np.quantile(boot_mse, [.025, .975]).tolist(),
                ev_ci95=np.quantile(boot_ev, [.025, .975]).tolist(),
                mse_by_image=mse.tolist(), target_mean_by_image=means.tolist(),
                target_second_moment_by_image=seconds.tolist())


def patchify(x, p=2):
    b, c, height, width = x.shape
    return x.reshape(b, c, height//p, p, width//p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p*p*c)


def sample_pixels(x, positions):
    return x.flatten(2).gather(2, positions[:, None, :].expand(-1, x.shape[1], -1)).transpose(1, 2)


def neighborhoods(x, positions):
    """Gather 3x3 neighborhoods without materializing a full-image unfold."""
    width = x.shape[-1]
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1))
    rows, cols = positions // width + 1, positions % width + 1
    return torch.cat([sample_pixels(padded, (rows + dr) * (width + 2) + cols + dc)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)], -1)
