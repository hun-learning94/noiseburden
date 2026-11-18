"""Tractable patch distributions with closed-form population denoisers.

Time convention follows the paper (Sec. setup, App. exp-training):
    X_t = t X_1 + (1 - t) eps,   Y_t = X_t / t = X_1 + sigma_t eps,   sigma_t = (1 - t) / t.
Every distribution exposes
    d, mean (d,), cov (d, d)            exact moments, float64
    sample(n, gen)                      clean patches X_1
    posterior_mean(y, sigma)            population denoiser E[X_1 | Y_t = y]
    affine(y, sigma)                    best affine denoiser l_t(y) from the noisy patch
All denoiser arithmetic runs in float64.
"""
import hashlib
import math

import torch


def _digest(*tensors):
    h = hashlib.sha256()
    for x in tensors:
        h.update(torch.as_tensor(x).detach().to("cpu", torch.float64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _digest_args(**kw):
    """Identity from generating arguments (robust to last-bit differences of QR across CPUs)."""
    import json
    return hashlib.sha256(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:16]


def sigma_of_t(t):
    return (1.0 - t) / t


class _Moments:
    """Best affine denoiser from exact first and second moments (App. affine-baseline)."""

    def _init_moments(self, mean, cov):
        self.mean = mean.to(torch.float64)
        self.cov = cov.to(torch.float64)
        self._lin_cache = {}

    def to(self, device):
        for name, val in list(vars(self).items()):
            if torch.is_tensor(val):
                setattr(self, name, val.to(device))
        self._lin_cache = {}
        return self

    def affine_matrix(self, sigma):
        key = float(sigma)
        if key not in self._lin_cache:
            evals, evecs = torch.linalg.eigh(self.cov)
            shrink = evals / (evals + sigma ** 2)
            self._lin_cache[key] = (evecs * shrink) @ evecs.T
        return self._lin_cache[key]

    def affine(self, y, sigma):
        y = y.to(torch.float64)
        return self.mean + (y - self.mean) @ self.affine_matrix(sigma).T


class BinaryLatentGMM(_Moments):
    """X_1 = U z + sigma0 n with z_j in {-s_j, +s_j} i.i.d. uniform and n ~ N(0, I_d).

    This is a mixture of 2^k equal-weight Gaussians whose means are the hypercube
    vertices U z inside the k-dimensional subspace span(U), with shared covariance
    sigma0^2 I. Writing v = sigma0^2 + sigma^2 and a = sigma0^2 / v, the posterior
    factorizes in the U basis:
        E[X_1 | y] = a y + (1 - a) U E[z | y],   E[z_j | y] = s_j tanh(s_j u_j / v),
    with u = U^T y. On span(U)^perp both E[X_1 | y] and l_t(y) equal a y, so the
    residual R_t = E[X_1 | Y_t] - l_t(Y_t) lies in span(U) and Cov(R_t) has rank <= k.
    Its eigenvectors are the columns of U (coordinates are independent).
    """

    def __init__(self, d, scales, sigma0, seed=0):
        scales = torch.as_tensor(scales, dtype=torch.float64)
        k = scales.numel()
        assert k <= d
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        self.d, self.k = d, k
        self.U = q[:, :k].contiguous()
        self.s = scales
        self.sigma0 = float(sigma0)
        self.digest = _digest_args(kind="binary", d=d, scales=[round(x, 12) for x in scales.tolist()],
                                   sigma0=self.sigma0, seed=seed)
        cov = (self.U * scales ** 2) @ self.U.T + self.sigma0 ** 2 * torch.eye(d, dtype=torch.float64)
        self._init_moments(torch.zeros(d, dtype=torch.float64), cov)

    def sample(self, n, gen, dtype=torch.float32):
        dev = self.U.device
        signs = torch.randint(0, 2, (n, self.k), generator=gen, device=dev).to(torch.float64) * 2 - 1
        noise = torch.randn(n, self.d, generator=gen, device=dev, dtype=torch.float64)
        x = (signs * self.s) @ self.U.T + self.sigma0 * noise
        return x.to(dtype)

    def posterior_mean(self, y, sigma):
        y = y.to(torch.float64)
        v = self.sigma0 ** 2 + sigma ** 2
        a = self.sigma0 ** 2 / v
        u = y @ self.U
        ez = self.s * torch.tanh(self.s * u / v)
        return a * y + (1 - a) * (ez @ self.U.T)

    def residual_eigs_quadrature(self, sigma, n_grid=200001, width=12.0, sort=True):
        """Exact Var of the residual per latent coordinate by 1-D quadrature.

        Coordinate j of U^T Y is u = z + sqrt(v) e with z = +-s_j; along u_j,
        E[X|y] = a u + (1-a) s tanh(s u / v) and l_t = c u with
        c = (s^2 + sigma0^2) / (s^2 + v). Returns the k eigenvalues of Cov(R_t),
        sorted in decreasing order.
        """
        v = self.sigma0 ** 2 + sigma ** 2
        a = self.sigma0 ** 2 / v
        out = []
        for s in self.s.tolist():
            c = (s ** 2 + self.sigma0 ** 2) / (s ** 2 + v)
            lim = s + width * math.sqrt(v)
            u = torch.linspace(-lim, lim, n_grid, dtype=torch.float64)
            dens = 0.5 * (torch.exp(-(u - s) ** 2 / (2 * v)) + torch.exp(-(u + s) ** 2 / (2 * v)))
            dens = dens / math.sqrt(2 * math.pi * v)
            r = a * u + (1 - a) * s * torch.tanh(s * u / v) - c * u
            du = u[1] - u[0]
            out.append(float(torch.sum(r ** 2 * dens) * du))  # E r = 0 by symmetry
        if not sort:  # column order of U: Var of the residual along each latent direction
            return torch.tensor(out, dtype=torch.float64)
        return torch.tensor(sorted(out, reverse=True), dtype=torch.float64)


class FullCovGMM(_Moments):
    """General Gaussian mixture sum_k pi_k N(mu_k, Sigma_k) with exact posterior mean.

    E[X_1 | y] = sum_k w_k(y) [mu_k + Sigma_k (Sigma_k + sigma^2 I)^{-1} (y - mu_k)],
    w_k(y) propto pi_k N(y; mu_k, Sigma_k + sigma^2 I). Per-component eigendecompositions
    Sigma_k = V_k diag(lam_k) V_k^T make every solve diagonal.
    """

    def __init__(self, weights, means, covs):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.means = torch.as_tensor(means, dtype=torch.float64)
        covs = torch.as_tensor(covs, dtype=torch.float64)
        covs = 0.5 * (covs + covs.transpose(-1, -2))
        self.digest = _digest(self.weights, self.means, covs)  # content identity of the mixture
        self.K, self.d = self.means.shape
        self.lam, self.V = torch.linalg.eigh(covs)
        self.lam = self.lam.clamp_min(0.0)
        self.chol = self.V * self.lam.sqrt().unsqueeze(-2)  # V diag(sqrt lam)
        mean = self.weights @ self.means
        second = torch.einsum("k,kij->ij", self.weights, covs) + torch.einsum(
            "k,ki,kj->ij", self.weights, self.means, self.means)
        self._init_moments(mean, second - torch.outer(mean, mean))

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu")
        return cls(blob["weights"], blob["means"], blob["covs"])

    def sample(self, n, gen, dtype=torch.float32):
        dev = self.means.device
        comp = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        xi = torch.randn(n, self.d, generator=gen, device=dev, dtype=torch.float64)
        x = torch.empty(n, self.d, device=dev, dtype=torch.float64)
        for k in torch.unique(comp).tolist():
            idx = (comp == k).nonzero(as_tuple=True)[0]
            x[idx] = self.means[k] + xi[idx] @ self.chol[k].T
        return x.to(dtype)

    def posterior_mean(self, y, sigma, chunk=8):
        y = y.to(torch.float64)
        n = y.shape[0]
        logits = torch.empty(n, self.K, device=y.device, dtype=torch.float64)
        var = self.lam + sigma ** 2  # (K, d)
        logdet = torch.log(var).sum(-1)
        for k0 in range(0, self.K, chunk):
            ks = slice(k0, min(k0 + chunk, self.K))
            proj = torch.einsum("nd,kde->kne", y, self.V[ks]) - torch.einsum(
                "kd,kde->ke", self.means[ks], self.V[ks]).unsqueeze(1)
            maha = (proj ** 2 / var[ks].unsqueeze(1)).sum(-1)  # (kc, n)
            logits[:, ks] = (torch.log(self.weights[ks]) - 0.5 * logdet[ks]).unsqueeze(0) - 0.5 * maha.T
        w = torch.softmax(logits, dim=1)
        out = torch.zeros_like(y)
        shrink = self.lam / var
        for k0 in range(0, self.K, chunk):
            ks = slice(k0, min(k0 + chunk, self.K))
            proj = torch.einsum("nd,kde->kne", y, self.V[ks]) - torch.einsum(
                "kd,kde->ke", self.means[ks], self.V[ks]).unsqueeze(1)
            comp_mean = self.means[ks].unsqueeze(1) + torch.einsum(
                "kne,kde->knd", proj * shrink[ks].unsqueeze(1), self.V[ks])
            out += torch.einsum("nk,knd->nd", w[:, ks], comp_mean)
        return out


class SharedLatentImage(_Moments):
    """An "image" of M patches of dimension d_p with shared and local binary latents:
        X_m = A c + B z_m + sigma0 n_m,   c_j = +-sc_j (shared by all patches), z_{m,i} = +-sl_i,
    with [A, B] orthonormal (d_p x (k_s + k_l)). The posterior factorizes:
        E[c_j | Y]       = sc_j tanh(sc_j sum_m (A^T y_m)_j / v)       (pools every patch)
        E[z_{m,i} | Y]   = sl_i tanh(sl_i (B^T y_m)_i / v)            (own patch only)
        E[X_m | Y]       = a y_m + (1 - a) (A E[c|Y] + B E[z_m|Y]),   v = sigma0^2 + sigma^2, a = sigma0^2 / v.
    `affine` is the best affine denoiser from the noisy patch alone (patch-marginal moments), so
    R_t contains both local nonlinearity and information from the other patches. With M = 1 and
    k_s = 0 this is BinaryLatentGMM. Samples have shape (n, M, d_p).
    """

    def __init__(self, d, M, shared_scales, local_scales, sigma0, seed=0):
        sc = torch.as_tensor(shared_scales, dtype=torch.float64).reshape(-1)
        sl = torch.as_tensor(local_scales, dtype=torch.float64).reshape(-1)
        ks, kl = sc.numel(), sl.numel()
        assert ks + kl <= d
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        self.d, self.M, self.ks, self.kl = d, M, ks, kl
        self.A, self.B = q[:, :ks].contiguous(), q[:, ks:ks + kl].contiguous()
        self.sc, self.sl, self.sigma0 = sc, sl, float(sigma0)
        self.digest = _digest_args(kind="shared", d=d, M=M, shared=[round(x, 12) for x in sc.tolist()],
                                   local=[round(x, 12) for x in sl.tolist()], sigma0=self.sigma0, seed=seed)
        cov = (self.A * sc ** 2) @ self.A.T + (self.B * sl ** 2) @ self.B.T
        cov = cov + self.sigma0 ** 2 * torch.eye(d, dtype=torch.float64)
        self._init_moments(torch.zeros(d, dtype=torch.float64), cov)

    def sample(self, n, gen, dtype=torch.float32):
        dev = self.A.device
        c = (torch.randint(0, 2, (n, self.ks), generator=gen, device=dev).to(torch.float64) * 2 - 1) * self.sc
        z = (torch.randint(0, 2, (n, self.M, self.kl), generator=gen, device=dev).to(torch.float64) * 2 - 1) * self.sl
        noise = torch.randn(n, self.M, self.d, generator=gen, device=dev, dtype=torch.float64)
        x = (c @ self.A.T).unsqueeze(1) + z @ self.B.T + self.sigma0 * noise
        return x.to(dtype)

    def posterior_mean(self, y, sigma):
        y = y.to(torch.float64)
        v = self.sigma0 ** 2 + sigma ** 2
        a = self.sigma0 ** 2 / v
        ec = self.sc * torch.tanh(self.sc * (y @ self.A).sum(1) / v)  # (n, ks)
        ez = self.sl * torch.tanh(self.sl * (y @ self.B) / v)  # (n, M, kl)
        return a * y + (1 - a) * ((ec @ self.A.T).unsqueeze(1) + ez @ self.B.T)

    def posterior_mean_per_sample(self, y, sigma):
        """Same as posterior_mean with one noise level per image: sigma has shape (n,)."""
        y = y.to(torch.float64)
        v = self.sigma0 ** 2 + sigma.to(torch.float64) ** 2
        a = (self.sigma0 ** 2 / v).view(-1, 1, 1)
        ec = self.sc * torch.tanh(self.sc * (y @ self.A).sum(1) / v.view(-1, 1))
        ez = self.sl * torch.tanh(self.sl * (y @ self.B) / v.view(-1, 1, 1))
        return a * y + (1 - a) * ((ec @ self.A.T).unsqueeze(1) + ez @ self.B.T)


def scales_from_spec(spec):
    """'flat:K:S' -> K equal scales S;  'pow:K:S1:ALPHA' -> S1 * j^-ALPHA, j = 1..K;  'none' -> []."""
    if spec in ("none", ""):
        return []
    kind, *vals = spec.split(":")
    if kind == "flat":
        k, s = int(vals[0]), float(vals[1])
        return [s] * k
    if kind == "pow":
        k, s1, alpha = int(vals[0]), float(vals[1]), float(vals[2])
        return [s1 * (j + 1) ** (-alpha) for j in range(k)]
    raise ValueError(spec)


def build_dist(args):
    if args.dist == "binary":
        return BinaryLatentGMM(args.d, scales_from_spec(args.scales), args.sigma0, args.dist_seed)
    if args.dist == "gmm":
        return FullCovGMM.load(args.gmm_path)
    if args.dist == "shared":
        return SharedLatentImage(args.d, args.M, scales_from_spec(args.shared_scales),
                                 scales_from_spec(args.scales), args.sigma0, args.dist_seed)
    raise ValueError(args.dist)


def add_dist_args(p):
    p.add_argument("--dist", choices=["binary", "gmm", "shared"], default="binary")
    p.add_argument("--M", type=int, default=1, help="shared: patches per image")
    p.add_argument("--shared_scales", default="none", help="shared: scale spec of the shared latents")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--scales", default="flat:16:0.5")
    p.add_argument("--sigma0", type=float, default=0.1)
    p.add_argument("--dist_seed", type=int, default=0)
    p.add_argument("--gmm_path", default=None)
