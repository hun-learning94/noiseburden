"""Monte Carlo residual spectrum Cov(R_t), R_t = E[X_1 | Y_t] - l_t(Y_t), for one (distribution, t).

Theorem joint-width: under affine patch recovery, the best approximation error at width
d_h is sum_{j > d_h - d_p} lambda_j(Cov(R_t)). This script stores the lambda_j and the
eigenvectors, plus the Bayes and affine risks, for the sweep analysis.
"""
import argparse
import json
import os
import sys

import torch

from common import SCHEMA, code_hash
from gmm import add_dist_args, build_dist, sigma_of_t


@torch.no_grad()
def residual_stats(dist, t, n, batch, seed, device):
    sigma = sigma_of_t(t)
    d = dist.d
    gen = torch.Generator(device=device).manual_seed(seed)
    s_r = torch.zeros(d, dtype=torch.float64, device=device)
    s_rr = torch.zeros(d, d, dtype=torch.float64, device=device)
    s_ry = torch.zeros(d, d, dtype=torch.float64, device=device)
    s_y = torch.zeros(d, dtype=torch.float64, device=device)
    bayes = affine_risk = 0.0
    done = 0
    while done < n:
        b = min(batch, n - done)
        x1 = dist.sample(b, gen, dtype=torch.float64)
        y = x1 + sigma * torch.randn(x1.shape, generator=gen, device=device, dtype=torch.float64)
        dstar = dist.posterior_mean(y, sigma)
        lin = dist.affine(y, sigma)
        # multi-patch samples: pool patches (all positions share the same statistics)
        x1, y, dstar, lin = (q.reshape(-1, d) for q in (x1, y, dstar, lin))
        r = dstar - lin
        s_r += r.sum(0)
        s_rr += r.T @ r
        s_y += y.sum(0)
        s_ry += r.T @ y
        bayes += float(((x1 - dstar) ** 2).sum())
        affine_risk += float(((x1 - lin) ** 2).sum())
        done += b
    n = n * getattr(dist, "M", 1)
    mean_r = s_r / n
    cov = s_rr / n - torch.outer(mean_r, mean_r)
    cross = s_ry / n - torch.outer(mean_r, s_y / n)  # Cov(R, Y), zero in population
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return {
        "eigvals": evals[order].clamp_min(0).cpu(),
        "eigvecs": evecs[:, order].cpu(),
        "resid_energy": float(torch.trace(cov)),
        "mean_r_norm": float(mean_r.norm()),
        "cross_cov_max": float(cross.abs().max()),
        "bayes": bayes / n,
        "affine_risk": affine_risk / n,
    }


def main():
    p = argparse.ArgumentParser()
    add_dist_args(p)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--batch", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_code_hash = code_hash("spectrum.py")
    dist = build_dist(args).to(args.device)
    st = residual_stats(dist, args.t, args.n, args.batch, args.seed, args.device)
    meta = {k: v for k, v in vars(args).items()}
    meta["digest"] = dist.digest
    meta["argv"] = [x for x in sys.argv[1:]]
    meta["code_hash"] = run_code_hash
    meta["rows"] = args.n * getattr(dist, "M", 1)
    summary = {k: st[k] for k in ("resid_energy", "mean_r_norm", "cross_cov_max", "bayes", "affine_risk")}
    if args.dist == "binary":
        quad = dist.residual_eigs_quadrature(sigma_of_t(args.t))
        summary["quad_eigs_head"] = quad[:8].tolist()
        summary["mc_eigs_head"] = st["eigvals"][:8].tolist()
        summary["quad_energy"] = float(quad.sum())
        st["quad_eigvals"] = quad
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.{os.getpid()}.tmp"
    torch.save({"meta": meta, **st}, tmp)
    os.replace(tmp, args.out)  # .pt first: a sidecar always implies a matching .pt
    with open(tmp + ".json", "w") as f:  # sidecar read by sweep.py to validate reuse
        json.dump({"schema": SCHEMA, "meta": meta}, f, indent=1)
    os.replace(tmp + ".json", args.out + ".json")
    print(json.dumps({"meta": meta, **summary}, indent=1))


if __name__ == "__main__":
    main()
