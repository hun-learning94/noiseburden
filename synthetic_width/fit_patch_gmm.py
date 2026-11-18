"""Tier 2: fit a full-covariance Gaussian mixture to ImageNet 8x8 RGB patches (d_p = 192).

Images follow the ImageNet128 preprocessing (ADM center crop to 128, [-1, 1]), patches are
the 16x16 grid of the JiT patchify, flattened as (p, p, C). The fitted mixture is a tractable
patch prior in the style of EPLL (Zoran & Weiss, 2011): its population denoiser is exact, so
the width sweep can be scored against E[X_1 | Y_t] instead of a learned proxy.

EM in float64 on GPU with a covariance floor `reg`. Validation patches come from images that
contribute no fitting patches. The patch cache records its configuration and image lists and is
rejected if they differ from the current arguments. Fitted mixtures are written under a name that
includes their content digest, so refitting never overwrites a mixture a spectrum refers to.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from calibration.frozen.crop import center_crop_arr  # noqa: E402


def list_images(root, n_images, seed):
    rng = random.Random(seed)
    classes = sorted(os.listdir(root))
    per_class = math.ceil(n_images / len(classes))
    files = []
    for c in classes:
        names = sorted(os.listdir(os.path.join(root, c)))
        files += [os.path.join(root, c, f) for f in rng.sample(names, min(per_class, len(names)))]
    rng.shuffle(files)
    return files[:n_images]


def load_patches(path, size, p, per_image, rng):
    img = center_crop_arr(Image.open(path).convert("RGB"), size)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # (H, W, C) in [-1, 1]
    g = size // p
    patches = arr.reshape(g, p, g, p, 3).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * 3)
    idx = rng.sample(range(g * g), per_image) if per_image < g * g else range(g * g)
    return patches[list(idx)]


def extract(files, size, p, per_image, seed, workers):
    from concurrent.futures import ProcessPoolExecutor

    def chunks():
        for i in range(0, len(files), 256):
            yield files[i:i + 256], seed + i

    out = []
    with ProcessPoolExecutor(workers) as ex:
        for arr in ex.map(_load_chunk, [(c, s, size, p, per_image) for c, s in chunks()]):
            out.append(arr)
    return np.concatenate(out)


def _load_chunk(job):
    files, seed, size, p, per_image = job
    rng = random.Random(seed)
    return np.concatenate([load_patches(f, size, p, per_image, rng) for f in files])


def log_gauss(x, means, covs, reg):
    """(N, K) log N(x; mu_k, Sigma_k + reg I)."""
    n, d = x.shape
    out = torch.empty(n, means.shape[0], dtype=x.dtype, device=x.device)
    eye = torch.eye(d, dtype=x.dtype, device=x.device)
    for k in range(means.shape[0]):
        L = torch.linalg.cholesky(covs[k] + reg * eye)
        z = torch.linalg.solve_triangular(L, (x - means[k]).T, upper=False)
        out[:, k] = -0.5 * (z ** 2).sum(0) - torch.log(torch.diagonal(L)).sum() - 0.5 * d * math.log(2 * math.pi)
    return out


def kmeans_init(x, K, iters, gen):
    idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:K]
    c = x[idx].clone()
    for _ in range(iters):
        dist = (x ** 2).sum(1, keepdim=True) - 2 * x @ c.T + (c ** 2).sum(1)
        a = dist.argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                c[k] = x[m].mean(0)
    return a


def em(x, x_val, K, iters, reg, seed, log):
    n, d = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    a = kmeans_init(x, K, 15, gen)
    resp = torch.nn.functional.one_hot(a, K).to(x.dtype)
    prev = -float("inf")
    for it in range(iters):
        nk = resp.sum(0) + 1e-12
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        covs = torch.empty(K, d, d, dtype=x.dtype, device=x.device)
        for k in range(K):
            xc = x - means[k]
            covs[k] = (xc * resp[:, k:k + 1]).T @ xc / nk[k]
        lg = log_gauss(x, means, covs, reg) + torch.log(weights)
        ll = torch.logsumexp(lg, 1)
        resp = torch.softmax(lg, 1)
        train_ll = float(ll.mean()) / d
        val_ll = float(torch.logsumexp(log_gauss(x_val, means, covs, reg) + torch.log(weights), 1).mean()) / d
        log({"iter": it, "train_ll_per_dim": train_ll, "val_ll_per_dim": val_ll,
             "min_weight": float(weights.min())})
        if abs(train_ll - prev) < 1e-6:
            break
        prev = train_ll
    covs = covs + reg * torch.eye(d, dtype=x.dtype, device=x.device)
    return weights, means, covs, val_ll


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--n_images", type=int, default=40000)
    p.add_argument("--per_image", type=int, default=32)
    p.add_argument("--val_images", type=int, default=2000)
    p.add_argument("--Ks", default="32,128")
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--extract_only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "fit_log.jsonl"), "a")

    def log(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    t0 = time.time()
    cache_cfg = {key: getattr(args, key) for key in
                 ("root", "size", "patch", "n_images", "per_image", "val_images", "seed")}
    if args.cache and os.path.exists(args.cache):
        side = json.load(open(args.cache + ".json"))
        assert side["config"] == cache_cfg, ("patch cache was built with other settings", side["config"])
        blob = np.load(args.cache)
        fit_data, val_data = blob["fit"], blob["val"]
    else:
        files = list_images(args.root, args.n_images, args.seed)
        val_files, fit_files = files[:args.val_images], files[args.val_images:]
        fit_data = extract(fit_files, args.size, args.patch, args.per_image, args.seed, args.workers)
        val_data = extract(val_files, args.size, args.patch, args.per_image, args.seed + 1, args.workers)
        if args.cache:
            np.savez(args.cache + ".tmp.npz", fit=fit_data, val=val_data)
            with open(args.cache + ".json", "w") as f:
                json.dump({"config": cache_cfg, "fit_images": fit_files, "val_images": val_files}, f)
            os.replace(args.cache + ".tmp.npz", args.cache)
    assert fit_data.shape[1] == val_data.shape[1] == 3 * args.patch ** 2, fit_data.shape
    log({"fit_patches": int(fit_data.shape[0]), "val_patches": int(val_data.shape[0]),
         "dim": int(fit_data.shape[1]), "load_sec": time.time() - t0})
    if args.extract_only:
        return

    x = torch.from_numpy(fit_data).to("cuda", torch.float64)
    x_val = torch.from_numpy(val_data).to("cuda", torch.float64)
    for K in [int(k) for k in args.Ks.split(",")]:
        log({"fit_K": K})
        w, m, c, val_ll = em(x, x_val, K, args.iters, args.reg, args.seed, log)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gmm import FullCovGMM
        digest = FullCovGMM(w.cpu(), m.cpu(), c.cpu()).digest
        path = os.path.join(args.out_dir, f"patch_gmm_K{K}_{digest}.pt")
        torch.save({"weights": w.cpu(), "means": m.cpu(), "covs": c.cpu(),
                    "meta": {**vars(args), "K": K, "val_ll_per_dim": val_ll, "digest": digest}}, path)
        log({"saved": path, "K": K, "val_ll_per_dim": val_ll, "digest": digest})


if __name__ == "__main__":
    main()
