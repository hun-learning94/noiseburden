"""[train_resume.py: train.py plus exact checkpoint/resume for preemptible GPUs. Every --ckpt_every
steps the model, optimizer, sampling-generator state, step, and curve are saved atomically to
<out>.resume.pt; a restart with identical argv and code continues from it; the file is removed at
the end. Used for tier 2, whose runs are long; tier 1 keeps train.py so its results stay valid.]

Train one fixed-t noise predictor on a tractable patch distribution and evaluate it
against the exact population denoiser.

Objective: the paper's implemented image-space loss (eq. width-implemented-loss), which at a
fixed t in [tau, 1 - tau] equals the induced clean-image squared error,
    X1_hat = (X_t - (1 - t) eps_hat) / t,   L = mean ||X1_hat - X_1||^2 / d_p.
With --target dstar the regression target is the exact posterior mean E[X_1 | Y_t] instead of X_1.
Both objectives have the same minimizer over any model class (they differ by the constant Bayes
risk), so this only removes the gradient noise of the irreducible error; it is used for the
constrained theorem check, while the JiT-like arm keeps the X_1 target.
Fresh samples every step, so there is no train/test gap; the evaluation set is shared by all
runs with the same distribution and t (fixed eval seed).

Reported (all per patch, i.e. summed over the d_p coordinates):
    approx  = E||X1_hat - E[X_1|Y_t]||^2        approximation error in Thm. joint-width
    bayes   = E||X_1 - E[X_1|Y_t]||^2           irreducible risk
    total   = E||X1_hat - X_1||^2               = approx + bayes in population
    resid   = E||E[X_1|Y_t] - l_t(Y_t)||^2      = tr Cov(R_t), error of the affine baseline
    theory  = sum_{j > k} lambda_j(Cov R_t)      k = extra width (needs --spectrum); exact quadrature
                                                eigenvalues for the binary family, MC otherwise
    delta_in_resid_span = share of E||X1_hat - l_t||^2 inside the support of Cov(R_t)
Frozen-feature readout (held-out, PROBE_SEED fit and test sets, exact posterior targets):
    approx_refit = approximation error after refitting the affine noise head on frozen h by least
              squares against eps* = (X_t - t E[X_1|Y_t]) / (1 - t). The theorem takes the infimum
              over the head as well, so this is the quantity compared with `theory`; approx_head is
              the trained head on the same test samples.
    align_captured = output-direction alignment of the refit correction: the correction minus its
              affine dependence on Y, top-k eigenspace P (numerically null directions dropped),
              tr(P Cov(R_t)) / sum_{j<=|P|} lambda_j. Diagnostic only; approx_refit is primary.
    probe_r2_input / probe_r2_dstar: affine recovery of Y_t,m / E[X_1|Y_t] from the readout input h
Binary family only (its residual eigenvectors are the latent directions U): per-direction capture
    capture_j = 1 - E((Delta - R_t) . u_j)^2 / E(R_t . u_j)^2,  lambda_j = E(R_t . u_j)^2,
with err_in_span = sum_j E((Delta - R_t) . u_j)^2 and uncaptured_lambda = sum of lambda_j over
directions with capture_j < 0.5; the theorem's error structure is approx ~ sum of the residual
variance in directions the extra coordinates do not represent.

Only the constrained architecture belongs to the theorem's representation class, so `theory` is a
lower bound for it alone. The free (JiT-like) network may legitimately fall below it: for these
distributions eps* is affine in a d_p-dimensional feature, so width d_p suffices in principle.
The eval set is shared across runs because every run uses the same EVAL_SEED, eval_batch, and device.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch

from common import SCHEMA
from extra_hash import script_hash
from gmm import add_dist_args, build_dist, sigma_of_t
from models import build_model

EVAL_SEED = 20261108
PROBE_SEED = 20261109


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * prog))


def batch_xt(dist, t, n, gen, device):
    x1 = dist.sample(n, gen, dtype=torch.float64)
    eps = torch.randn(x1.shape, generator=gen, device=device, dtype=torch.float64)
    return x1, t * x1 + (1 - t) * eps


@torch.no_grad()
def evaluate(model, dist, t, n, batch, device, resid_eig=None, k=0):
    """resid_eig = (eigvals desc, eigvecs) of the MC Cov(R_t), or None."""
    sigma = sigma_of_t(t)
    gen = torch.Generator(device=device).manual_seed(EVAL_SEED)
    acc = dict(approx=0.0, bayes=0.0, total=0.0, resid=0.0, delta=0.0)
    d = dist.d
    s_d = torch.zeros(d, dtype=torch.float64, device=device)
    s_dd = torch.zeros(d, d, dtype=torch.float64, device=device)
    U = getattr(dist, "U", None) if isinstance(getattr(dist, "s", None), torch.Tensor) else None
    if U is not None:
        e_j = torch.zeros(U.shape[1], dtype=torch.float64, device=device)
        v_j = torch.zeros(U.shape[1], dtype=torch.float64, device=device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        x1, xt = batch_xt(dist, t, b, gen, device)
        y = xt / t
        dstar = dist.posterior_mean(y, sigma)
        lin = dist.affine(y, sigma)
        eps_hat = model(xt.float()).double()
        x1_hat = (xt - (1 - t) * eps_hat) / t
        acc["approx"] += float(((x1_hat - dstar) ** 2).sum())
        acc["bayes"] += float(((x1 - dstar) ** 2).sum())
        acc["total"] += float(((x1_hat - x1) ** 2).sum())
        acc["resid"] += float(((dstar - lin) ** 2).sum())
        delta = x1_hat - lin
        acc["delta"] += float((delta ** 2).sum())
        s_d += delta.sum(0)
        s_dd += delta.T @ delta
        if U is not None:
            pr = (dstar - lin) @ U
            e_j += (((delta @ U) - pr) ** 2).sum(0)
            v_j += (pr ** 2).sum(0)
        done += b
    out = {key: val / n for key, val in acc.items()}
    out["delta_in_resid_span"] = None
    if U is not None:
        cap = 1 - e_j / v_j.clamp_min(1e-300)
        out["capture_j"] = [round(float(c), 4) for c in cap]
        out["lambda_j"] = [float(v) / n for v in v_j]
        out["err_in_span"] = float(e_j.sum()) / n
        out["uncaptured_lambda"] = float(v_j[cap < 0.5].sum()) / n
        out["n_captured"] = int((cap >= 0.5).sum())
    if resid_eig is not None:
        lam, vecs = resid_eig
        support = vecs[:, lam > 1e-10 * float(lam[0])]
        out["delta_in_resid_span"] = float(torch.trace(support.T @ s_dd @ support)) / float(torch.trace(s_dd))
    return out


@torch.no_grad()
def collect(model, dist, t, n, batch, gen, device):
    """Readout input h, noisy patch y, exact E[X|y], affine baseline, trained-head eps_hat."""
    sigma = sigma_of_t(t)
    cols = {"h": [], "y": [], "dstar": [], "lin": [], "eps_head": []}
    done = 0
    while done < n:
        b = min(batch, n - done)
        _, xt = batch_xt(dist, t, b, gen, device)
        y = xt / t
        h = model.features(xt.float())
        cols["h"].append(h.double())
        cols["eps_head"].append(model.out(h).double())
        cols["y"].append(y)
        cols["dstar"].append(dist.posterior_mean(y, sigma))
        cols["lin"].append(dist.affine(y, sigma))
        done += b
    return {key: torch.cat(val) for key, val in cols.items()}


def affine_fit(x_fit, tgt_fit, rel_ridge=1e-8):
    """Least-squares affine map x -> tgt; returns a predictor."""
    mu_x, mu_t = x_fit.mean(0), tgt_fit.mean(0)
    xc = x_fit - mu_x
    gram = xc.T @ xc / xc.shape[0]
    ridge = rel_ridge * float(torch.diagonal(gram).mean()) + 1e-300
    w = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
                           xc.T @ (tgt_fit - mu_t) / xc.shape[0])
    return lambda x: (x - mu_x) @ w + mu_t


def affine_r2(h_fit, tgt_fit, h_test, tgt_test):
    pred = affine_fit(h_fit, tgt_fit)(h_test)
    sse = float(((tgt_test - pred) ** 2).sum())
    sst = float(((tgt_test - tgt_test.mean(0)) ** 2).sum())
    return 100.0 * (1 - sse / sst)


def alignment(delta, y, k, resid_eig):
    """Top-k output directions of the nonlinear part of a correction vs the residual spectrum."""
    if resid_eig is None or k <= 0:
        return None
    lam, vecs = resid_eig
    nl = delta - affine_fit(y, delta)(y)
    if float((nl - nl.mean(0)).pow(2).sum()) <= 1e-10 * float((delta - delta.mean(0)).pow(2).sum()):
        return None  # purely affine correction: no nonlinear output span
    ev, evec = torch.linalg.eigh(torch.cov(nl.T))
    order = torch.argsort(ev, descending=True)
    keep = [i for i in order[:k].tolist() if ev[i] > 1e-8 * float(ev[order[0]])]
    if not keep or float(ev[order[0]]) <= 0:
        return None
    p = evec[:, keep]
    cov_r = (vecs * lam) @ vecs.T
    return {"value": float(torch.trace(p.T @ cov_r @ p)) / float(lam[:len(keep)].sum()), "rank": len(keep)}


def _align_fields(a):
    return {"align_captured": None if a is None else a["value"], "align_rank": None if a is None else a["rank"]}


@torch.no_grad()
def frozen_readout(model, dist, t, n_fit, n_test, batch, device, k, resid_eig):
    gen = torch.Generator(device=device).manual_seed(PROBE_SEED)
    fit = collect(model, dist, t, n_fit, batch, gen, device)
    test = collect(model, dist, t, n_test, batch, gen, device)
    eps_star = lambda c: (t * c["y"] - t * c["dstar"]) / (1 - t)  # noqa: E731  (X_t = t y)
    head = affine_fit(fit["h"], eps_star(fit))
    x1_refit = test["y"] - (1 - t) / t * head(test["h"])
    x1_head = test["y"] - (1 - t) / t * test["eps_head"]
    return {
        "approx_refit": float(((x1_refit - test["dstar"]) ** 2).sum(1).mean()),
        "approx_head": float(((x1_head - test["dstar"]) ** 2).sum(1).mean()),
        **_align_fields(alignment(x1_refit - test["lin"], test["y"], k, resid_eig)),
        "probe_r2_input": affine_r2(fit["h"], fit["y"], test["h"], test["y"]),
        "probe_r2_dstar": affine_r2(fit["h"], fit["dstar"], test["h"], test["dstar"]),
    }


def main():
    p = argparse.ArgumentParser()
    add_dist_args(p)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--arch", choices=["free", "constrained"], required=True)
    p.add_argument("--width", type=int, default=None, help="free: model width d_h")
    p.add_argument("--extra", type=int, default=None, help="constrained: k = d_h - d_p")
    p.add_argument("--g_width", type=int, default=256)
    p.add_argument("--g_final_norm", choices=["rms", "none"], default="rms", help="constrained g only")
    p.add_argument("--readout_init", type=float, default=0.0, help="constrained: std of the g-part readout init")
    p.add_argument("--target", choices=["x1", "dstar"], default="x1")
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--affine_init", action="store_true",
                   help="constrained: initialize the X_t readout at the best affine noise head")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--final_norm", choices=["rms", "none"], default="rms", help="free net only")
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--bs", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_n", type=int, default=262144)
    p.add_argument("--probe_n", type=int, default=65536)
    p.add_argument("--eval_batch", type=int, default=16384)
    p.add_argument("--curve_points", type=int, default=4)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--spectrum", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_code_hash = script_hash("train_resume.py")  # the code this run actually executes
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = args.device
    dist = build_dist(args).to(device)
    d_p = dist.d
    model = build_model(args, d_p).to(device)
    k = (args.width - d_p) if args.arch == "free" else args.extra
    n_params = sum(q.numel() for q in model.parameters())

    resid_eig = theory_eigs = None
    if args.spectrum:
        spec = torch.load(args.spectrum, map_location="cpu")
        meta = spec["meta"]
        assert abs(meta["t"] - args.t) < 1e-12, (meta["t"], args.t)
        assert meta.get("digest") == dist.digest, ("spectrum was computed for another distribution",
                                                   meta.get("digest"), dist.digest)
        keys = {"binary": ("dist", "d", "scales", "sigma0", "dist_seed"), "gmm": ("dist", "gmm_path"),
                "shared": ("dist", "d", "M", "scales", "shared_scales", "sigma0", "dist_seed")}[args.dist]
        for key in keys:
            assert meta[key] == getattr(args, key), (key, meta[key], getattr(args, key))
        resid_eig = (spec["eigvals"].double().to(device), spec["eigvecs"].double().to(device))
        if "quad_eigvals" in spec:  # exact for the binary family; zero outside span(U)
            theory_eigs = torch.zeros(d_p, dtype=torch.float64)
            theory_eigs[:spec["quad_eigvals"].numel()] = spec["quad_eigvals"].double()
        else:
            theory_eigs = spec["eigvals"].double()

    t = args.t
    if args.affine_init:  # eps = (I - S)(X_t - t mu) / (1 - t) realizes l_t; stays in the class
        assert args.arch == "constrained"
        S = dist.affine_matrix(sigma_of_t(t))
        A = (torch.eye(d_p, dtype=torch.float64, device=device) - S) / (1 - t)
        with torch.no_grad():
            model.out.weight[:, :d_p] = A.float()
            model.out.bias.copy_((-t * (A @ dist.mean)).float())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=args.adam_eps)
    gen = torch.Generator(device=device).manual_seed(1000 + args.seed)
    curve_at = {round(args.steps * (i + 1) / args.curve_points) for i in range(args.curve_points)}
    curve, running, t0 = [], 0.0, time.time()
    resume_path, start_step, sec_before = args.out[:-5] + ".resume.pt", 0, 0.0
    for stale_tmp in glob.glob(resume_path + ".*.tmp"):  # left by a kill during torch.save
        os.remove(stale_tmp)
    blob = None
    if os.path.exists(resume_path):
        try:  # load on CPU: load_state_dict moves tensors; keeps AdamW step counters off the GPU
            blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"unreadable resume file ignored: {exc}", flush=True)
    if blob is not None:
        if blob["argv"] == sys.argv[1:] and blob["code_hash"] == run_code_hash:
            model.load_state_dict(blob["model"])
            opt.load_state_dict(blob["opt"])
            gen.set_state(blob["gen_state"].cpu())
            start_step, curve, running, sec_before = blob["step"], blob["curve"], blob["running"], blob["train_sec"]
            print(f"resumed from step {start_step}", flush=True)
        else:
            print("ignoring a resume file from another configuration or code", flush=True)
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x1, xt = batch_xt(dist, t, args.bs, gen, device)
        if args.target == "dstar":
            x1 = dist.posterior_mean(xt / t, sigma_of_t(t))
        x1, xt = x1.float(), xt.float()
        x1_hat = (xt - (1 - t) * model(xt)) / t
        loss = ((x1_hat - x1) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.detach()
        if (step + 1) in curve_at:
            model.eval()
            ev = evaluate(model, dist, t, 65536, args.eval_batch, device)
            model.train()
            curve.append({"step": step + 1, "train_loss_avg": float(running) / (args.steps / args.curve_points),
                          "approx_65k": ev["approx"]})
            running = 0.0
        if (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
            tmp = f"{resume_path}.{os.getpid()}.tmp"
            torch.save({"argv": sys.argv[1:], "code_hash": run_code_hash, "model": model.state_dict(),
                        "opt": opt.state_dict(), "gen_state": gen.get_state(), "step": step + 1, "curve": curve,
                        "running": float(running), "train_sec": sec_before + time.time() - t0}, tmp)
            os.replace(tmp, resume_path)
            if os.environ.get("WSYNTH_STOP_AFTER") and step + 1 >= int(os.environ["WSYNTH_STOP_AFTER"]):
                sys.exit(3)  # test hook: simulate preemption right after a checkpoint
    train_sec = sec_before + time.time() - t0

    model.eval()
    ev = evaluate(model, dist, t, args.eval_n, args.eval_batch, device, resid_eig, k)
    pr = frozen_readout(model, dist, t, args.probe_n, args.probe_n, args.eval_batch, device, k, resid_eig)
    res = {"schema": SCHEMA, "argv": sys.argv[1:], "code_hash": run_code_hash, "digest": dist.digest,
           "config": vars(args), "d_p": d_p,
           "k": k, "n_params": n_params, "train_sec": train_sec, **ev, **pr, "curve": curve}
    if theory_eigs is not None:
        kk = max(0, min(k, theory_eigs.numel()))
        res["theory"] = float(theory_eigs[kk:].sum())
        res["theory_resid"] = float(theory_eigs.sum())
        res["theory_mc"] = float(resid_eig[0][kk:].sum())
        res["spectrum_argv"] = meta.get("argv")
        res["spectrum_code_hash"] = meta.get("code_hash")
        if args.arch == "constrained" and res["approx_refit"] < 0.98 * res["theory"] - 1e-6:
            print(f"WARNING: constrained approx_refit {res['approx_refit']:.5g} is >2% below the estimated bound "
                  f"{res['theory']:.5g}; check sampling error (repeat / bootstrap) and the spectrum match", flush=True)
            res["below_bound_warning"] = True
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out[:-5] + ".pt" if args.out.endswith(".json") else args.out + ".pt")
    tmp = f"{args.out}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, args.out)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(json.dumps({key: res[key] for key in ("k", "n_params", "approx", "approx_head", "approx_refit",
                                                 "bayes", "resid", "align_captured", "probe_r2_input",
                                                 "probe_r2_dstar", "train_sec")}
                     | ({"theory": res["theory"]} if "theory" in res else {})))


if __name__ == "__main__":
    main()
