"""Noise predictors for one patch at a fixed time t.

FreeNet mirrors the JiT token path (model_jit.py) with a single token: learned linear patch
embedding d_p -> d_h, residual blocks (RMSNorm, modulation, SwiGLU FFN, gate), and the final
layer (RMSNorm, modulation, linear). At fixed t without class labels the adaLN modulation
vectors are constants, so they are plain parameters with JiT's zero initialization.
Attention is omitted: with one token it reduces to a linear map of the normalized stream.
The readout input h = modulate(RMSNorm(x)) is the probed feature, as in App. width-readouts.

ConstrainedNet realizes the representation class of Assumption patch-recovery exactly:
h = [X_t ; g(X_t)] with g(X_t) in R^k, k = d_h - d_p, followed by one affine readout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):  # util/model_util.py
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SwiGLUFFN(nn.Module):  # model_jit.py
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio))
        self.mod = nn.Parameter(torch.zeros(3, dim))  # shift, scale, gate (adaLN-zero)

    def forward(self, x):
        shift, scale, gate = self.mod
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class Trunk(nn.Module):
    """Embedding, blocks, final norm and modulation; returns the readout input."""

    def __init__(self, d_in, width, depth, mlp_ratio, final_norm="rms"):
        super().__init__()
        self.embed = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([Block(width, mlp_ratio) for _ in range(depth)])
        # JiT applies RMSNorm before the readout, which puts h on a scaled sphere and makes exact
        # affine recovery of an unbounded input impossible; "none" removes it for the ablation.
        self.norm = RMSNorm(width) if final_norm == "rms" else nn.Identity()
        self.mod = nn.Parameter(torch.zeros(2, width))  # shift, scale
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        shift, scale = self.mod
        return self.norm(x) * (1 + scale) + shift


class FreeNet(nn.Module):
    def __init__(self, d_p, width, depth=6, mlp_ratio=4.0, final_norm="rms"):
        super().__init__()
        self.trunk = Trunk(d_p, width, depth, mlp_ratio, final_norm)
        self.out = nn.Linear(width, d_p)
        nn.init.zeros_(self.out.weight)  # JiT zero-initializes the output layer
        nn.init.zeros_(self.out.bias)

    def features(self, x_t):
        return self.trunk(x_t)

    def forward(self, x_t):
        return self.out(self.features(x_t))


class ConstrainedNet(nn.Module):
    def __init__(self, d_p, extra, g_width=256, depth=6, mlp_ratio=4.0, g_final_norm="rms", readout_init=0.0):
        super().__init__()
        self.extra = extra
        if extra > 0:
            self.g = Trunk(d_p, g_width, depth, mlp_ratio, g_final_norm)
            self.g_out = nn.Linear(g_width, extra)  # not zero-initialized: readout starts at zero
            nn.init.xavier_uniform_(self.g_out.weight)
            nn.init.zeros_(self.g_out.bias)
        self.out = nn.Linear(d_p + extra, d_p)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if extra > 0 and readout_init > 0:  # small random readout of g breaks the zero-zero start
            nn.init.normal_(self.out.weight[:, d_p:], std=readout_init)

    def features(self, x_t):
        if self.extra == 0:
            return x_t
        return torch.cat([x_t, self.g_out(self.g(x_t))], dim=-1)

    def forward(self, x_t):
        return self.out(self.features(x_t))


def build_model(args, d_p):
    if args.arch == "free":
        return FreeNet(d_p, args.width, args.depth, args.mlp_ratio, args.final_norm)
    if args.arch == "constrained":
        return ConstrainedNet(d_p, args.extra, args.g_width, args.depth, args.mlp_ratio,
                              getattr(args, "g_final_norm", "rms"), getattr(args, "readout_init", 0.0))
    raise ValueError(args.arch)
