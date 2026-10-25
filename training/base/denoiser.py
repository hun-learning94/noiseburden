import torch
import torch.nn as nn
from model_jit import JiT_models


class TimeCoeff(nn.Module):
    """Learns time-dependent shortcut coefficients: out = a(t) * z + b(t) * F.

    Zero-init of the last layer gives a(t) = 0, b(t) = 1 at init, i.e. exactly
    the plain eps-pred parameterization. If training discovers the analytic
    preconditioning, a(t) -> 1/(1-t) and b(t) -> -t/(1-t) (up to what F absorbs).
    """
    def __init__(self, n_freq=8, hidden=64, a0=0.0, precond_base=False, t_eps=0.05):
        super().__init__()
        self.n_freq = n_freq
        self.a0 = a0
        self.precond_base = precond_base   # add analytic 1/(1-t), -t/(1-t) as a fixed baseline
        self.t_eps = t_eps
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        tf = t.flatten()[:, None]
        k = torch.arange(1, self.n_freq + 1, device=t.device, dtype=t.dtype)
        feat = torch.cat([torch.sin(2 * torch.pi * tf * k),
                          torch.cos(2 * torch.pi * tf * k)], dim=1)
        out = self.mlp(feat)
        a = self.a0 + out[:, 0].view(-1, 1, 1, 1)
        b = 1.0 + out[:, 1].view(-1, 1, 1, 1)
        if self.precond_base:
            tt = t.view(-1, 1, 1, 1)
            denom = (1 - tt).clamp_min(self.t_eps)
            a = a - self.a0 + 1.0 / denom          # remove a0, use analytic baseline
            b = b - 1.0 - tt / denom               # remove +1, use analytic baseline
        return a, b


class LearnedSchedule(nn.Module):
    """Piecewise-linear a(t), b(t) with K nodes on t in [0,1] for z = a(t) x + b(t) eps.

    Endpoints are pinned to (a,b)(0) = (0,1) and (a,b)(1) = (1,0) so sampling still
    starts from pure noise and ends at data. Interior node values are learnable
    (softplus-positive) and initialized to the linear schedule a=t, b=1-t.
    Derivatives a'(t), b'(t) are piecewise constant.
    """
    def __init__(self, n_nodes=10):
        super().__init__()
        ts = torch.linspace(0.0, 1.0, n_nodes)
        self.register_buffer('ts', ts)
        inv = lambda y: y + torch.log(-torch.expm1(-y))  # softplus^-1
        self.raw_a = nn.Parameter(inv(ts[1:-1].clone()))
        self.raw_b = nn.Parameter(inv((1.0 - ts[1:-1]).clone()))

    def nodes(self):
        a = torch.cat([self.ts.new_zeros(1), torch.nn.functional.softplus(self.raw_a), self.ts.new_ones(1)])
        b = torch.cat([self.ts.new_ones(1), torch.nn.functional.softplus(self.raw_b), self.ts.new_zeros(1)])
        return a, b

    def coeffs(self, t):
        """t: (B,1,1,1) -> a, b, a', b' with the same shape."""
        shape = t.shape
        tf = t.flatten().clamp(0.0, 1.0)
        a_n, b_n = self.nodes()
        idx = (torch.searchsorted(self.ts, tf, right=True) - 1).clamp(0, self.ts.numel() - 2)
        t0, t1 = self.ts[idx], self.ts[idx + 1]
        w = (tf - t0) / (t1 - t0)
        a = a_n[idx] * (1 - w) + a_n[idx + 1] * w
        b = b_n[idx] * (1 - w) + b_n[idx + 1] * w
        ap = (a_n[idx + 1] - a_n[idx]) / (t1 - t0)
        bp = (b_n[idx + 1] - b_n[idx]) / (t1 - t0)
        return a.view(shape), b.view(shape), ap.view(shape), bp.view(shape)


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            z_inject=(getattr(args, 'z_inject', 'none') == 'all'),
            z_scale=(getattr(args, 'z_scale', 'off') == 'on'),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.pred_type = getattr(args, 'pred_type', 'x')
        self.loss_type = getattr(args, 'loss_type', 'v')
        self.shortcut = getattr(args, 'shortcut', 'none')
        if self.shortcut in ('learned', 'learned_add', 'learned_precond'):
            # zero-init head => residual starts at 0, so the coefficients start at:
            #   learned         : (a,b) = (0, 1)              plain eps-pred
            #   learned_add     : (a,b) = (1, 1)              the add shortcut
            #   learned_precond : (a,b) = (1/(1-t), -t/(1-t)) analytic preconditioning
            self.time_coeff = TimeCoeff(
                a0=1.0 if self.shortcut == 'learned_add' else 0.0,
                precond_base=(self.shortcut == 'learned_precond'),
                t_eps=self.t_eps,
            )
        else:
            self.time_coeff = None
        self.schedule_type = getattr(args, 'schedule', 'linear')
        if self.schedule_type == 'learned':
            assert self.shortcut == 'none', 'learned schedule currently requires --shortcut none'
            assert self.loss_type in ('x', 'eps') and self.pred_type in ('x', 'eps'), \
                'learned schedule supports pred/loss in {x, eps} only'
            self.schedule = LearnedSchedule(getattr(args, 'schedule_nodes', 10))
        else:
            self.schedule = None

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def out_to_v(self, out, z, t):
        # unify the network's direct output into v-space (Tab. 1 of the JiT paper)
        if self.pred_type == 'x':
            return (out - z) / (1 - t).clamp_min(self.t_eps)
        elif self.pred_type == 'eps':
            return (z - out) / t.clamp_min(self.t_eps)
        elif self.pred_type == 'v':
            return out
        else:
            raise NotImplementedError

    def apply_shortcut(self, raw, z, t):
        # optional input shortcut on the network's direct output
        if self.shortcut == 'none':
            return raw
        elif self.shortcut == 'add':
            # out = z + F : the network predicts a correction to the input
            return z + raw
        elif self.shortcut == 'precond':
            # out = (z - t*F) / (1-t) : exact eps-parameterization of an x-predictor
            # (with pred_type='eps' this makes F play the role of x_hat)
            return (z - t * raw) / (1 - t).clamp_min(self.t_eps)
        elif self.shortcut in ('learned', 'learned_add', 'learned_precond'):
            # out = a(t) * z + b(t) * F with learned coefficients (see __init__ for inits)
            a, b = self.time_coeff(t)
            return a * z + b * raw
        else:
            raise NotImplementedError

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        if self.schedule is not None:
            a, b, _, _ = self.schedule.coeffs(t)
            z = a * x + b * e
            out = self.net(z, t.flatten(), labels_dropped)
            if self.pred_type == 'eps':
                e_hat = out
                x_hat = (z - b * e_hat) / a.clamp_min(self.t_eps)
            else:  # 'x'
                x_hat = out
                e_hat = (z - a * x_hat) / b.clamp_min(self.t_eps)
            if self.loss_type == 'x':
                loss = (x - x_hat) ** 2
            else:  # 'eps'
                loss = (e - e_hat) ** 2
            return loss.mean(dim=(1, 2, 3)).mean()

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        raw = self.net(z, t.flatten(), labels_dropped)
        out = self.apply_shortcut(raw, z, t)
        v_pred = self.out_to_v(out, z, t)

        # l2 loss in the chosen loss space, written as a reweighted v-space loss:
        # ||x - x_pred||^2 = (1-t)^2 ||v - v_pred||^2 ; ||e - e_pred||^2 = t^2 ||v - v_pred||^2
        if self.loss_type == 'v':
            w = torch.ones_like(t)
        elif self.loss_type == 'x':
            w = (1 - t) ** 2
        elif self.loss_type == 'eps':
            w = t ** 2
        else:
            raise NotImplementedError

        loss = w * (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        if self.schedule is not None:
            a, b, ap, bp = self.schedule.coeffs(t)

            def _to_v(out):
                if self.pred_type == 'eps':
                    e_hat = out
                    x_hat = (z - b * e_hat) / a.clamp_min(self.t_eps)
                else:
                    x_hat = out
                    e_hat = (z - a * x_hat) / b.clamp_min(self.t_eps)
                return ap * x_hat + bp * e_hat

            v_cond = _to_v(self.net(z, t.flatten(), labels))
            v_uncond = _to_v(self.net(z, t.flatten(), torch.full_like(labels, self.num_classes)))
        else:
            # conditional
            out_cond = self.apply_shortcut(self.net(z, t.flatten(), labels), z, t)
            v_cond = self.out_to_v(out_cond, z, t)

            # unconditional
            out_uncond = self.apply_shortcut(self.net(z, t.flatten(), torch.full_like(labels, self.num_classes)), z, t)
            v_uncond = self.out_to_v(out_uncond, z, t)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)