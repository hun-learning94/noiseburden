import torch
import torch.nn as nn
from model_jit import JiT_models


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
            patch_embed=getattr(args, 'patch_embed', 'bottleneck'),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # Table-2 parameterizations: network output interpreted in pred_space,
        # loss computed in loss_space. Defaults reproduce the release (x-pred, v-loss).
        self.pred_space = getattr(args, 'pred_space', 'x')
        self.loss_space = getattr(args, 'loss_space', 'v')
        # Keep this clone's existing conversions by default. weighted_v matches
        # the pl229 experiments, including their t_eps endpoint handling.
        self.loss_impl = getattr(args, 'loss_impl', 'direct')
        assert self.pred_space in ('x', 'eps', 'v')
        assert self.loss_space in ('x', 'eps', 'v')
        assert self.loss_impl in ('direct', 'weighted_v')

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

    def _out_to_x(self, out, z, t):
        # convert network output (interpreted in self.pred_space) to x-hat
        if self.pred_space == 'x':
            return out
        elif self.pred_space == 'v':
            return z + (1 - t) * out
        elif self.pred_space == 'eps':
            return (z - (1 - t) * out) / t.clamp_min(self.t_eps)

    def _out_to_v(self, out, z, t):
        if self.loss_impl == 'direct':
            return (self._out_to_x(out, z, t) - z) / (1 - t).clamp_min(self.t_eps)
        if self.pred_space == 'x':
            return (out - z) / (1 - t).clamp_min(self.t_eps)
        if self.pred_space == 'eps':
            return (z - out) / t.clamp_min(self.t_eps)
        return out

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e

        out = self.net(z, t.flatten(), labels_dropped)
        if self.loss_impl == 'weighted_v':
            v = (x - z) / (1 - t).clamp_min(self.t_eps)
            v_pred = self._out_to_v(out, z, t)
            if self.loss_space == 'x':
                weight = (1 - t) ** 2
            elif self.loss_space == 'eps':
                weight = t ** 2
            else:
                weight = torch.ones_like(t)
            return (weight * (v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

        x_pred = self._out_to_x(out, z, t)

        # l2 loss in loss_space
        if self.loss_space == 'x':
            loss = (x - x_pred) ** 2
        elif self.loss_space == 'v':
            v = (x - z) / (1 - t).clamp_min(self.t_eps)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
            loss = (v - v_pred) ** 2
        elif self.loss_space == 'eps':
            e_pred = (z - t * x_pred) / (1 - t).clamp_min(self.t_eps)
            loss = (e - e_pred) ** 2
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
        # conditional
        v_cond = self._out_to_v(self.net(z, t.flatten(), labels), z, t)

        # unconditional
        v_uncond = self._out_to_v(
            self.net(z, t.flatten(), torch.full_like(labels, self.num_classes)), z, t
        )

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
