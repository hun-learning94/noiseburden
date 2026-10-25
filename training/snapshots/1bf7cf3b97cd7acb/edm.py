"""Common EDM output wrapper; only the input scale differs between variants."""
import torch
from denoiser import Denoiser

VARIANTS = ('jit_input', 'reflow_input')
EPOCHS = (1, 41, 81, 121, 161, 200)


class EDMDenoiser(Denoiser):
    def __init__(self, args):
        super().__init__(args)
        assert args.edm_variant in VARIANTS
        assert self.pred_type == self.loss_type == 'x'
        assert self.shortcut == 'none' and self.schedule is None
        self.edm_variant = args.edm_variant
        self.sigma_data = args.sigma_data

    def image_estimate(self, z, t, labels):
        # Xt coordinates: finite even at t=0; coefficients computed in FP32.
        t = t.float()
        denominator = self.sigma_data**2 * t.square() + (1-t).square()
        network_input = z if self.edm_variant == 'jit_input' else z / denominator.sqrt()
        raw = self.net(network_input, t.flatten(), labels)
        return self.sigma_data**2 * t / denominator * z + self.sigma_data * (1-t) / denominator.sqrt() * raw

    def forward(self, x, labels):
        labels = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.shape[0], x.device).view(-1, 1, 1, 1)
        z = t*x + (1-t)*torch.randn_like(x)*self.noise_scale
        image = self.image_estimate(z, t, labels)
        # Preserve Pengxi's weighted-v implementation, including its endpoint clamp.
        velocity = (x-z)/(1-t).clamp_min(self.t_eps)
        predicted = self.out_to_v(image, z, t)
        return ((1-t).square() * (velocity-predicted).square()).mean()

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        conditional = self.out_to_v(self.image_estimate(z, t, labels), z, t)
        unconditional = self.out_to_v(self.image_estimate(z, t, torch.full_like(labels, self.num_classes)), z, t)
        low, high = self.cfg_interval
        mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(mask, self.cfg_scale, 1.)
        return unconditional + scale*(conditional-unconditional)
