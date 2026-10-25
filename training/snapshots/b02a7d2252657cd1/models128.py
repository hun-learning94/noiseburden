"""Full-width, 12-block JiT models; retain the existing /8 head dimension of 64."""
from functools import partial
import model_jit


for width in (64, 128, 192, 256, 320, 384):
    model_jit.JiT_models[f"JiT-B{width}/8"] = partial(
        model_jit.JiT, depth=12, hidden_size=width, num_heads=width // 64,
        bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=8)


def validate(model, width):
    net = model.net if hasattr(model, "net") else model
    assert net.hidden_size == width and len(net.blocks) == 12
    assert net.num_heads == width // 64
    assert isinstance(net.x_embedder, model_jit.LinearPatchEmbed)
    assert not hasattr(net.x_embedder, "proj1") and not hasattr(net.x_embedder, "proj2")
    assert tuple(net.x_embedder.proj.weight.shape) == (width, 3, 8, 8)
    assert net.x_embedder.num_patches == 256
    assert tuple(net.final_layer.linear.weight.shape) == (192, width)
    for block in net.blocks:
        assert tuple(block.attn.qkv.weight.shape) == (3 * width, width)
        assert tuple(block.attn.proj.weight.shape) == (width, width)
        assert block.attn.q_norm.weight.numel() == 64
    return dict(width=width, head_dim=64, heads=width // 64, depth=12,
                full_attention=True, image_size=128, patch_size=8, raw_patch_dim=192,
                spatial_tokens=256, patch_embed="linear", extra_patch_bottleneck=False,
                embedding_reduces_dimension=width < 192,
                trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
