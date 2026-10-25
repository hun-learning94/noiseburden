import torch
import torch.nn.functional as F
import model_jit
_NATIVE_ATTN = model_jit.scaled_dot_product_attention

def configure(attention='native', tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if attention == 'sdpa':
        def attention_fn(query,key,value,dropout_p=0.):
            # Native RoPE promotes Q/K to FP32; FlashAttention needs a uniform dtype.
            return F.scaled_dot_product_attention(query.to(value.dtype),key.to(value.dtype),value,dropout_p=dropout_p)
        model_jit.scaled_dot_product_attention = attention_fn
    else:
        model_jit.scaled_dot_product_attention = _NATIVE_ATTN

@torch.no_grad()
def update_ema_foreach(model):
    params=list(model.parameters())
    for ema,decay in ((model.ema_params1,model.ema_decay1),(model.ema_params2,model.ema_decay2)):
        torch._foreach_mul_(ema,decay)
        torch._foreach_add_(ema,params,alpha=1-decay)
