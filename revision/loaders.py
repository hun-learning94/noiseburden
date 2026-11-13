"""Load exact model shapes and EMA1 weights for portable revision evaluations."""
import copy,json,sys
from functools import partial
from pathlib import Path
from evaluation.runtime import ROOT,digest

def load(folder,device='cuda'):
    import torch
    c=json.loads((Path(folder)/'case.json').read_text())
    assert digest(c['checkpoint'])==c['checkpoint_sha256']
    for name,value in c['source_hashes'].items():assert digest(Path(c['source'])/name)==value
    sys.path.insert(0,str(ROOT/'evaluation/third_party'));sys.path.insert(0,c['source'])
    import model_jit
    from denoiser import Denoiser
    ck=torch.load(c['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert ck['epoch']+1==c['completed_epochs']
    args=copy.deepcopy(ck['args'])
    if c['kind']=='width':
        model_jit.JiT_models[args.model]=partial(model_jit.JiT,depth=c['depth'],hidden_size=c['width'],num_heads=c['heads'],
            patch_size=c['patch'],bottleneck_dim=128,in_context_len=32,in_context_start=4)
    model=Denoiser(args);model.load_state_dict(ck['model_ema1'],strict=True);del ck
    model=model.to(device).eval().requires_grad_(False)
    if c.get('expected_parameters'):assert sum(p.numel() for p in model.parameters())-model.net.pos_embed.numel()==c['expected_parameters']
    return model,args,c
