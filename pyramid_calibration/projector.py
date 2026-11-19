"""Match the projector to the patch weights actually used under native BF16 autocast."""
import shutil
import numpy as np
import torch
from safetensors import safe_open
from .common import *
def main():
    torch.set_num_threads(8)
    assert read(ROOT/'PREPARED.json')['complete']
    archive=STORE/'assets/nominal-fp32-projector';archive.mkdir(exist_ok=True)
    for name in ['encoder_rows.npy','encoder.json']:
     if not (archive/name).exists():shutil.copy2(STORE/'assets'/name,archive/name)
    with safe_open(str(STORE/'assets/model/model.safetensors'),framework='pt',device='cpu')as f:
     a0=f.get_tensor('patch_embed.proj1.weight').reshape(1024,3072)
     a=a0.to(torch.bfloat16).double().numpy()
     b=f.get_tensor('patch_embed.proj2.weight').reshape(2944,1024).to(torch.bfloat16).double().numpy()
    _,s,rows=np.linalg.svd(a,full_matrices=False)
    assert np.linalg.matrix_rank(b)==1024 and np.linalg.matrix_rank(a)==1024
    np.testing.assert_allclose(rows@rows.T,np.eye(1024),atol=1e-10,rtol=0)
    np.testing.assert_allclose(a-(a@rows.T)@rows,0,atol=1e-10,rtol=0)
    np.save(STORE/'assets/encoder_rows.npy',rows)
    nominal=np.load(archive/'encoder_rows.npy')
    leak=np.linalg.norm(a-(a@nominal.T)@nominal)/np.linalg.norm(a)
    save(STORE/'assets/encoder.json',dict(rank=1024,patch_dimension=3072,discarded_dimension=2048,
     projector_weight_precision='BF16-cast released FP32 weights, matching native autocast',
     nominal_projector_relative_leakage=float(leak),rows_sha256=digest(STORE/'assets/encoder_rows.npy'),
     input_quantization_note='Native BF16 input/output rounding means ideal Gaussian risk identities are checked empirically, not asserted bit-exact.'))
    save(ROOT/'PROJECTOR_READY.json',dict(complete=True,rows_sha256=digest(STORE/'assets/encoder_rows.npy'),nominal_leakage=float(leak)))
    released=np.load(MEASUREMENTS/'encoder_rows.npy')
    np.testing.assert_allclose(a-(a@released.T)@released,0,atol=1e-10,rtol=0)
    np.testing.assert_allclose(released@released.T,np.eye(1024),atol=1e-10,rtol=0)
    for name in ['encoder_rows.npy','encoder.json']:
        shutil.copy2(MEASUREMENTS/name,STORE/'assets'/name)
    save(ROOT/'PROJECTOR_READY.json',dict(complete=True,rows_sha256=digest(STORE/'assets/encoder_rows.npy'),nominal_leakage=float(leak),released_basis_verified=True))
    print(read(ROOT/'PROJECTOR_READY.json'))
