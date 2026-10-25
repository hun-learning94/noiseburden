"""Prepare validation arrays from user-supplied official images; no image download."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
import torch
from PIL import Image
from .runtime import ROOT,SPEC,digest,save

POSITIONS=[(7,7),(3,3),(3,12),(12,3),(12,12)]
def patches(x,size):
    p=size//16
    return np.stack([x[:,:,r*p:(r+1)*p,c*p:(c+1)*p].transpose(0,2,3,1).reshape(len(x),3*p*p) for r,c in POSITIONS],1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--images',type=Path)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--input-noise',action='store_true')
    p.add_argument('--noise-only',action='store_true');a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not a.noise_only:
        if not a.images:p.error('--images must contain the official ILSVRC2012_val_*.JPEG files')
        spec=importlib.util.spec_from_file_location('frozen_crop',ROOT/'training/base/util/crop.py')
        crop=importlib.util.module_from_spec(spec);spec.loader.exec_module(crop)
        samples=json.loads((ROOT/'configs/validation_samples.json').read_text())
        labels=np.array([r['label'] for r in samples],dtype=np.int64);np.save(a.cache/'labels.npy',labels)
        save(a.cache/'samples.json',samples)
        for lo,hi,n in [(0,6000,6),(6000,10000,4),(0,8000,8),(8000,10000,2)]:
            assert np.array_equal(np.bincount(labels[lo:hi],minlength=1000),np.full(1000,n))
        arrays={r:np.lib.format.open_memmap(a.cache/f'images{r}.npy',mode='w+',dtype=np.uint8,shape=(10000,3,r,r)) for r in [128,256]}
        for i,row in enumerate(samples):
            with Image.open(a.images/row['filename']) as im:
                im=im.convert('RGB')
                for r,array in arrays.items():array[i]=np.asarray(crop.center_crop_arr(im,r)).transpose(2,0,1)
            if i%1000==0:print('Cropped',i,'/ 10000',flush=True)
        for array in arrays.values():array.flush()
        ids=np.r_[np.random.default_rng(251025).permutation(8000)[:512],np.arange(8000,10000)]
        np.save(a.cache/'input-probe-indices.npy',ids)
        for res in [128,256]:
            clean=np.lib.format.open_memmap(a.cache/f'spectral-clean{res}.npy',mode='w+',dtype=np.float32,shape=(10000,5,3*(res//16)**2))
            for lo in range(0,10000,100):clean[lo:lo+100]=patches(np.asarray(arrays[res][lo:lo+100]),res).astype(np.float32)/127.5-1
            clean.flush()
            noise=np.lib.format.open_memmap(a.cache/f'spectral-noise{res}.npy',mode='w+',dtype=np.float32,shape=(4000,5,3*(res//16)**2))
            for lo in range(6000,10000,16):
                hi=min(lo+16,10000)
                values=torch.stack([torch.randn((3,res,res),generator=torch.Generator().manual_seed(SPEC['noise_seed']+i)) for i in range(lo,hi)])
                noise[lo-6000:hi-6000]=patches(values.numpy(),res)
            noise.flush()
    if a.input_noise:
        if not torch.cuda.is_available():raise RuntimeError('The historical first 640 input-probe fields require CUDA; prepare images on CPU first, then use --noise-only --input-noise on a GPU.')
        # Preserve the original single draw shape and CUDA reduction order.
        full=torch.randn((640,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(99003))
        for res in [256,128]:
            values=full if res==256 else full.reshape(640,3,128,2,128,2).sum((3,5))/2
            first=values[0].cpu().numpy()
            assert hashlib.sha256(first.tobytes()).hexdigest()==SPEC['input_noise_legacy'][str(res)]['first_image_sha256']
            target=np.lib.format.open_memmap(a.cache/f'input-noise{res}.npy',mode='w+',dtype=np.float32,shape=(2512,3,res,res));target[:640]=values.cpu().numpy()
            for lo in range(640,2512,16):
                hi=min(lo+16,2512);target[lo:hi]=torch.stack([torch.randn((3,res,res),generator=torch.Generator().manual_seed(99003+i)) for i in range(lo,hi)]).numpy()
            target.flush()
    hashes={f.name:digest(f) for f in a.cache.iterdir() if f.is_file() and f.name!='complete.json'}
    save(a.cache/'complete.json',hashes)
    print('Prepared cache:',a.cache.resolve())

if __name__=='__main__':main()
