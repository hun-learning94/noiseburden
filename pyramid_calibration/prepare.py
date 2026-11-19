"""CPU-only download, deterministic COCO cohorts, pixel banks, and encoder SVD."""
import concurrent.futures
import fcntl
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from .common import *

def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.part')
    subprocess.run(['curl','--fail','--location','--retry','8','--retry-all-errors',
                    '--connect-timeout','30','--continue-at','-', '--output',str(partial),url], check=True)
    partial.replace(dest)
    return dest

def model_assets():
    from huggingface_hub import snapshot_download
    records = {}
    for name, repo, revision, patterns in [
        ('model','Linum-AI/pyramid-jit',MODEL_REV,['config.json','export_manifest.json','model.safetensors','README.md']),
        ('text','Qwen/Qwen3.5-4B',TEXT_REV,['*.json','*.jinja','*.txt','*.safetensors','LICENSE'])]:
        folder = STORE/'assets'/name
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(folder),
                          allow_patterns=patterns, max_workers=4)
        records[name] = {'repo':repo,'revision':revision,'sha256':{
            str(p.relative_to(folder)):digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        save(STORE/'assets'/f'{name}-receipt.json', records[name])
    from safetensors import safe_open
    import torch
    torch.set_num_threads(8)
    with safe_open(str(STORE/'assets/model/model.safetensors'),framework='pt',device='cpu') as f:
        a = f.get_tensor('patch_embed.proj1.weight').double().reshape(1024,3072).numpy()
        b = f.get_tensor('patch_embed.proj2.weight').double().reshape(2944,1024).numpy()
    _, s, rows = np.linalg.svd(a,full_matrices=False)
    rank = int((s > s[0]*max(a.shape)*np.finfo(np.float64).eps).sum())
    assert rank == 1024 and np.linalg.matrix_rank(b) == 1024, 'Unexpected embedding rank'
    np.testing.assert_allclose(rows@rows.T,np.eye(rank),atol=1e-10,rtol=0)
    residual = np.linalg.norm(a-(a@rows.T)@rows)/np.linalg.norm(a)
    assert residual < 1e-10
    np.save(STORE/'assets/encoder_rows.npy',rows)
    save(STORE/'assets/encoder.json', {'rank':rank,'patch_dimension':3072,'discarded_dimension':2048,
        'row_projection_relative_error':float(residual),'smallest_singular_value':float(s[-1]),
        'rows_sha256':digest(STORE/'assets/encoder_rows.npy')})
    return records

def crop(im):
    """ADM/JiT deterministic square center crop, explicitly also used for FID real images."""
    im = im.convert('RGB')
    while min(im.size) >= 1024:
        im = im.resize(tuple(x//2 for x in im.size),resample=Image.Resampling.BOX)
    scale = 512/min(im.size)
    im = im.resize(tuple(round(x*scale) for x in im.size),resample=Image.Resampling.BICUBIC)
    ar = np.asarray(im)
    y, x = (ar.shape[0]-512)//2, (ar.shape[1]-512)//2
    return ar[y:y+512,x:x+512].copy()

def data_assets():
    archives = STORE/'coco'
    urls = {name:f'http://images.cocodataset.org/{path}' for name,path in {
        'annotations.zip':'annotations/annotations_trainval2014.zip',
        'train2014.zip':'zips/train2014.zip','val2014.zip':'zips/val2014.zip'}.items()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fs = {name:ex.submit(download,url,archives/name) for name,url in urls.items()}
        ann_file = fs['annotations.zip'].result()
        with zipfile.ZipFile(ann_file) as z:
            train = json.loads(z.read('annotations/captions_train2014.json'))
            val = json.loads(z.read('annotations/captions_val2014.json'))
        rng = np.random.default_rng(20261118)
        by_image = {}
        for a in sorted(train['annotations'],key=lambda x:x['id']):
            by_image.setdefault(a['image_id'],[]).append(a)
        fit_ids = rng.choice(sorted(by_image), FIT_N,replace=False)
        fit = [dict(by_image[int(i)][int(rng.integers(len(by_image[int(i)])))]) for i in fit_ids]
        by_val = {}
        for a in sorted(val['annotations'],key=lambda x:x['id']):
            by_val.setdefault(a['image_id'],[]).append(a)
        test_ids = rng.choice(sorted(by_val),TEST_N,replace=False)
        test = [dict(by_val[int(i)][int(rng.integers(len(by_val[int(i)])))]) for i in test_ids]
        anns = sorted(val['annotations'],key=lambda x:x['id'])
        generation = [dict(anns[int(i)]) for i in rng.choice(len(anns),GEN_N,replace=False)]
        image_maps = {'train':{i['id']:i for i in train['images']},'val':{i['id']:i for i in val['images']}}
        for split, records in [('fit',fit),('test',test),('generation',generation)]:
            for index, row in enumerate(records):
                row['sample_id'] = index
                row['file_name'] = image_maps['train' if split=='fit' else 'val'][row['image_id']]['file_name']
                row['noise_seed'] = (81000000 if split=='fit' else 82000000 if split=='test' else 0)+index
        assert not set(fit_ids)&set(test_ids)
        reference = sorted(val['images'],key=lambda x:x['id'])
        assert len(reference)==40504
        cohort = {'fit':fit,'test':test,'generation':generation,'reference':reference,
                  'generation_unique_images':len({r['image_id']for r in generation}),
                  'caption_selection':'50,000 distinct validation caption IDs, uniform without replacement; captions may share images; first10K is a fixed prefix',
                  'seed':20261118}
        target = ROOT/'cohorts.json'
        if target.exists():assert read(target)==cohort
        else:save(target,cohort)
        for name,f in fs.items():f.result()
    receipts = {}
    for name in urls:
        with zipfile.ZipFile(archives/name) as z:
            bad = z.testzip()
            assert bad is None, (name,bad)
        receipts[name] = {'url':urls[name],'sha256':digest(archives/name),'bytes':(archives/name).stat().st_size,'zip_crc_checked':True}
    for split, records, archive in [('fit',fit,'train2014.zip'),('test',test,'val2014.zip'),('reference',reference,'val2014.zip')]:
        out = STORE/'banks'/f'{split}.npy'
        if out.exists():continue
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp = out.with_suffix('.partial.npy')
        bank = np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(records),512,512,3))
        with zipfile.ZipFile(archives/archive) as z:
            prefix = archive.removesuffix('.zip')
            # Parallel Pillow decode, ordered assignment. ZipFile protects its seek/read operations.
            def decode(row):
                with Image.open(io.BytesIO(z.read(prefix+'/'+row['file_name']))) as im:return crop(im)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8)as ex:
                for i, arr in enumerate(ex.map(decode,records)):
                    bank[i]=arr
                    if i%1000==0:print('BANK',split,i,len(records),flush=True)
        bank.flush();del bank;tmp.replace(out)
    save(STORE/'coco/DOWNLOADS.json',receipts)
    return {'cohorts_sha256':digest(ROOT/'cohorts.json'),'banks':{
        split:{'sha256':digest(STORE/'banks'/f'{split}.npy'),'count':n}for split,n in [('fit',FIT_N),('test',TEST_N),('reference',40504)]}}

def main():
    STORE.mkdir(parents=True,exist_ok=True)
    lock=(STORE/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_source()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:
        model=ex.submit(model_assets);data=ex.submit(data_assets)
        receipt={'models':model.result(),'data':data.result(),'complete':True,'time':time.time()}
    expected=read(CONFIG/'assets.json')
    assert receipt['data']==expected['data'], 'COCO identities differ from release'
    for name in ['model','text']:
        for filename,sha in expected['models'][name]['sha256'].items():
            assert receipt['models'][name]['sha256'][filename]==sha, (name,filename)
    save(ROOT/'PREPARED.json',receipt)
    print('PREPARED',flush=True)

if __name__=='__main__':main()
