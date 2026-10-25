"""Adapt the freshly extracted distributed entry points to one physical GPU.

No model, objective, optimizer, EMA, sample order or update-count changes.
The same correction is used by the existing successful B200 baseline.
"""
from pathlib import Path
import ast
import hashlib
import json
import os

assert int(os.environ.get('WORLD_SIZE', '1')) == 1
root = Path(os.environ['OUTPUT_ROOT'])
case = json.loads((root / 'case.json').read_text())
assert case['b200'] and case['gpus'] == 1
audit = {}

def replace_file(name, changes):
    path = Path(name)
    before = path.read_text()
    after = before
    for old, new, count in changes:
        assert after.count(old) == count, (name, old, after.count(old))
        after = after.replace(old, new)
    ast.parse(after)
    path.write_text(after)
    audit[name] = dict(before=hashlib.sha256(before.encode()).hexdigest(),
                      after=hashlib.sha256(after.encode()).hexdigest())

replace_file('main_jit.py', [
    ('    data_loader_train = torch.utils.data.DataLoader(\n'
     '        dataset_train, sampler=sampler_train,\n'
     '        batch_size=args.batch_size,\n',
     '    from single_gpu_loader import CompleteUpdateBatchSampler\n'
     '    data_loader_train = torch.utils.data.DataLoader(\n'
     '        dataset_train, batch_sampler=CompleteUpdateBatchSampler(\n'
     '            sampler_train, args.batch_size, args.grad_accum_steps),\n', 1),
    ('        drop_last=True\n', '', 1),
    ('    misc.init_distributed_mode(args)\n',
     '    misc.init_distributed_mode(args)\n'
     '    assert not args.distributed\n'
     '    args.rank = 0; args.world_size = 1; args.gpu = 0\n'
     '    torch.cuda.set_device(0)\n', 1),
    ('    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])\n'
     '    model_without_ddp = model.module\n',
     '    model_without_ddp = model\n', 1),
    ('        if args.distributed:\n            data_loader_train.sampler.set_epoch(epoch)\n',
     '        sampler_train.set_epoch(epoch)\n', 1),
])
replace_file('engine_jit.py', [
    ('torch.distributed.barrier()',
     '(torch.distributed.barrier() if misc.is_dist_avail_and_initialized() else None)', 3),
])
replace_file('continue_width.py', [
    ('dist.barrier()', '(dist.barrier() if dist.is_initialized() else None)', 2),
    ('dist.destroy_process_group()',
     '(dist.destroy_process_group() if dist.is_initialized() else None)', 1),
])
(root / 'single-gpu-compat.json').write_text(json.dumps(audit, indent=2) + '\n')
print('SINGLE_GPU_COMPAT_READY', json.dumps(audit), flush=True)
