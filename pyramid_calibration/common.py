"""Frozen identities and atomic receipts for the P-JiT512 calibration campaign."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/pyramid_calibration'
MEASUREMENTS = REPO / 'metrics/pyramid_calibration'
ROOT = Path(os.environ.get('PYRAMID_CALIBRATION_WORKSPACE', 'work/pyramid-calibration')).resolve()
STORE = ROOT / 'output'
SOURCE = Path(__file__).resolve().parent / 'frozen'
MODEL_REV = '05b4ca725545096b5d1485e185c074e70a72a769'
TEXT_REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SOURCE_REV = 'dbe6d654d8c2770e00299d5ee562430447a40f8c'
FIT_N, TEST_N, GEN_N = 10000, 2000, 50000
NEGATIVE = 'watermark, signature, logo, copyright, url'
FIT_BATCH, SAMPLE_BATCH, TEXT_BATCH = 64, 40, 16
TIMES = sorted(set([round(i/50, 6) for i in range(1, 50)] + [.05, .95]))
TEST_TIMES = [.05] + [round(i/10, 6) for i in range(1, 10)] + [.95, .98]

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def key(t):
    return f'{t:.6f}'

def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)

def verify_source():
    manifest = read(REPO/'PYRAMID_CALIBRATION_MANIFEST.json')
    for name, sha in manifest['files'].items():
        assert digest(REPO/name) == sha, name
    return read(CONFIG/'protocol.json')
