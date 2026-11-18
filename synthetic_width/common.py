"""Shared constants. Bump SCHEMA whenever result fields or their meaning change; results also store
a hash of the source files they depend on, so sweep.py reruns anything produced by other code."""
import hashlib
import os

SCHEMA = 3
HERE = os.path.dirname(os.path.abspath(__file__))
DEPS = {
    "spectrum.py": ("spectrum.py", "gmm.py", "common.py"),
    "train.py": ("train.py", "models.py", "gmm.py", "common.py"),
    "train_jit.py": ("train_jit.py", "models_jit.py", "models.py", "train.py", "gmm.py", "common.py"),
}


def code_hash(script):
    h = hashlib.sha256()
    for name in DEPS[script]:
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]
