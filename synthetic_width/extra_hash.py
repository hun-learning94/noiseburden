"""Code hashes for scripts added after the tier-1 sweep started. Kept out of common.py, which every
result's hash covers, so that adding scripts here does not invalidate finished results."""
import hashlib
import os

from common import DEPS, HERE, code_hash

EXTRA_DEPS = {
    "train_resume.py": ("train_resume.py", "models.py", "gmm.py", "common.py"),
}


def script_hash(script):
    if script in DEPS:
        return code_hash(script)
    h = hashlib.sha256()
    for name in EXTRA_DEPS[script]:
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]
