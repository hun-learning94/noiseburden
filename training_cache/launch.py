"""Run a JiT training entry point with node-local crops enabled by default."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_cache.cache import build, cache_directory, load_crop, publish


def check_transform(transform, crop, size):
    """Refuse augmentations whose semantics precomputing a crop would change."""
    import numpy as np
    from PIL import Image
    import torch
    from torchvision import transforms

    operations = getattr(transform, "transforms", ())
    if len(operations) != 3 or tuple(type(op) for op in operations) != (
            transforms.Lambda, transforms.RandomHorizontalFlip, transforms.PILToTensor):
        raise ValueError("Local crop caching requires center crop, RandomHorizontalFlip, and PILToTensor. "
                         "Use JIT_LOCAL_CACHE=0 for a different augmentation protocol.")
    torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        generator = np.random.default_rng(78213)
        for h, w in ((size, size), (size * 3, size * 2 + 7), (size - 3, size + 17)):
            image = Image.fromarray(generator.integers(0, 256, (h, w, 3), dtype=np.uint8))
            actual = operations[0](image.copy())
            assert np.array_equal(np.asarray(actual), np.asarray(crop(image, size))), "Center crop differs"
        assert torch.equal(torch_state, torch.get_rng_state()), "Center crop consumes torch RNG"
        assert pickle.dumps(np_state) == pickle.dumps(np.random.get_state()), "Center crop consumes NumPy RNG"
        assert py_state == random.getstate(), "Center crop consumes Python RNG"
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)


def install(source, cache, crop_file, size):
    from torchvision import get_image_backend
    from torchvision import datasets
    from torchvision.datasets.folder import default_loader, pil_loader
    from training_cache.dataset import CachedImageFolder

    original = datasets.ImageFolder
    assert get_image_backend() == "PIL", "Local crop caching requires the PIL image backend"
    crop = load_crop(crop_file)

    def image_folder(root, transform=None, target_transform=None, loader=default_loader,
                     is_valid_file=None, allow_empty=False):
        if Path(root).resolve() != (source / "train").resolve():
            return original(root, transform=transform, target_transform=target_transform,
                            loader=loader, is_valid_file=is_valid_file, allow_empty=allow_empty)
        if loader not in (default_loader, pil_loader) or is_valid_file is not None:
            raise ValueError("Custom ImageFolder loaders/filters require JIT_LOCAL_CACHE=0.")
        check_transform(transform, crop, size)
        return CachedImageFolder(root, transform, target_transform, cache=cache)

    datasets.ImageFolder = image_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--crop-file", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(128, 256), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    source = options.source.resolve()
    entrypoint = options.entrypoint.resolve()
    assert entrypoint.is_file() and options.crop_file.is_file()
    enabled = os.environ.get("JIT_LOCAL_CACHE", "1").lower()
    if enabled not in ("0", "1", "false", "true", "off", "on"):
        raise ValueError("JIT_LOCAL_CACHE must be 1/on/true or 0/off/false")
    record = dict(enabled=enabled in ("1", "true", "on"), source=str(source), image_size=options.size)
    if record["enabled"]:
        local_disk = Path("/scratch") if Path("/scratch").is_dir() else Path("/tmp")
        root = Path(os.environ.get("JIT_CACHE_ROOT", str(local_disk / getpass.getuser() / "epredbad")))
        cache = cache_directory(root, source, options.size, options.crop_file)
        workers = int(os.environ.get("JIT_CACHE_WORKERS", "32"))
        ready = build(source, cache, workers, options.crop_file, options.size)
        record.update(cache_path=str(cache), cache=ready,
            code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob("*.py")})
        install(source, cache, options.crop_file, options.size)
    # The application still saves its persistent source data path in checkpoints.
    # Cache setup happens before its seeding and distributed initialization.
    if int(os.environ.get("RANK", "0")) == 0:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        publish(options.report, record)
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *options.arguments]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
