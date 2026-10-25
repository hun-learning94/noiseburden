"""Cache exact deterministic RGB crops in a node-local uint8 array."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
from PIL import Image

EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
CHUNK = 256


def load_crop(path):
    spec = importlib.util.spec_from_file_location("frozen_training_cache_crop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.center_crop_arr


def cache_directory(root, source, size, crop_file):
    crop_sha = hashlib.sha256(crop_file.read_bytes()).hexdigest()
    base = root / f"imagenet{size}-{crop_sha[:12]}"
    # Reuse the existing width-64 cache; distinct datasets get distinct paths.
    meta = base / "metadata.json"
    if meta.exists() and json.loads(meta.read_text())["source"] != str(source.resolve()):
        source_sha = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
        return root / f"imagenet{size}-{crop_sha[:12]}-{source_sha[:12]}"
    return base


def publish(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory(source):
    classes = sorted(entry.name for entry in os.scandir(source / "train") if entry.is_dir())
    samples = []
    for label, name in enumerate(classes):
        for folder, _, names in sorted(os.walk(source / "train" / name, followlinks=True)):
            for filename in sorted(names):
                if filename.lower().endswith(EXTENSIONS):
                    path = Path(folder) / filename
                    samples.append((str(path.relative_to(source / "train")), label))
    return classes, samples


def pixels(path, crop, size):
    # Match torchvision's PIL loader before applying the frozen center crop.
    with path.open("rb") as stream:
        with Image.open(stream) as image:
            return np.asarray(crop(image.convert("RGB"), size))


def build(source, dest, workers, crop_file, size, expected_count=1281167, expected_classes=1000):
    assert size > 0 and workers > 0
    source = source.resolve()
    crop = load_crop(crop_file)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (dest / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        crop_sha = hashlib.sha256(crop_file.read_bytes()).hexdigest()
        ready_path = dest / "READY.json"
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            assert ready["source"] == str(source) and ready["crop_sha256"] == crop_sha
            assert ready["images"] == expected_count and ready["class_count"] == expected_classes
            assert ready["shape"] == [expected_count, size, size, 3]
            assert (dest / "images.npy").stat().st_size == ready["array_bytes"]
            labels = np.load(dest / "labels.npy", mmap_mode="r")
            assert labels.shape == (expected_count,) and labels.dtype == np.int32
            print("LOCAL_CACHE_REUSED", str(dest), expected_count, "images", flush=True)
            return ready
        print("LOCAL_CACHE_INVENTORY", str(source), flush=True)
        classes, samples = inventory(source)
        assert len(samples) == expected_count, len(samples)
        assert len(classes) == expected_classes, len(classes)
        encoded = json.dumps(dict(classes=classes, samples=samples), separators=(",", ":"))
        sample_sha = hashlib.sha256(encoded.encode()).hexdigest()
        metadata = dict(source=str(source), classes=classes, images=len(samples), class_count=len(classes),
                        crop_sha256=crop_sha, sample_order_sha256=sample_sha, shape=[len(samples), size, size, 3],
                        dtype="uint8", format="lossless deterministic center crop; RGB; HWC")
        meta_path = dest / "metadata.json"
        if meta_path.exists():
            assert json.loads(meta_path.read_text()) == metadata, "Incompatible partial cache"
        else:
            needed = len(samples) * size * size * 3 + 5 * 2**30
            free = shutil.disk_usage(dest).free
            if free <= needed:
                raise RuntimeError(f"Local image cache needs {needed / 2**30:.1f} GiB free at {dest}; "
                                   f"only {free / 2**30:.1f} GiB is available. "
                                   "Choose another JIT_CACHE_ROOT or explicitly set JIT_LOCAL_CACHE=0.")
            array = np.lib.format.open_memmap(dest / "images.npy", mode="w+", dtype=np.uint8,
                                             shape=tuple(metadata["shape"]))
            array.flush()
            del array
            np.save(dest / "labels.npy", np.asarray([label for _, label in samples], dtype=np.int32))
            (dest / "samples.json").write_text(encoded)
            publish(meta_path, metadata)
        header = np.load(dest / "images.npy", mmap_mode="r")
        offset = header.offset
        assert tuple(header.shape) == tuple(metadata["shape"]) and header.dtype == np.uint8
        del header
        chunks = dest / "chunks"
        chunks.mkdir(exist_ok=True)

        def cache_chunk(start):
            stop = min(start + CHUNK, len(samples))
            marker = chunks / f"{start:07d}.json"
            block = np.memmap(dest / "images.npy", mode="r+", dtype=np.uint8,
                              offset=offset + start * size * size * 3, shape=(stop-start, size, size, 3))
            if marker.exists():
                prior = json.loads(marker.read_text())
                assert prior["sha256"] == hashlib.sha256(block).hexdigest(), start
            else:
                for local_index, (relative, _) in enumerate(samples[start:stop]):
                    block[local_index] = pixels(source / "train" / relative, crop, size)
                block.flush()
                publish(marker, dict(start=start, stop=stop, sha256=hashlib.sha256(block).hexdigest()))
            block._mmap.close()
            return stop - start

        completed = 0
        previous = started
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(cache_chunk, start) for start in range(0, len(samples), CHUNK)]
            for future in as_completed(pending):
                completed += future.result()
                now = time.monotonic()
                if now - previous >= 20 or completed == len(samples):
                    report = dict(completed=completed, total=len(samples), seconds=now-started,
                                  images_per_second=completed/(now-started), host=socket.gethostname())
                    publish(dest / "progress.json", report)
                    print("LOCAL_CACHE_PROGRESS", json.dumps(report), flush=True)
                    previous = now
        # Check pixel equality against the original JPEG loader, and crop idempotence.
        array = np.load(dest / "images.npy", mmap_mode="r")
        labels = np.load(dest / "labels.npy", mmap_mode="r")
        indices = np.unique(np.linspace(0, len(samples)-1, min(256, len(samples)), dtype=int))
        for index in indices:
            relative, label = samples[index]
            assert labels[index] == label
            assert np.array_equal(array[index], pixels(source / "train" / relative, crop, size)), relative
            recropped = np.asarray(crop(Image.fromarray(array[index]), size))
            assert np.array_equal(array[index], recropped), relative
        ready = dict(metadata, array_bytes=(dest / "images.npy").stat().st_size,
                     pixel_checks=len(indices), seconds=time.monotonic()-started,
                     host=socket.gethostname(), workers=workers)
        publish(ready_path, ready)
        print("LOCAL_CACHE_READY", str(dest), len(samples), "images; pixel checks", len(indices), flush=True)
        return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--crop-file", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.dest, args.workers, args.crop_file, args.size)
    if args.report:
        publish(args.report, dict(report, cache_path=str(args.dest)))
