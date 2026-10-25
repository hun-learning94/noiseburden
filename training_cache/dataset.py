"""Read cached RGB crops while preserving the original torchvision transform."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CachedImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, *, cache):
        self.root = str(root)
        self.cache = Path(cache)
        assert (self.cache / "READY.json").is_file()
        self.metadata = json.loads((self.cache / "metadata.json").read_text())
        self.classes = self.metadata["classes"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform, self.target_transform = transform, target_transform
        self.images = np.load(self.cache / "images.npy", mmap_mode="r")
        self.targets = np.load(self.cache / "labels.npy", mmap_mode="r")
        assert len(self.images) == len(self.targets) == self.metadata["images"]
        self._indices = None
        self._samples = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        source_index = index if self._indices is None else int(self._indices[index])
        image = Image.fromarray(self.images[source_index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    @property
    def samples(self):
        if self._samples is None:
            record = json.loads((self.cache / "samples.json").read_text())
            self._samples = [(str(Path(self.root) / relative), label) for relative, label in record["samples"]]
        return self._samples

    @samples.setter
    def samples(self, value):
        # Support the launchers' optional held-out-image exclusion without
        # changing the cached pixel/label indexing.
        if value is self._samples:
            return
        record = json.loads((self.cache / "samples.json").read_text())
        order = {str(Path(self.root) / relative): i for i, (relative, _) in enumerate(record["samples"])}
        self._indices = np.asarray([order[path] for path, _ in value], dtype=np.int64)
        self.targets = np.asarray([label for _, label in value], dtype=np.int32)
        self._samples = value

    @property
    def imgs(self):
        return self.samples

    @imgs.setter
    def imgs(self, value):
        self.samples = value

    def __repr__(self):
        return f"CachedImageFolder(images={len(self)}, classes={len(self.classes)}, cache={self.cache})"
