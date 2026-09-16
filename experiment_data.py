"""Dataset and deterministic sampling utilities for multi-model experiments."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Sample = Tuple[str, int]


def scan_prefix_dataset(directory: str | Path, prefix_to_label: Dict[str, int]) -> List[Sample]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    normalized = {str(key).lower(): int(value) for key, value in prefix_to_label.items()}
    samples: List[Sample] = []
    ignored: List[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = normalized.get(path.name[0].lower())
        if label is None:
            ignored.append(path.name)
            continue
        samples.append((str(path.resolve()), label))
    if ignored:
        print(f"[WARN] {root}: ignored {len(ignored)} files with unknown prefixes")
    if not samples:
        raise RuntimeError(f"No labeled images found in {root}")
    return samples


def balanced_nested_subset(samples: Sequence[Sample], ratio: float, seed: int) -> List[Sample]:
    """Return a class-balanced, seed-stable prefix subset.

    Calling this function with the same seed and increasing ratios produces
    nested subsets, so data-volume comparisons differ only by added samples.
    """

    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    by_class: Dict[int, List[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    if len(by_class) < 2:
        raise RuntimeError("At least two classes are required")
    minimum = min(len(items) for items in by_class.values())
    per_class = max(1, int(round(minimum * float(ratio))))

    selected: List[Sample] = []
    for label in sorted(by_class):
        items = sorted(by_class[label])
        random.Random(int(seed) + label * 100_003).shuffle(items)
        selected.extend(items[:per_class])
    random.Random(int(seed) + 7_919).shuffle(selected)
    return selected


class PrefixImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, path


def build_transforms(image_size: int, robust_augmentation: bool = True):
    image_size = int(image_size)
    train_ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
    ]
    if robust_augmentation:
        train_ops.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.35, 0.35, 0.25, 0.08)], probability := 0.8
                ),
                transforms.RandomGrayscale(p=0.08),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.8))], p=0.2),
                transforms.RandomPerspective(distortion_scale=0.12, p=0.15),
                transforms.RandomAutocontrast(p=0.15),
                transforms.RandomEqualize(p=0.08),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.4, 2.5), value="random"),
            # Neutral from-scratch normalization; it does not import ImageNet statistics.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 1.08))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(train_ops), eval_transform


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    samples: Sequence[Sample],
    transform,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        PrefixImageDataset(samples, transform),
        batch_size=min(int(batch_size), len(samples)),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=int(workers) > 0,
    )


def write_manifest(path: str | Path, samples: Sequence[Sample], project_root: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(project_root).resolve()
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        for sample_path, label in samples:
            try:
                display_path = str(Path(sample_path).resolve().relative_to(project_root))
            except ValueError:
                display_path = str(Path(sample_path).resolve())
            writer.writerow([display_path, label])


__all__ = [
    "PrefixImageDataset",
    "balanced_nested_subset",
    "build_transforms",
    "make_loader",
    "scan_prefix_dataset",
    "seed_everything",
    "write_manifest",
]
