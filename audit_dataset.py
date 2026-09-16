"""Audit exact and optional near-duplicate leakage across dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from experiment_data import scan_prefix_dataset


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: str | Path, size: int = 8) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def write_rows(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_224.json")
    parser.add_argument("--output-dir", default="reports/data_audit")
    parser.add_argument(
        "--near-hamming",
        type=int,
        default=-1,
        help="Optional dHash threshold; -1 disables the more expensive near-duplicate scan",
    )
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    data_cfg = config["data"]
    prefix_to_label = data_cfg["prefix_to_label"]

    splits: Dict[str, list] = {}
    for split_name in ("train", "validation", "test", "test1"):
        value = data_cfg.get(split_name)
        if not value:
            continue
        directory = Path(value)
        if not directory.is_absolute():
            directory = (project_root / directory).resolve()
        splits[split_name] = scan_prefix_dataset(directory, prefix_to_label)

    hash_index: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    split_counts = {}
    for split_name, samples in splits.items():
        split_counts[split_name] = len(samples)
        print(f"Hashing {split_name}: {len(samples)} images")
        for path, label in samples:
            hash_index[sha256_file(path)].append((split_name, path, label))

    exact_rows = []
    for digest, occurrences in hash_index.items():
        involved_splits = sorted({item[0] for item in occurrences})
        if len(involved_splits) < 2:
            continue
        for left_index in range(len(occurrences)):
            for right_index in range(left_index + 1, len(occurrences)):
                left, right = occurrences[left_index], occurrences[right_index]
                if left[0] == right[0]:
                    continue
                exact_rows.append(
                    {
                        "sha256": digest,
                        "split_a": left[0],
                        "path_a": left[1],
                        "label_a": left[2],
                        "split_b": right[0],
                        "path_b": right[1],
                        "label_b": right[2],
                    }
                )

    near_rows = []
    if args.near_hamming >= 0:
        hashed = {
            split_name: [(path, label, difference_hash(path)) for path, label in samples]
            for split_name, samples in splits.items()
        }
        names = list(hashed)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                print(f"Near-duplicate scan: {left_name} vs {right_name}")
                for path_a, label_a, hash_a in hashed[left_name]:
                    for path_b, label_b, hash_b in hashed[right_name]:
                        distance = (hash_a ^ hash_b).bit_count()
                        if distance <= args.near_hamming:
                            near_rows.append(
                                {
                                    "hamming": distance,
                                    "split_a": left_name,
                                    "path_a": path_a,
                                    "label_a": label_a,
                                    "split_b": right_name,
                                    "path_b": path_b,
                                    "label_b": label_b,
                                }
                            )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    write_rows(
        output_dir / "exact_duplicates.csv",
        ["sha256", "split_a", "path_a", "label_a", "split_b", "path_b", "label_b"],
        exact_rows,
    )
    write_rows(
        output_dir / "near_duplicates.csv",
        ["hamming", "split_a", "path_a", "label_a", "split_b", "path_b", "label_b"],
        near_rows,
    )
    report = {
        "config": str(config_path),
        "split_counts": split_counts,
        "exact_cross_split_pairs": len(exact_rows),
        "near_cross_split_pairs": len(near_rows),
        "near_hamming_threshold": args.near_hamming,
        "status": "leakage_detected" if exact_rows else "pass",
    }
    with (output_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if exact_rows and not args.allow_exact_duplicates:
        raise RuntimeError(
            "Exact cross-split duplicates were detected. Review reports/data_audit before training."
        )


if __name__ == "__main__":
    main()
