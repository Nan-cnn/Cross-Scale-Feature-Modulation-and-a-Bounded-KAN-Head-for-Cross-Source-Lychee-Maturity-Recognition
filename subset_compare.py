import os
import random
import csv
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from model.MobileNetV2KAN import MobileNetV2KAN

# =========================
# Global config
# =========================
EPOCHS = 20
PATIENCE = 10
BATCH_SIZE = 64
IMG_SIZE = 64

SUBSET_RATIOS = [0.2, 0.4, 0.6, 1.0]
ZOOM_EPOCHS = 3

TRAIN_DIR = "./data/banana/train"
VAL_DIR = "./data/banana/val"

LR = 5e-3
ETA_MIN = 3e-4
SEED = 42

# Multi-seed: run 10 seeds
NUM_SEEDS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ["MobileNetV2KAN", "MobileNetV2", "ResNet50", "VGG16", "EfficientNetB3"]


def get_next_folder(base_dir="subset_compare/subset_compare"):
    idx = 1
    while True:
        folder = f"{base_dir}_{idx}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return folder
        idx += 1


RUN_DIR = get_next_folder("subset_compare/subset_compare")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (best-effort)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_all_paths(data_dir: str, seed: int = 42):
    rng = random.Random(seed)
    u_imgs, r_imgs = [], []
    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(data_dir, img_name)
        head = img_name[0].lower()
        if head == "o":
            u_imgs.append((p, 0))
        elif head == "r":
            r_imgs.append((p, 1))
    rng.shuffle(u_imgs)
    rng.shuffle(r_imgs)
    return u_imgs, r_imgs


def make_balanced_subset(u_imgs, r_imgs, ratio: float, seed: int = 42):
    rng = random.Random(seed)
    u = u_imgs.copy()
    r = r_imgs.copy()
    rng.shuffle(u)
    rng.shuffle(r)

    base = min(len(u), len(r))
    n = max(1, int(base * float(ratio)))
    u = u[:n]
    r = r[:n]

    combined = []
    for ui, ri in zip(u, r):
        combined.append(ui)
        combined.append(ri)

    paths, labels = zip(*combined)
    return list(paths), list(labels), n


class DeterministicAugment:

    def __init__(self, transform, base_seed: int):
        self.transform = transform
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def apply(self, img, idx: int):
        # A stable seed mapping: base_seed + epoch + idx
        seed = self.base_seed + self.epoch * 1_000_003 + int(idx) * 9_176

        py_state = random.getstate()
        random.seed(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            out = self.transform(img)

        random.setstate(py_state)
        return out


class LycheeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            if hasattr(self.transform, "apply"):
                img = self.transform.apply(img, idx)
            else:
                img = self.transform(img)
        return img, self.labels[idx]


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def build_model(name: str):
    if name == "MobileNetV2KAN":
        return MobileNetV2KAN(input_size=IMG_SIZE, num_classes=2)

    if name == "MobileNetV2":
        return models.mobilenet_v2(weights=None, num_classes=2)

    if name == "ResNet50":
        return models.resnet50(weights=None, num_classes=2)

    if name == "VGG16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
        return m

    if name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=None)
        if isinstance(m.classifier, nn.Sequential) and len(m.classifier) >= 2:
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        else:
            m.classifier = nn.Linear(getattr(m.classifier, "in_features", 1536), 2)
        return m

    raise ValueError(f"Unknown model: {name}")


def evaluate(model, loader, device, criterion):
    model.eval()
    v_loss_sum = 0.0
    correct, total = 0, 0
    all_pred, all_lab = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            v_loss_sum += loss.item()

            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

            all_pred.extend(pred.detach().cpu().numpy().tolist())
            all_lab.extend(y.detach().cpu().numpy().tolist())

    val_loss = v_loss_sum / max(1, len(loader))
    val_acc = 100.0 * correct / max(1, total)
    val_f1 = f1_score(all_lab, all_pred, average="macro", zero_division=0)
    return val_loss, val_acc, val_f1


def train_one_tier(model, name: str, train_loader, val_loader, epochs: int, patience: int, save_path: str,
                   eta_min: float):
    """
    Keep core training logic aligned to only_subset_compare.
    Returns history dict (train_loss/val_loss/train_acc/val_acc/val_f1) with early stop.
    """
    device = DEVICE
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(LR),
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs) * 3),
        eta_min=float(eta_min)
    )

    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_acc = -1.0
    no_gain = 0

    for ep in range(int(epochs)):
        # Sync epoch for deterministic augmentation (if enabled)
        if hasattr(train_loader.dataset, "transform") and hasattr(train_loader.dataset.transform, "set_epoch"):
            train_loader.dataset.transform.set_epoch(ep)

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"{name} [Epoch {ep + 1}/{epochs}]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = outputs.argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        scheduler.step()
        # Clamp LR to eta_min
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], float(eta_min))

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["val_f1"].append(val_f1)

        # Early stop on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_gain = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_gain += 1
            if no_gain >= int(patience):
                break

    return hist


def _safe_ylim(values, pad_ratio=0.08):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmax, vmin):
        return vmin - 1.0, vmax + 1.0
    pad = (vmax - vmin) * float(pad_ratio)
    return vmin - pad, vmax + pad


def plot_metric_with_inset(model_dir, title, ylabel, ratio_to_mean, ratio_to_std, save_name, inset_loc,
                           acc_inset_threshold=0.5):
    """
    Plot mean curve with std shading for different ratios, plus an inset of last ZOOM_EPOCHS
    only when curves are close (difference <= threshold).
    """
    # align by minimum epoch length
    min_len = min(len(s) for s in ratio_to_mean.values() if len(s) > 0)
    epochs = list(range(1, min_len + 1))

    # decide whether to draw inset
    finals = [ratio_to_mean[r][-1] for r in ratio_to_mean if len(ratio_to_mean[r]) > 0]
    need_inset = (max(finals) - min(finals)) <= float(acc_inset_threshold) if len(finals) > 1 else False

    fig, ax = plt.subplots(figsize=(10, 6))
    for ratio in SUBSET_RATIOS:
        mean_series = np.array(ratio_to_mean[ratio][:min_len], dtype=float)
        std_series = np.array(ratio_to_std[ratio][:min_len], dtype=float)
        ax.plot(epochs, mean_series, label=f"ratio={ratio:.2f}")
        ax.fill_between(epochs, mean_series - std_series, mean_series + std_series, alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    fig.subplots_adjust(right=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    if need_inset and min_len >= 2:
        axins = inset_axes(ax, width="40%", height="40%", loc=inset_loc, borderpad=1.2)
        k = min(int(ZOOM_EPOCHS), min_len)
        zoom_x = epochs[-k:]
        zoom_vals = []

        for ratio in SUBSET_RATIOS:
            mean_series = np.array(ratio_to_mean[ratio][:min_len], dtype=float)
            std_series = np.array(ratio_to_std[ratio][:min_len], dtype=float)
            y = mean_series[-k:]
            s = std_series[-k:]
            axins.plot(zoom_x, y)
            axins.fill_between(zoom_x, y - s, y + s, alpha=0.18)
            zoom_vals.extend((y - s).tolist() + (y + s).tolist())

        axins.set_title(f"Last {k} Epochs", fontsize=9)
        axins.grid(True)
        y0, y1 = _safe_ylim(np.array(zoom_vals, dtype=float))
        axins.set_ylim(y0, y1)

    out_path = os.path.join(model_dir, save_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_models_same_ratio(out_dir, ratio, metric_key, metric_label, per_model_mean, per_model_std,
                           inset_threshold=0.5, inset_loc="lower left"):
    min_len = min(len(per_model_mean[mn][ratio][metric_key]) for mn in MODEL_NAMES)
    x = list(range(1, min_len + 1))

    finals = []
    for mn in MODEL_NAMES:
        finals.append(per_model_mean[mn][ratio][metric_key][min_len - 1])
    need_inset = (max(finals) - min(finals)) <= float(inset_threshold) if len(finals) > 1 else False

    fig, ax = plt.subplots(figsize=(10, 6))

    for mn in MODEL_NAMES:
        mean_series = np.array(per_model_mean[mn][ratio][metric_key][:min_len], dtype=float)
        std_series = np.array(per_model_std[mn][ratio][metric_key][:min_len], dtype=float)
        ax.plot(x, mean_series, label=mn)
        ax.fill_between(x, mean_series - std_series, mean_series + std_series, alpha=0.18)

    ax.set_title(f"{metric_label} Comparison @ ratio={ratio:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if need_inset and min_len >= 2:
        axins = inset_axes(ax, width="36%", height="36%", loc=inset_loc, borderpad=1.2)
        k = min(int(ZOOM_EPOCHS), min_len)
        xz = list(range(min_len - k + 1, min_len + 1))
        zoom_vals = []

        for mn in MODEL_NAMES:
            mean_series = np.array(per_model_mean[mn][ratio][metric_key][:min_len], dtype=float)
            std_series = np.array(per_model_std[mn][ratio][metric_key][:min_len], dtype=float)
            y = mean_series[-k:]
            s = std_series[-k:]
            axins.plot(xz, y)
            axins.fill_between(xz, y - s, y + s, alpha=0.18)
            zoom_vals.extend((y - s).tolist() + (y + s).tolist())

        axins.set_title(f"Last {k} Epochs", fontsize=9)
        axins.grid(True)
        y0, y1 = _safe_ylim(np.array(zoom_vals, dtype=float))
        axins.set_ylim(y0, y1)

    p = os.path.join(out_dir, f"compare_{metric_key}_ratio{ratio:.2f}.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p


def _mean_std_curves(list_of_series: List[List[float]]) -> Tuple[List[float], List[float]]:
    min_len = min(len(s) for s in list_of_series if len(s) > 0)
    arr = np.array([np.array(s[:min_len], dtype=float) for s in list_of_series], dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    return mean.tolist(), std.tolist()


def main():
    print("RUN_DIR:", RUN_DIR)
    print("MODELS:", MODEL_NAMES)
    print("RATIOS:", SUBSET_RATIOS)
    print("NUM_SEEDS:", NUM_SEEDS)
    os.makedirs(RUN_DIR, exist_ok=True)

    # Seed list
    seeds = [SEED + i for i in range(int(NUM_SEEDS))]
    primary_seed = seeds[0]

    # Load and fix validation set order ONCE using primary seed
    set_global_seed(primary_seed)
    train_u_all, train_r_all = load_all_paths(TRAIN_DIR, seed=primary_seed)
    val_u, val_r = load_all_paths(VAL_DIR, seed=primary_seed)

    # Build val set in a balanced + deterministic order
    val_combined = []
    min_len = min(len(val_u), len(val_r))
    for i in range(min_len):
        val_combined.append(val_u[i])
        val_combined.append(val_r[i])
    extra = val_u[min_len:] if len(val_u) > min_len else val_r[min_len:]
    val_combined.extend(extra)
    val_paths, val_labels = zip(*val_combined) if len(val_combined) > 0 else ([], [])

    val_dataset = LycheeDataset(list(val_paths), list(val_labels), transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    # Store histories per seed for later mean±std aggregation
    # seed_hist[seed][model][ratio] = dict(hist + meta)
    seed_hist: Dict[int, Dict[str, Dict[float, Dict]]] = {}

    t0 = time.time()
    for seed in seeds:
        # Decide where to write this seed's outputs
        if seed == primary_seed:
            seed_root = RUN_DIR
            seed_tag = "primary"
        else:
            seed_root = os.path.join(RUN_DIR, "seeds", f"seed_{seed}")
            seed_tag = f"seed_{seed}"
        os.makedirs(seed_root, exist_ok=True)

        print(f"\n==================== TRAIN SEED: {seed_tag} ====================")
        set_global_seed(seed)

        # Re-load train lists per seed
        train_u, train_r = load_all_paths(TRAIN_DIR, seed=seed)

        # Pre-build per-ratio train subsets ONCE per seed
        ratio_to_data = {}
        for ri, ratio in enumerate(SUBSET_RATIOS):
            ratio_seed = seed + ri * 13
            train_paths, train_labels, n_each = make_balanced_subset(
                train_u, train_r, ratio=ratio, seed=ratio_seed
            )
            ratio_to_data[ratio] = (train_paths, train_labels, n_each, ratio_seed)

        seed_hist[seed] = {}

        for mn in MODEL_NAMES:
            model_dir = os.path.join(seed_root, mn)
            os.makedirs(model_dir, exist_ok=True)

            ckpt_dir = os.path.join(model_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Write epoch curves for this seed/model
            model_csv = os.path.join(model_dir, "epoch_curves.csv")
            with open(model_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ratio", "epoch", "train_loss", "val_loss", "val_acc", "val_f1"])

            seed_hist[seed][mn] = {}

            print(f"\n---- MODEL: {mn} ----")
            for ri, ratio in enumerate(SUBSET_RATIOS):
                train_paths, train_labels, n_each, ratio_seed = ratio_to_data[ratio]
                subset_size = len(train_paths)

                tier_dir = os.path.join(model_dir, f"ratio_{ratio:.2f}".replace(".", "_"))
                os.makedirs(tier_dir, exist_ok=True)

                if seed == primary_seed:
                    with open(os.path.join(tier_dir, "subset_train_list.txt"), "w", encoding="utf-8") as f:
                        for p, y in zip(train_paths, train_labels):
                            f.write(f"{p}\t{y}\n")

                # Build a deterministic-augmentation train loader (synced per ratio across models)
                aug_seed = ratio_seed * 100  # depends on seed+ratio_idx, not on model
                det_aug = DeterministicAugment(train_transform, base_seed=aug_seed)
                train_dataset = LycheeDataset(train_paths, train_labels, transform=det_aug)

                bs = min(int(BATCH_SIZE), len(train_dataset))
                if bs < 2:
                    raise RuntimeError(f"Subset too small: subset_size={len(train_dataset)}")

                # Make shuffle deterministic and synced across models
                g = torch.Generator().manual_seed(ratio_seed)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=bs,
                    shuffle=True,
                    drop_last=True,
                    generator=g
                )

                # Reset RNG before each (model, ratio) training to prevent cross-model RNG drift
                set_global_seed(ratio_seed)

                model = build_model(mn)

                best_ckpt_path = os.path.join(ckpt_dir, f"{mn}_ratio{ratio:.2f}_best.pth")
                hist = train_one_tier(
                    model=model,
                    name=f"{mn} r={ratio:.2f}",
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    save_path=best_ckpt_path,
                    eta_min=ETA_MIN
                )

                # record
                seed_hist[seed][mn][ratio] = {
                    "hist": hist,
                    "subset_size": subset_size,
                    "per_class": n_each,
                    "best_ckpt": best_ckpt_path,
                    "stopped_epoch": len(hist["val_acc"]),
                    "best_acc": max(hist["val_acc"]) if hist["val_acc"] else 0.0,
                    "best_f1": max(hist["val_f1"]) if hist["val_f1"] else 0.0,
                    "best_epoch": (int(np.argmax(hist["val_acc"])) + 1) if hist["val_acc"] else 0,
                }

                # append epoch curves (for this seed/model)
                with open(model_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for e in range(len(hist["val_acc"])):
                        w.writerow([
                            ratio,
                            e + 1,
                            hist["train_loss"][e],
                            hist["val_loss"][e],
                            hist["val_acc"][e],
                            hist["val_f1"][e],
                        ])

                print(
                    f"{mn} | ratio={ratio:.2f} | subset={subset_size} | per_class={n_each} | "
                    f"BestAcc={seed_hist[seed][mn][ratio]['best_acc']:.2f}% | "
                    f"BestEpoch={seed_hist[seed][mn][ratio]['best_epoch']} | "
                    f"Stopped@{seed_hist[seed][mn][ratio]['stopped_epoch']}"
                )

                # release
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n==================== AGGREGATE (mean±std over seeds) ====================")

    # Collect mean/std for epoch curves
    per_model_mean: Dict[str, Dict[float, Dict[str, List[float]]]] = {}
    per_model_std: Dict[str, Dict[float, Dict[str, List[float]]]] = {}
    mean_std_rows = []

    for mn in MODEL_NAMES:
        per_model_mean[mn] = {}
        per_model_std[mn] = {}

        for ratio in SUBSET_RATIOS:
            # series lists across seeds
            tr_list, vl_list, va_list, vf_list = [], [], [], []
            best_acc_list, best_f1_list, best_ep_list = [], [], []

            for seed in seeds:
                h = seed_hist[seed][mn][ratio]["hist"]
                tr_list.append(h["train_loss"])
                vl_list.append(h["val_loss"])
                va_list.append(h["val_acc"])
                vf_list.append(h["val_f1"])
                best_acc_list.append(seed_hist[seed][mn][ratio]["best_acc"])
                best_f1_list.append(seed_hist[seed][mn][ratio]["best_f1"])
                best_ep_list.append(seed_hist[seed][mn][ratio]["best_epoch"])

            tr_mean, tr_std = _mean_std_curves(tr_list)
            vl_mean, vl_std = _mean_std_curves(vl_list)
            va_mean, va_std = _mean_std_curves(va_list)
            vf_mean, vf_std = _mean_std_curves(vf_list)

            per_model_mean[mn][ratio] = {
                "train_loss": tr_mean,
                "val_loss": vl_mean,
                "val_acc": va_mean,
                "val_f1": vf_mean,
            }
            per_model_std[mn][ratio] = {
                "train_loss": tr_std,
                "val_loss": vl_std,
                "val_acc": va_std,
                "val_f1": vf_std,
            }

            mean_std_rows.append([
                mn,
                ratio,
                float(np.mean(best_acc_list)),
                float(np.std(best_acc_list, ddof=0)),
                float(np.mean(best_f1_list)),
                float(np.std(best_f1_list, ddof=0)),
                float(np.mean(best_ep_list)),
                float(np.std(best_ep_list, ddof=0)),
            ])

    # Save mean±std table
    out_csv_mean_std = os.path.join(RUN_DIR, "subset_results_mean_std.csv")
    with open(out_csv_mean_std, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "ratio", "best_val_acc_mean", "best_val_acc_std", "best_val_f1_mean", "best_val_f1_std",
                    "best_epoch_mean", "best_epoch_std"])
        for row in mean_std_rows:
            w.writerow(row)

    # Save primary seed best table (keeps legacy behavior)
    out_csv_primary = os.path.join(RUN_DIR, "subset_results.csv")
    with open(out_csv_primary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "ratio", "subset_size", "best_val_acc", "best_val_f1", "best_epoch", "best_ckpt"])
        for mn in MODEL_NAMES:
            for ratio in SUBSET_RATIOS:
                meta = seed_hist[primary_seed][mn][ratio]
                w.writerow([
                    mn,
                    ratio,
                    meta["subset_size"],
                    meta["best_acc"],
                    meta["best_f1"],
                    meta["best_epoch"],
                    meta["best_ckpt"],
                ])

    print("Saved:")
    print(" -", os.path.relpath(out_csv_primary, RUN_DIR))
    print(" -", os.path.relpath(out_csv_mean_std, RUN_DIR))

    for mn in MODEL_NAMES:
        model_dir = os.path.join(RUN_DIR, mn)
        os.makedirs(model_dir, exist_ok=True)

        p1 = plot_metric_with_inset(
            model_dir=model_dir,
            title=f"{mn} - Val Loss (ratios) mean±std",
            ylabel="Val Loss",
            ratio_to_mean={r: per_model_mean[mn][r]["val_loss"] for r in SUBSET_RATIOS},
            ratio_to_std={r: per_model_std[mn][r]["val_loss"] for r in SUBSET_RATIOS},
            save_name="loss_ratios.png",
            inset_loc="upper left",
            acc_inset_threshold=0.05,
        )
        p2 = plot_metric_with_inset(
            model_dir=model_dir,
            title=f"{mn} - Val Accuracy (ratios) mean±std",
            ylabel="Val Acc (%)",
            ratio_to_mean={r: per_model_mean[mn][r]["val_acc"] for r in SUBSET_RATIOS},
            ratio_to_std={r: per_model_std[mn][r]["val_acc"] for r in SUBSET_RATIOS},
            save_name="acc_ratios.png",
            inset_loc="lower left",
            acc_inset_threshold=0.8,
        )
        p3 = plot_metric_with_inset(
            model_dir=model_dir,
            title=f"{mn} - Val F1 (ratios) mean±std",
            ylabel="Val F1 (macro)",
            ratio_to_mean={r: per_model_mean[mn][r]["val_f1"] for r in SUBSET_RATIOS},
            ratio_to_std={r: per_model_std[mn][r]["val_f1"] for r in SUBSET_RATIOS},
            save_name="f1_ratios.png",
            inset_loc="lower left",
            acc_inset_threshold=0.02,
        )

        print(f"\n[{mn}] Saved mean±std figures:")
        print(" -", os.path.relpath(p1, RUN_DIR))
        print(" -", os.path.relpath(p2, RUN_DIR))
        print(" -", os.path.relpath(p3, RUN_DIR))

    test_by_radio_dir = os.path.join(RUN_DIR, "test_by_radio")
    os.makedirs(test_by_radio_dir, exist_ok=True)

    for ratio in SUBSET_RATIOS:
        plot_models_same_ratio(
            out_dir=test_by_radio_dir,
            ratio=ratio,
            metric_key="val_loss",
            metric_label="Val Loss (mean±std)",
            per_model_mean=per_model_mean,
            per_model_std=per_model_std,
            inset_threshold=0.05,
            inset_loc="upper left",
        )
        plot_models_same_ratio(
            out_dir=test_by_radio_dir,
            ratio=ratio,
            metric_key="val_acc",
            metric_label="Val Acc (%) (mean±std)",
            per_model_mean=per_model_mean,
            per_model_std=per_model_std,
            inset_threshold=0.8,
            inset_loc="lower left",
        )
        plot_models_same_ratio(
            out_dir=test_by_radio_dir,
            ratio=ratio,
            metric_key="val_f1",
            metric_label="Val F1 (mean±std)",
            per_model_mean=per_model_mean,
            per_model_std=per_model_std,
            inset_threshold=0.02,
            inset_loc="lower left",
        )

    print(f"\nAll done. Total time: {time.time() - t0:.1f}s")
    print("All outputs saved to:", RUN_DIR)


if __name__ == "__main__":
    main()
