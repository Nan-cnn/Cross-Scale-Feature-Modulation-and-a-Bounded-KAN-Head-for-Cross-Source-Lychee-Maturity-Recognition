import os
import glob
import re
import csv
import math
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.metrics import f1_score

# ================= CONFIG (defaults; can be overridden by CLI) ================= #
DEFAULT_MODEL_NAMES = ["MobileNetV2KAN", "MobileNetV2", "ResNet50", "VGG16", "EfficientNetB3"]
DEFAULT_RATIOS = [0.1, 0.2, 0.4, 1.0]


def safe_torch_load_state_dict(path: str, device):
    """Load a checkpoint safely across PyTorch versions.

    - Uses weights_only=True when supported to avoid pickle warnings.
    - Falls back to legacy torch.load signature if needed.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def find_latest_subset_compare_run(base_dir: str = "subset_compare", prefix: str = "subset_compare_") -> str:
    """Auto-pick the latest subset_compare run directory (highest numeric suffix, fallback to mtime)."""
    if not os.path.isdir(base_dir):
        return ""

    cands = []
    for name in os.listdir(base_dir):
        if not name.startswith(prefix):
            continue
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            cands.append(p)

    if not cands:
        return ""

    def _key(p: str):
        bn = os.path.basename(p)
        m = re.search(r"(\d+)$", bn)
        n = int(m.group(1)) if m else -1
        try:
            mt = os.path.getmtime(p)
        except Exception:
            mt = 0.0
        return (n, mt)

    return max(cands, key=_key)


# ================= MODELS ================= #
from model.MobileNetV2KAN import MobileNetV2KAN


def build_model(name: str, img_size: int, num_classes: int = 2) -> nn.Module:
    if name == "MobileNetV2KAN":
        return MobileNetV2KAN(input_size=img_size, num_classes=num_classes)

    if name == "MobileNetV2":
        return models.mobilenet_v2(weights=None, num_classes=num_classes)

    if name == "ResNet50":
        return models.resnet50(weights=None, num_classes=num_classes)

    if name == "VGG16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m

    if name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=None)
        if isinstance(m.classifier, nn.Sequential) and len(m.classifier) >= 2:
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        else:
            m.classifier = nn.Linear(getattr(m.classifier, "in_features", 1536), num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")


# ================= DATA ================= #
class FolderDataset(Dataset):
    def __init__(self, data_dir: str, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for fn in sorted(os.listdir(data_dir)):
            if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            head = fn[0].lower()
            if head == "u":
                y = 0
            elif head == "r":
                y = 1
            else:
                continue
            self.samples.append((os.path.join(data_dir, fn), y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, y, p


def make_test_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# ================= EVAL ================= #
@torch.no_grad()
def eval_one_checkpoint(model: nn.Module, ckpt_path: str, loader: DataLoader, device: torch.device):
    state = safe_torch_load_state_dict(ckpt_path, device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.to(device).eval()

    correct, total = 0, 0
    all_pred, all_lab = [], []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        pred = out.argmax(1)

        total += y.size(0)
        correct += (pred == y).sum().item()

        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_lab.extend(y.detach().cpu().numpy().tolist())

    acc = 100.0 * correct / total if total > 0 else 0.0
    f1 = f1_score(all_lab, all_pred, average="macro", zero_division=0) if total > 0 else 0.0
    return acc, f1


# ================= GRAD-CAM ================= #
def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def _fwd_hook(module, inp, out):
            self.activations = out

        def _bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._h1 = target_layer.register_forward_hook(_fwd_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            self._h2 = target_layer.register_full_backward_hook(_bwd_hook)
        else:
            self._h2 = target_layer.register_backward_hook(lambda m, gi, go: _bwd_hook(m, gi, go))

    def remove(self):
        try:
            self._h1.remove()
        except Exception:
            pass
        try:
            self._h2.remove()
        except Exception:
            pass

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            x = x.requires_grad_(True)
            out = self.model(x)

            if class_idx is None:
                class_idx = int(out.argmax(1).item())

            score = out[:, class_idx].sum()
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            acts = self.activations
            grads = self.gradients

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam_map = (weights * acts).sum(dim=1, keepdim=True)
            cam_map = torch.relu(cam_map)

            cam_map = torch.nn.functional.interpolate(
                cam_map,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False
            )

            cam_map = cam_map[0, 0]
            cam_map = cam_map - cam_map.min()
            cam_map = cam_map / (cam_map.max() + 1e-8)

            return cam_map.detach().cpu().numpy()


def overlay_cam_on_image(pil_img: Image.Image, cam_map: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    cam_map = np.nan_to_num(cam_map, nan=0.0, posinf=1.0, neginf=0.0)
    cam_map = np.clip(cam_map, 0.0, 1.0)

    cam_img = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255.0).astype(np.float32)
    img_f = img.astype(np.float32)

    overlay = (1.0 - alpha) * img_f + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_gradcam_for_ckpt(
        ckpt_path: str,
        model_name: str,
        ratio: float,
        img_size: int,
        sample_paths: List[str],
        out_dir: str,
        device: torch.device
):
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(model_name, img_size=img_size, num_classes=2)
    state = safe_torch_load_state_dict(ckpt_path, device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.to(device).eval()

    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model, target_layer)

    tf = make_test_transform(img_size)

    for p in sample_paths:
        pil = Image.open(p).convert("RGB")
        x = tf(pil).unsqueeze(0).to(device)

        try:
            cam_map = cam.generate(x)
        except Exception as e:
            print(f"[WARN] Grad-CAM failed for {model_name} ratio={ratio:.2f} img={os.path.basename(p)}: {e}")
            continue

        overlay = overlay_cam_on_image(pil, cam_map, alpha=0.4)

        base = os.path.splitext(os.path.basename(p))[0]
        fn = f"{model_name}_ratio{ratio:.2f}_{base}.jpg"
        overlay.save(os.path.join(out_dir, fn), quality=95)

    cam.remove()


# ================= MULTI-SEED DISCOVERY ================= #
def discover_seed_roots(run_dir: str) -> List[str]:
    roots = []
    if os.path.isdir(run_dir):
        roots.append(run_dir)

    seeds_dir = os.path.join(run_dir, "seeds")
    if os.path.isdir(seeds_dir):
        for sd in sorted(glob.glob(os.path.join(seeds_dir, "seed_*"))):
            if os.path.isdir(sd):
                roots.append(sd)
    return roots


def find_ckpt(seed_root: str, model_name: str, ratio: float) -> Optional[str]:
    ckpt_dir = os.path.join(seed_root, model_name, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    expected = os.path.join(ckpt_dir, f"{model_name}_ratio{ratio:.2f}_best.pth")
    if os.path.isfile(expected):
        return expected

    key = f"ratio{ratio:.2f}_best.pth"
    candidates = [p for p in glob.glob(os.path.join(ckpt_dir, "*.pth")) if key in os.path.basename(p)]
    return candidates[0] if candidates else None


# ================= STATS + PLOTS ================= #
@dataclass
class MetricAgg:
    accs: List[float]
    f1s: List[float]

    def mean_std(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        def _ms(v):
            if len(v) == 0:
                return 0.0, 0.0
            if len(v) == 1:
                return float(v[0]), 0.0
            return float(np.mean(v)), float(np.std(v, ddof=1))

        return _ms(self.accs), _ms(self.f1s)


def plot_mean_std_curves(out_path: str, ratios: List[float], model_to_stats: Dict[str, Dict[float, MetricAgg]],
                         key: str):
    plt.figure(figsize=(10, 6))
    xs = ratios

    for mn, rmap in model_to_stats.items():
        means = []
        stds = []
        for r in xs:
            agg = rmap.get(r, MetricAgg([], []))
            (acc_m, acc_s), (f1_m, f1_s) = agg.mean_std()
            if key == "acc":
                means.append(acc_m)
                stds.append(acc_s)
            else:
                means.append(f1_m)
                stds.append(f1_s)

        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)

        plt.plot(xs, means, marker="o", label=mn)
        plt.fill_between(xs, means - stds, means + stds, alpha=0.2)

    plt.title(f"Test {key.upper()} vs Subset Ratio (mean±std over seeds)")
    plt.xlabel("Subset ratio")
    plt.ylabel("Accuracy (%)" if key == "acc" else "F1 (macro)")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ================= MAIN ================= #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="",
                        help="RUN_DIR from subset_compare (contains model folders, and optionally seeds/seed_*)")
    parser.add_argument("--test_dir", type=str, default="./data/banana/test",
                        help="Test dataset folder (u*/r* filenames)")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ratios", type=str, default="0.2,0.4,0.6,1.0")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_NAMES))

    parser.add_argument("--do_gradcam", action="store_true",
                        help="Also generate Grad-CAM overlays for each model/ratio (best ckpt per seed root)")
    parser.add_argument("--gradcam_per_class", type=int, default=1,
                        help="How many test images per class to visualize (deterministic)")
    parser.add_argument("--gradcam_seed", type=int, default=123)

    args = parser.parse_args()

    run_dir = (args.run_dir or "").strip()
    if run_dir == "":
        run_dir = find_latest_subset_compare_run(base_dir="subset_compare", prefix="subset_compare_")
    if run_dir == "" or (not os.path.isdir(run_dir)):
        raise SystemExit("Cannot find run_dir. Provide --run_dir or ensure subset_compare/subset_compare_* exists.")

    print("Using RUN_DIR:", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip() != ""]
    model_names = [x.strip() for x in args.models.split(",") if x.strip() != ""]
    seed_roots = discover_seed_roots(run_dir)

    print("Discovered seeds:")
    for s in seed_roots:
        if s == run_dir:
            print(" - primary:", s)
        else:
            print(" -", os.path.basename(s) + ":", s)

    if len(seed_roots) == 0:
        raise RuntimeError(f"No valid run_dir found: {run_dir}")

    tf = make_test_transform(args.img_size)
    test_ds = FolderDataset(args.test_dir, tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Pick deterministic Grad-CAM samples from test set
    sample_paths = []
    if args.do_gradcam:
        rng = random.Random(args.gradcam_seed)
        u_paths = [p for p, y in test_ds.samples if y == 0]
        r_paths = [p for p, y in test_ds.samples if y == 1]
        u_paths.sort()
        r_paths.sort()
        rng.shuffle(u_paths)
        rng.shuffle(r_paths)
        sample_paths = u_paths[:args.gradcam_per_class] + r_paths[:args.gradcam_per_class]

    out_dir = os.path.join(run_dir, "test_eval_multiseed")
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate: model -> ratio -> MetricAgg
    model_to_stats: Dict[str, Dict[float, MetricAgg]] = {
        mn: {r: MetricAgg([], []) for r in ratios} for mn in model_names
    }

    # Per-seed detailed CSV
    per_seed_csv = os.path.join(out_dir, "per_seed_metrics.csv")
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed_root", "model", "ratio", "ckpt", "test_acc", "test_f1"])

        print("\n==================== TEST EVAL (per seed_root) ====================")
        for sr in seed_roots:
            for mn in model_names:
                for r in ratios:
                    ckpt = find_ckpt(sr, mn, r)
                    if ckpt is None:
                        print(f"[WARN] Missing ckpt: seed_root={os.path.relpath(sr, run_dir)} model={mn} ratio={r:.2f}")
                        continue

                    model = build_model(mn, img_size=args.img_size, num_classes=2)
                    acc, f1 = eval_one_checkpoint(model, ckpt, test_loader, device)

                    model_to_stats[mn][r].accs.append(acc)
                    model_to_stats[mn][r].f1s.append(f1)

                    w.writerow(
                        [os.path.relpath(sr, run_dir), mn, f"{r:.2f}", os.path.relpath(ckpt, run_dir), f"{acc:.4f}",
                         f"{f1:.6f}"])
                    print(
                        f"{os.path.relpath(sr, run_dir):<18} | {mn:<12} | ratio={r:.2f} | Acc={acc:.2f}% | F1={f1:.4f}")

    # Mean±std CSV
    mean_std_csv = os.path.join(out_dir, "mean_std_metrics.csv")
    with open(mean_std_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "ratio", "n", "acc_mean", "acc_std", "f1_mean", "f1_std"])

        print("\n==================== TEST EVAL (mean±std over seed_roots) ====================")
        for mn in model_names:
            for r in ratios:
                agg = model_to_stats[mn][r]
                (acc_m, acc_s), (f1_m, f1_s) = agg.mean_std()
                n = len(agg.accs)
                w.writerow([mn, f"{r:.2f}", n, f"{acc_m:.4f}", f"{acc_s:.4f}", f"{f1_m:.6f}", f"{f1_s:.6f}"])
                print(f"{mn:<12} | ratio={r:.2f} | n={n:<2d} | Acc={acc_m:.2f}±{acc_s:.2f} | F1={f1_m:.4f}±{f1_s:.4f}")

    # Plots (mean±std curves with shadow bands)
    acc_plot = os.path.join(out_dir, "test_acc_mean_std.png")
    f1_plot = os.path.join(out_dir, "test_f1_mean_std.png")
    plot_mean_std_curves(acc_plot, ratios, model_to_stats, key="acc")
    plot_mean_std_curves(f1_plot, ratios, model_to_stats, key="f1")

    print("\nSaved:")
    print(" -", os.path.relpath(per_seed_csv, run_dir))
    print(" -", os.path.relpath(mean_std_csv, run_dir))
    print(" -", os.path.relpath(acc_plot, run_dir))
    print(" -", os.path.relpath(f1_plot, run_dir))

    # Grad-CAM overlays (use the first seed_root that has the ckpt by default)
    if args.do_gradcam and len(sample_paths) > 0:
        gradcam_root = os.path.join(out_dir, "gradcam")
        os.makedirs(gradcam_root, exist_ok=True)

        # Use the primary run_dir ckpt if present; otherwise fall back to the first available seed_root.
        primary_sr = seed_roots[0]

        for mn in model_names:
            for r in ratios:
                ckpt = find_ckpt(primary_sr, mn, r)
                if ckpt is None:
                    continue
                subdir = os.path.join(gradcam_root, f"{mn}_ratio{r:.2f}")
                save_gradcam_for_ckpt(
                    ckpt_path=ckpt,
                    model_name=mn,
                    ratio=r,
                    img_size=args.img_size,
                    sample_paths=sample_paths,
                    out_dir=subdir,
                    device=device
                )

        print(" -", os.path.relpath(gradcam_root, run_dir))


if __name__ == "__main__":
    main()
