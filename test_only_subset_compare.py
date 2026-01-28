import os
import re
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch.nn.functional as F
from matplotlib import cm

from model.MobileNetV2KAN import MobileNetV2KAN


def find_latest_only_run(base_dir="model_result", prefix="subset_multi"):
    if not os.path.isdir(base_dir):
        return None
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    best = None
    best_idx = -1
    for name in os.listdir(base_dir):
        m = pat.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_idx:
            best_idx = idx
            best = os.path.join(base_dir, name)
    return best


def discover_seed_dirs(run_dir: str):
    seed_dirs = [run_dir]
    seeds_root = os.path.join(run_dir, "seeds")
    if os.path.isdir(seeds_root):
        for name in sorted(os.listdir(seeds_root)):
            p = os.path.join(seeds_root, name)
            if os.path.isdir(p) and name.startswith("seed_"):
                seed_dirs.append(p)
    return seed_dirs


def parse_ratio_from_dirname(name: str):
    if not name.startswith("ratio_"):
        return None
    s = name[len("ratio_"):]
    s = s.replace("_", ".")
    try:
        return float(s)
    except Exception:
        return None


def list_ratios(seed_dir: str, user_ratios=None):
    if user_ratios is not None and len(user_ratios) > 0:
        return sorted(set(user_ratios))
    ratios = []
    for name in os.listdir(seed_dir):
        p = os.path.join(seed_dir, name)
        if not os.path.isdir(p):
            continue
        r = parse_ratio_from_dirname(name)
        if r is not None:
            ratios.append(r)
    return sorted(set(ratios))


class ImageFolderByPrefix(Dataset):
    def __init__(self, data_dir: str, transform, return_path: bool = False):
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        for img_name in os.listdir(data_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            p = os.path.join(data_dir, img_name)
            c = img_name[0].lower()
            if c == "u":
                self.samples.append((p, 0))
            elif c == "r":
                self.samples.append((p, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        if self.return_path:
            return x, y, path
        return x, y


def load_state_dict_safely(ckpt_path: str, device):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def evaluate_ckpt(ckpt_path: str, test_loader, device, img_size: int):
    model = MobileNetV2KAN(input_size=img_size)
    state = load_state_dict_safely(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    acc = 100.0 * correct / total if total > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if len(all_labels) > 0 else 0.0
    return acc, f1


def find_last_conv_module(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = self.target_layer.register_forward_hook(self._forward_hook)
        self.h2 = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor):
        cam = cam - cam.min()
        denom = cam.max() - cam.min()
        if denom.abs().item() < 1e-12:
            return cam * 0.0
        return cam / denom

    def generate(self, x: torch.Tensor, class_idx: int = None):
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[:, class_idx].sum()

        score.backward(retain_graph=False)

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Failed to capture activations/gradients for GradCAM")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = self._normalize(cam[0, 0]).detach().cpu().numpy()
        return cam, class_idx


def overlay_cam_on_image(img_rgb_uint8: np.ndarray, cam_01: np.ndarray, alpha: float = 0.4):
    h, w = img_rgb_uint8.shape[:2]
    if cam_01.shape[0] != h or cam_01.shape[1] != w:
        cam_01 = np.array(Image.fromarray((cam_01 * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)) / 255.0

    heat = cm.get_cmap("jet")(cam_01)[..., :3]
    heatmap = (heat * 255.0).astype(np.uint8)

    overlay = (img_rgb_uint8.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def pick_gradcam_samples(test_dir: str, img_size: int, per_class: int, seed: int):
    rng = random.Random(seed)
    files_u, files_r = [], []
    for name in os.listdir(test_dir):
        if not name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(test_dir, name)
        c = name[0].lower()
        if c == "u":
            files_u.append(p)
        elif c == "r":
            files_r.append(p)

    rng.shuffle(files_u)
    rng.shuffle(files_r)

    sel_u = files_u[:per_class]
    sel_r = files_r[:per_class]
    return [(p, 0) for p in sel_u] + [(p, 1) for p in sel_r]


def save_gradcam_examples(run_dir: str, ckpt_path: str, ratio: float, test_dir: str, img_size: int,
                          per_class: int, sample_seed: int, device):
    out_dir = os.path.join(run_dir, "gradcam", f"ratio_{str(ratio).replace('.', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    model = MobileNetV2KAN(input_size=img_size)
    state = load_state_dict_safely(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    target = None
    if hasattr(model, "backbone"):
        bb = getattr(model, "backbone")
        if hasattr(bb, "features") and isinstance(bb.features, nn.Sequential) and len(bb.features) > 0:
            target = bb.features[-1]
    if target is None and hasattr(model, "features") and isinstance(model.features, nn.Sequential) and len(model.features) > 0:
        target = model.features[-1]
    if target is None:
        target = find_last_conv_module(model)
    if target is None:
        raise RuntimeError("Cannot find a Conv2d layer for GradCAM target")

    cam_engine = GradCAM(model, target)

    resize_only = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    samples = pick_gradcam_samples(test_dir, img_size, per_class=per_class, seed=sample_seed)
    for p, true_y in samples:
        img = Image.open(p).convert("RGB")
        img_r = img.resize((img_size, img_size), resample=Image.BILINEAR)
        x = resize_only(img).unsqueeze(0).to(device)

        cam, pred_y = cam_engine.generate(x, class_idx=None)

        img_rgb = np.array(img_r).astype(np.uint8)
        overlay = overlay_cam_on_image(img_rgb, cam, alpha=0.4)

        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}_T{true_y}_P{pred_y}.jpg")
        Image.fromarray(overlay).save(out_path)

    cam_engine.remove()
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="./data/lychee/test1")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ratios", type=str, default="")
    parser.add_argument("--do_gradcam", action="store_true")
    parser.add_argument("--gradcam_ratio", type=float, default=1.0)
    parser.add_argument("--gradcam_per_class", type=int, default=2)
    parser.add_argument("--gradcam_seed", type=int, default=123)
    args = parser.parse_args()

    # Reduce noisy torch.load FutureWarning spam on older PyTorch builds.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"You are using `torch\.load` with `weights_only=False`.*",
    )

    run_dir = args.run_dir.strip()
    if run_dir == "":
        run_dir = find_latest_only_run()
        if run_dir is None:
            raise RuntimeError("Cannot auto-detect run_dir. Please pass --run_dir.")
    print("Using RUN_DIR:", run_dir)

    user_ratios = None
    if args.ratios.strip() != "":
        user_ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip() != ""]

    seed_dirs = discover_seed_dirs(run_dir)
    print("Discovered seeds:")
    for d in seed_dirs:
        print(" -", d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    test_loader = DataLoader(
        ImageFolderByPrefix(args.test_dir, test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    per_seed = {}  # seed_dir -> ratio -> (acc, f1)
    ratios_union = set()

    for sd in seed_dirs:
        ratios = list_ratios(sd, user_ratios=user_ratios)
        ratios_union.update(ratios)

        per_seed[sd] = {}
        for r in ratios:
            tier_dir = os.path.join(sd, f"ratio_{str(r).replace('.', '_')}")
            ckpt = os.path.join(tier_dir, "kan_best.pth")
            if not os.path.isfile(ckpt):
                continue
            acc, f1 = evaluate_ckpt(ckpt, test_loader, device, args.img_size)
            per_seed[sd][r] = (acc, f1)

    ratios_sorted = sorted(ratios_union)

    print("\n==================== TEST EVAL (per seed) ====================")
    for sd in seed_dirs:
        name = "primary" if sd == run_dir else os.path.basename(sd)
        for r in ratios_sorted:
            if r in per_seed.get(sd, {}):
                acc, f1 = per_seed[sd][r]
                print(f"{name:10s} | ratio={r:.2f} | TestAcc={acc:.2f}% | F1={f1:.4f}")

    print("\n==================== MEAN±STD SUMMARY (Test) ====================")
    mean_rows = []
    for r in ratios_sorted:
        accs = []
        f1s = []
        for sd in seed_dirs:
            if r in per_seed.get(sd, {}):
                a, f = per_seed[sd][r]
                accs.append(a)
                f1s.append(f)
        if len(accs) == 0:
            continue
        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs, ddof=1)) if len(accs) >= 2 else 0.0
        f1_mean = float(np.mean(f1s))
        f1_std = float(np.std(f1s, ddof=1)) if len(f1s) >= 2 else 0.0
        mean_rows.append((r, acc_mean, acc_std, f1_mean, f1_std, len(accs)))
        # Detailed line (keeps backward compatibility with earlier output)
        print(f"ratio={r:.2f} | Acc={acc_mean:.2f}±{acc_std:.2f} | F1={f1_mean:.4f}±{f1_std:.4f} | n={len(accs)}")

    # Accuracy-only table for quick copy/paste to logs or paper drafts.
    if len(mean_rows) > 0:
        print("\nAcc mean±std by ratio:")
        print("ratio\tacc_mean\tacc_std\tn")
        for r, am, ast, _fm, _fst, n in mean_rows:
            print(f"{r:.2f}\t{am:.2f}\t{ast:.2f}\t{n}")

    out_csv = os.path.join(run_dir, "test_eval_multiseed.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ratio,acc_mean,acc_std,f1_mean,f1_std,n\n")
        for r, am, ast, fm, fst, n in mean_rows:
            f.write(f"{r:.6f},{am:.6f},{ast:.6f},{fm:.6f},{fst:.6f},{n}\n")
    print("Saved:", out_csv)

    if len(mean_rows) > 0:
        ratios = [x[0] for x in mean_rows]
        acc_mean = np.array([x[1] for x in mean_rows], dtype=np.float32)
        acc_std = np.array([x[2] for x in mean_rows], dtype=np.float32)
        f1_mean = np.array([x[3] for x in mean_rows], dtype=np.float32)
        f1_std = np.array([x[4] for x in mean_rows], dtype=np.float32)

        plt.figure(figsize=(10, 6))
        plt.plot(ratios, acc_mean, marker="o", label="mean")
        plt.fill_between(ratios, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)
        plt.title("Test Accuracy vs Subset Ratio (mean±std)")
        plt.xlabel("Subset ratio")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        p = os.path.join(run_dir, "test_acc_mean_std.png")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        print("Saved:", p)

        plt.figure(figsize=(10, 6))
        plt.plot(ratios, f1_mean, marker="o", label="mean")
        plt.fill_between(ratios, f1_mean - f1_std, f1_mean + f1_std, alpha=0.2)
        plt.title("Test F1 vs Subset Ratio (mean±std)")
        plt.xlabel("Subset ratio")
        plt.ylabel("F1 (macro)")
        plt.grid(True)
        plt.legend()
        p = os.path.join(run_dir, "test_f1_mean_std.png")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        print("Saved:", p)

    if args.do_gradcam:
        primary_dir = run_dir
        r = float(args.gradcam_ratio)
        tier_dir = os.path.join(primary_dir, f"ratio_{str(r).replace('.', '_')}")
        ckpt = os.path.join(tier_dir, "kan_best.pth")
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"GradCAM ckpt not found: {ckpt}")
        out_dir = save_gradcam_examples(
            run_dir=run_dir,
            ckpt_path=ckpt,
            ratio=r,
            test_dir=args.test_dir,
            img_size=args.img_size,
            per_class=int(args.gradcam_per_class),
            sample_seed=int(args.gradcam_seed),
            device=device,
        )
        print("Saved GradCAM to:", out_dir)


if __name__ == "__main__":
    main()
