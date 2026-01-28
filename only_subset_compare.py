import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from model.MobileNetV2KAN import MobileNetV2KAN


def get_unique_result_folder(base="model_result", prefix="subset_multi"):
    os.makedirs(base, exist_ok=True)
    idx = 1
    while True:
        folder = os.path.join(base, f"{prefix}_{idx}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        idx += 1


def safe_softmax_prob(outputs: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(outputs, dim=1)[:, 1]
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return probs


def load_all_paths(data_dir, seed=42):
    rng = random.Random(seed)
    u_imgs, r_imgs = [], []
    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(data_dir, img_name)
        head = img_name[0].lower()
        if head == "u":
            u_imgs.append((p, 0))
        elif head == "r":
            r_imgs.append((p, 1))
    rng.shuffle(u_imgs)
    rng.shuffle(r_imgs)
    return u_imgs, r_imgs


def make_balanced_subset(u_imgs, r_imgs, ratio, seed=42):
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


class LycheeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def evaluate(model, val_loader, device, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(safe_softmax_prob(outputs).detach().cpu().numpy())

    val_acc = 100.0 * correct / total if total > 0 else 0.0
    val_f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0

    try:
        val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.5
    except Exception:
        val_auc = 0.5

    return val_loss / max(1, len(val_loader)), val_acc, val_f1, val_auc


def save_all_plots_single(history, model, val_loader, device, out_dir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, history["f1"], label="F1")
    ax3.set_title("F1 Score Curve")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1")
    ax3.grid(True)
    ax3.legend()

    fig.savefig(os.path.join(out_dir, "training_plot.png"), bbox_inches="tight")
    plt.close(fig)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(safe_softmax_prob(outputs).detach().cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xticks([0, 1], ["unripe", "ripe"])
    plt.yticks([0, 1], ["unripe", "ripe"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    try:
        if len(set(all_labels)) == 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
    except Exception:
        fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.5
    except Exception:
        precision, recall, ap = [1, 0], [0, 1], 0.5

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"PR Curve (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), bbox_inches="tight")
    plt.close()


def train_one_tier(model, name, train_loader, val_loader, epochs, patience, save_path, eta_min):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(eta_min)
    )

    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "f1": []
    }

    best_val_acc = 0.0
    no_gain = 0

    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"{name} [Epoch {ep + 1}/{epochs}]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        scheduler.step()
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], float(eta_min))

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        val_loss, val_acc, f1, val_auc = evaluate(model, val_loader, device, criterion)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        hist["f1"].append(f1)

        print(f"\nEpoch {ep + 1}/{epochs}")
        print(f" Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {f1:.4f} | AUC: {val_auc:.4f}")
        print(f" LR: {optimizer.param_groups[0]['lr']:.6e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_gain = 0
            torch.save(model.state_dict(), save_path)
            print(f"[Model Saved] {save_path}")
        else:
            no_gain += 1
            print(f"No improvement: {no_gain}/{patience}")
            if no_gain >= patience:
                break

    return hist


def plot_compare_curves(all_hist, out_dir, zoom_epochs=3, acc_inset_threshold=0.5):
    finals = []
    for _, h in all_hist.items():
        finals.append(h["val_acc"][-1] if len(h["val_acc"]) > 0 else 0.0)
    need_inset = (max(finals) - min(finals)) <= float(acc_inset_threshold) if len(finals) > 1 else False

    def _add_inset(ax, x_list, y_list, loc, title):
        axins = inset_axes(ax, width="36%", height="36%", loc=loc, borderpad=1.2)
        for x, y in zip(x_list, y_list):
            axins.plot(x, y)
        axins.set_title(title, fontsize=9)
        axins.grid(True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, h in all_hist.items():
        x = list(range(1, len(h["train_loss"]) + 1))
        ax.plot(x, h["train_loss"], label=f"{tier}-train")
        ax.plot(x, h["val_loss"], linestyle="--", label=f"{tier}-val")

    ax.set_title("Loss Curve Comparison (Different Subset Sizes)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if need_inset:
        xs, ys = [], []
        for _, h in all_hist.items():
            k = min(int(zoom_epochs), len(h["val_loss"]))
            xz = list(range(len(h["val_loss"]) - k + 1, len(h["val_loss"]) + 1))
            xs.append(xz)
            ys.append(h["val_loss"][-k:])
        _add_inset(ax, xs, ys, loc="upper left", title=f"Last {zoom_epochs} Epochs (Val Loss)")

    fig.savefig(os.path.join(out_dir, "loss_compare.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, h in all_hist.items():
        x = list(range(1, len(h["val_acc"]) + 1))
        ax.plot(x, h["val_acc"], label=tier)

    ax.set_title("Validation Accuracy Comparison (Different Subset Sizes)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if need_inset:
        xs, ys = [], []
        for _, h in all_hist.items():
            k = min(int(zoom_epochs), len(h["val_acc"]))
            xz = list(range(len(h["val_acc"]) - k + 1, len(h["val_acc"]) + 1))
            xs.append(xz)
            ys.append(h["val_acc"][-k:])
        _add_inset(ax, xs, ys, loc="lower left", title=f"Last {zoom_epochs} Epochs (Val Acc)")

    fig.savefig(os.path.join(out_dir, "val_acc_compare.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, h in all_hist.items():
        x = list(range(1, len(h["f1"]) + 1))
        ax.plot(x, h["f1"], label=tier)

    ax.set_title("F1 Score Comparison (Different Subset Sizes)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if need_inset:
        xs, ys = [], []
        for _, h in all_hist.items():
            k = min(int(zoom_epochs), len(h["f1"]))
            xz = list(range(len(h["f1"]) - k + 1, len(h["f1"]) + 1))
            xs.append(xz)
            ys.append(h["f1"][-k:])
        _add_inset(ax, xs, ys, loc="lower left", title=f"Last {zoom_epochs} Epochs (F1)")

    fig.savefig(os.path.join(out_dir, "f1_compare.png"), bbox_inches="tight")
    plt.close(fig)


def _run_one_seed(seed: int, seed_dir: str, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_u, train_r = load_all_paths(args.train_dir, seed=seed)
    val_u, val_r = load_all_paths(args.val_dir, seed=seed)

    val_combined = []
    min_len = min(len(val_u), len(val_r))
    for i in range(min_len):
        val_combined.append(val_u[i])
        val_combined.append(val_r[i])
    extra = val_u[min_len:] if len(val_u) > min_len else val_r[min_len:]
    val_combined.extend(extra)
    val_paths, val_labels = zip(*val_combined) if len(val_combined) > 0 else ([], [])

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    val_loader = DataLoader(
        LycheeDataset(list(val_paths), list(val_labels), val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    tiers = [float(x.strip()) for x in args.tiers.split(",") if x.strip() != ""]
    all_hist = {}

    for i, ratio in enumerate(tiers):
        tier_name = f"ratio_{str(ratio).replace('.', '_')}"
        tier_dir = os.path.join(seed_dir, tier_name)
        os.makedirs(tier_dir, exist_ok=True)

        train_paths, train_labels, n_each = make_balanced_subset(
            train_u, train_r, ratio=ratio, seed=seed + i * 13
        )

        with open(os.path.join(tier_dir, "subset_train_list.txt"), "w", encoding="utf-8") as f:
            for p, y in zip(train_paths, train_labels):
                f.write(f"{p}\t{y}\n")

        print(f"\n======= Seed={seed} | Tier: {tier_name} | per_class={n_each} | total={len(train_paths)} =======")

        train_loader = DataLoader(
            LycheeDataset(train_paths, train_labels, train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

        model = MobileNetV2KAN(input_size=args.image_size)
        save_path = os.path.join(tier_dir, "kan_best.pth")

        hist = train_one_tier(
            model=model,
            name=f"{tier_name}-seed{seed}",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            save_path=save_path,
            eta_min=args.eta_min
        )

        all_hist[tier_name] = hist

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.to(device).eval()
        save_all_plots_single(hist, model, val_loader, device, tier_dir)

    plot_compare_curves(
        all_hist,
        out_dir=seed_dir,
        zoom_epochs=args.zoom_epochs,
        acc_inset_threshold=args.acc_inset_threshold
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/lychee/train")
    parser.add_argument("--val_dir", type=str, default="./data/lychee/val")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tiers", type=str, default="0.1,0.2,0.4,0.6")
    parser.add_argument("--eta_min", type=float, default=3e-4)

    parser.add_argument("--acc_inset_threshold", type=float, default=0.5)
    parser.add_argument("--zoom_epochs", type=int, default=3)

    parser.add_argument("--num_runs", type=int, default=10)

    args = parser.parse_args()

    result_root = get_unique_result_folder(base="model_result", prefix="subset_multi")
    print("Results saved to:", result_root)

    num_runs = max(1, int(args.num_runs))
    seeds = [int(args.seed) + i for i in range(num_runs)]

    if len(seeds) > 1:
        os.makedirs(os.path.join(result_root, "seeds"), exist_ok=True)

    for i, s in enumerate(seeds):
        if i == 0:
            seed_dir = result_root
        else:
            seed_dir = os.path.join(result_root, "seeds", f"seed_{s}")
            os.makedirs(seed_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"[RUN {i + 1}/{len(seeds)}] Seed = {s}")
        print("=" * 70)
        _run_one_seed(s, seed_dir, args)

    print("\nDONE. All results saved to:", result_root)


if __name__ == "__main__":
    main()
