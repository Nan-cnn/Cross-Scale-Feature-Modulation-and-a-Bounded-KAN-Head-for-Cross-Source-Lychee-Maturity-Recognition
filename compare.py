import csv
import os
import random
import time
import math
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from gradcam import GradCAM
from model.MobileNetV2KAN import MobileNetV2KAN


# ==================== AUTO RESULT FOLDER ==================== #
def get_next_compare_folder(base="compare_result/compare_result"):
    idx = 1
    while True:
        folder = f"{base}_{idx}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return folder
        idx += 1


# ==================== CONFIG (aligned with subset_compare.py) ==================== #
ZOOM_EPOCHS = 3
EPOCHS = 20
BATCH_SIZE = 64
IMG_SIZE = 64

LR = 5e-3
LR_FLOOR = 3e-4

RESULT_ROOT = get_next_compare_folder("compare_result/compare_result")

TRAIN_DIR = "./data/lychee/train"
VAL_DIR = "./data/lychee/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ================= MODEL STATS ================= #
def _count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _estimate_param_size_mb(param_count: int, bytes_per_param: int = 4) -> float:
    return float(param_count * bytes_per_param) / (1024.0 ** 2)

def _state_dict_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return float(buf.getbuffer().nbytes) / (1024.0 ** 2)

def _file_size_mb(path: str) -> float:
    try:
        return float(os.path.getsize(path)) / (1024.0 ** 2)
    except OSError:
        return 0.0

# ==================== LOAD DATA ==================== #
def load_data(data_dir):
    u_imgs, r_imgs = [], []

    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(data_dir, img_name)

        if img_name[0].lower() == "u":
            u_imgs.append((path, 0))
        elif img_name[0].lower() == "r":
            r_imgs.append((path, 1))

    random.shuffle(u_imgs)
    random.shuffle(r_imgs)

    combined = []
    for u, r in zip(u_imgs, r_imgs):
        combined.append(u)
        combined.append(r)

    longer = u_imgs if len(u_imgs) > len(r_imgs) else r_imgs
    combined.extend(longer[len(combined) // 2:])

    paths, labels = zip(*combined)
    return list(paths), list(labels)


class LycheeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ==================== TRANSFORM (aligned with subset_compare.py) ==================== #
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
    transforms.RandomAffine(8, translate=(0.03, 0.03), fill=128),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.5),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


train_paths, train_labels = load_data(TRAIN_DIR)
val_paths, val_labels = load_data(VAL_DIR)

train_set = LycheeDataset(train_paths, train_labels, train_transform)
val_set = LycheeDataset(val_paths, val_labels, val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


def _safe_ylim(values, pad_ratio=0.08):
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return vmin - 1.0, vmax + 1.0
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


# ==================== TRAIN FUNCTION ==================== #
def train_model(model, name):
    model = model.to(device)

    # Loss (label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (SGD)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Scheduler (CosineAnnealingLR, with lr floor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, EPOCHS * 3),
        eta_min=float(LR_FLOOR)
    )

    train_loss_hist, val_loss_hist = [], []
    val_acc_hist, f1_hist = [], []
    epoch_times = []

    best = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        correct, total = 0, 0
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"{name} Epoch {epoch + 1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        scheduler.step()
        if optimizer.param_groups[0]["lr"] < LR_FLOOR:
            optimizer.param_groups[0]["lr"] = LR_FLOOR

        train_loss = running_loss / len(train_loader)
        train_loss_hist.append(train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        val_running_loss = 0.0
        f1_sum = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, pred = outputs.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

                f1_sum.append(
                    f1_score(labels.cpu().numpy(), pred.cpu().numpy(), zero_division=0)
                )

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct / total
        f1_mean = sum(f1_sum) / len(f1_sum)

        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        f1_hist.append(f1_mean)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"{name} | Epoch {epoch + 1}: "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"ValAcc={val_acc:.2f}% | "
            f"F1={f1_mean:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6e}"
        )

        if val_acc > best:
            best = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(RESULT_ROOT, f"{name}_best.pth")
            )

    acc_growth = (val_acc_hist[-1] - val_acc_hist[0]) / max(1, len(val_acc_hist) - 1)
    avg_time = sum(epoch_times) / len(epoch_times)

    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "acc": val_acc_hist,
        "f1": f1_hist,
        "acc_growth": acc_growth,
        "avg_time": avg_time
    }


# ==================== BUILD MODELS ==================== #
models_list = {
    "MobileNetV2KAN": MobileNetV2KAN(input_size=IMG_SIZE, num_classes=2),
    "MobileNetV2": models.mobilenet_v2(weights=None, num_classes=2),
    "ResNet50": models.resnet50(weights=None, num_classes=2),
    "VGG16": models.vgg16(weights=None),
    "EfficientNetB3": models.efficientnet_b3(weights=None),
}

# Make VGG16 / EfficientNetB3 output 2 classes
models_list["VGG16"].classifier[6] = nn.Linear(models_list["VGG16"].classifier[6].in_features, 2)
if isinstance(models_list["EfficientNetB3"].classifier, nn.Sequential) and len(models_list["EfficientNetB3"].classifier) >= 2:
    models_list["EfficientNetB3"].classifier[1] = nn.Linear(models_list["EfficientNetB3"].classifier[1].in_features, 2)
else:
    models_list["EfficientNetB3"].classifier = nn.Linear(getattr(models_list["EfficientNetB3"].classifier, "in_features", 1536), 2)



# ==================== MODEL STATS (print + csv) ==================== #
print("\n==================== MODEL STATS ====================")
stats_rows = []
for name, model in models_list.items():
    total_p, trainable_p = _count_parameters(model)
    param_mb = _estimate_param_size_mb(total_p)
    sd_mb = _state_dict_size_mb(model)
    stats_rows.append([name, total_p, trainable_p, f"{param_mb:.6f}", f"{sd_mb:.6f}"])
    print(f"{name:14s} | total={total_p:,} | trainable={trainable_p:,} | params≈{param_mb:.2f} MB | state_dict≈{sd_mb:.2f} MB")

stats_csv = os.path.join(RESULT_ROOT, "model_stats.csv")
with open(stats_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "total_params", "trainable_params", "param_size_mb", "state_dict_size_mb"])
    w.writerows(stats_rows)
print("Saved model stats:", stats_csv)




history = {}
for name, model in models_list.items():
    print(f"\n======= Training {name} =======")
    history[name] = train_model(model, name)


# ==================== PLOT COMPARISONS (aligned style) ==================== #
epochs = list(range(1, EPOCHS + 1))
zoom_epochs = epochs[-ZOOM_EPOCHS:]


# LOSS
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in history.items():
    ax.plot(epochs, r["train_loss"], label=f"{name}-train")
    ax.plot(epochs, r["val_loss"], linestyle="--", label=f"{name}-val")
ax.set_title("Loss Curve Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
fig.subplots_adjust(right=0.78)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

axins = inset_axes(ax, width="40%", height="40%", loc="upper left", borderpad=1.2)
all_zoom_vals = []
for name, r in history.items():
    zt = r["train_loss"][-ZOOM_EPOCHS:]
    zv = r["val_loss"][-ZOOM_EPOCHS:]
    axins.plot(zoom_epochs, zt)
    axins.plot(zoom_epochs, zv, linestyle="--")
    all_zoom_vals.extend(zt + zv)
axins.set_title(f"Last {ZOOM_EPOCHS} Epochs", fontsize=9)
axins.grid(True)
y0, y1 = _safe_ylim(all_zoom_vals)
axins.set_ylim(y0, y1)

plt.savefig(os.path.join(RESULT_ROOT, "loss_compare.png"), bbox_inches="tight")
plt.close()


# VAL ACC
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in history.items():
    ax.plot(epochs, r["acc"], label=name)
ax.set_title("Validation Accuracy Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.grid(True)
fig.subplots_adjust(right=0.78)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

axins = inset_axes(ax, width="40%", height="40%", loc="lower left", borderpad=1.2)
all_zoom_vals = []
for name, r in history.items():
    z = r["acc"][-ZOOM_EPOCHS:]
    axins.plot(zoom_epochs, z)
    all_zoom_vals.extend(z)
axins.set_title(f"Last {ZOOM_EPOCHS} Epochs", fontsize=9)
axins.grid(True)
y0, y1 = _safe_ylim(all_zoom_vals)
axins.set_ylim(y0, y1)

plt.savefig(os.path.join(RESULT_ROOT, "val_acc_compare.png"), bbox_inches="tight")
plt.close()


# F1
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in history.items():
    ax.plot(epochs, r["f1"], label=name)
ax.set_title("F1 Score Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.grid(True)
fig.subplots_adjust(right=0.78)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

axins = inset_axes(ax, width="40%", height="40%", loc="lower left", borderpad=1.2)
all_zoom_vals = []
for name, r in history.items():
    z = r["f1"][-ZOOM_EPOCHS:]
    axins.plot(zoom_epochs, z)
    all_zoom_vals.extend(z)
axins.set_title(f"Last {ZOOM_EPOCHS} Epochs", fontsize=9)
axins.grid(True)
y0, y1 = _safe_ylim(all_zoom_vals)
axins.set_ylim(y0, y1)

plt.savefig(os.path.join(RESULT_ROOT, "f1_compare.png"), bbox_inches="tight")
plt.close()


# ==================== GRAD-CAM ==================== #
print("\n======= Generating Grad-CAM =======")
cam_dir = os.path.join(RESULT_ROOT, "gradcam")
os.makedirs(cam_dir, exist_ok=True)

img, _ = val_set[0]
input_tensor = img.unsqueeze(0).to(device)
orig = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(cam_dir, "original.jpg"), orig)

for name, model in models_list.items():
    model.load_state_dict(
        torch.load(os.path.join(RESULT_ROOT, f"{name}_best.pth"), weights_only=True)
    )
    model.to(device).eval()

    if name == "MobileNetV2KAN":
        target_layer = model.backbone[-1]
    elif name == "MobileNetV2":
        target_layer = model.features[-1]
    elif name == "ResNet50":
        target_layer = model.layer4[-1].conv3
    elif name == "VGG16":
        target_layer = model.features[-1]
    else:  # EfficientNetB3
        target_layer = model.features[-1]

    cam = GradCAM(model, target_layer, IMG_SIZE).generate(input_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(os.path.join(cam_dir, f"{name}_gradcam.jpg"), overlay)


# ==================== FINAL SUMMARY ==================== #
print("\n========== MODEL SUMMARY ==========")
print(f"{'Model':15s} | {'FinalAcc':>9s} | {'AccGrowth':>10s} | {'AvgTime(s)':>11s}")
print("-" * 60)
for name, r in history.items():
    print(
        f"{name:15s} | "
        f"{r['acc'][-1]:9.2f} | "
        f"{r['acc_growth']:10.3f} | "
        f"{r['avg_time']:11.2f}"
    )

print("\nDONE. Results saved to:", RESULT_ROOT)
