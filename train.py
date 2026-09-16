import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)

from model.MobileNetV2KAN import MobileNetV2KAN

EPOCH = 30
BATCH_SIZE = 64
IMAGE_SIZE = 64
PATIENCE = 10

LR_FLOOR = 3e-4


def get_unique_result_folder(base="model_result", prefix="kan_result"):
    os.makedirs(base, exist_ok=True)
    index = 1
    while True:
        candidate = os.path.join(base, f"{prefix}_{index}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        index += 1


def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def load_data(data_dir):
    u_imgs, r_imgs = [], []

    for img_name in tqdm(os.listdir(data_dir), desc=f"Loading {data_dir}", unit="file"):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(data_dir, img_name)
        head = img_name[0].lower()

        if head == "u":
            u_imgs.append((img_path, 0))
        elif head == "r":
            r_imgs.append((img_path, 1))

    random.shuffle(u_imgs)
    random.shuffle(r_imgs)

    min_len = min(len(u_imgs), len(r_imgs))
    combined = []
    for i in range(min_len):
        combined.append(u_imgs[i])
        combined.append(r_imgs[i])

    extra = u_imgs[min_len:] if len(u_imgs) > min_len else r_imgs[min_len:]

    if len(extra) > 0 and len(combined) > 0:
        interval = max(1, len(combined) // len(extra))
        pos = interval
        for item in extra:
            if pos >= len(combined):
                combined.append(item)
            else:
                combined.insert(pos, item)
            pos += interval
    else:
        combined.extend(extra)

    paths, labels = zip(*combined) if combined else ([], [])
    return list(paths), list(labels)


class Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            all_probs.extend(probs.detach().cpu().numpy())

    val_acc = 100 * correct / total if total > 0 else 0.0
    val_f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) > 0 else 0.0

    try:
        val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.5
    except ValueError:
        val_auc = 0.5

    return val_loss / max(1, len(val_loader)), val_acc, val_f1, val_auc


def plot_training(train_losses, val_losses, train_acc, val_acc, f1s,
                  model, val_loader, device, result_folder):
    base_plot_path = get_unique_path(os.path.join(result_folder, "training_plot.png"))
    base_cm_path = get_unique_path(os.path.join(result_folder, "confusion_matrix.png"))
    base_roc_path = get_unique_path(os.path.join(result_folder, "roc_curve.png"))
    base_pr_path = get_unique_path(os.path.join(result_folder, "pr_curve.png"))

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1s, label="F1")
    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(base_plot_path)
    plt.close()
    print(f"[Plot Saved] {base_plot_path}")

    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            all_probs.extend(probs.detach().cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xticks([0, 1], ["unripe", "ripe"])
    plt.yticks([0, 1], ["unripe", "ripe"])
    plt.title("Confusion Matrix")
    plt.savefig(base_cm_path)
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
    plt.savefig(base_roc_path)
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
    plt.savefig(base_pr_path)
    plt.close()


def train_model(model, train_loader, val_loader, num_epochs, save_path, patience, result_folder):
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
        T_max=max(1, num_epochs * 3),
        eta_min=LR_FLOOR
    )

    train_losses, val_losses, train_accs, val_accs, f1_scores = [], [], [], [], []
    best_val_acc = 0.0
    no_gain = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{num_epochs}]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        for pg in optimizer.param_groups:
            if pg["lr"] < LR_FLOOR:
                pg["lr"] = LR_FLOOR

        train_loss = running_loss / max(1, len(train_loader))
        train_accuracy = 100 * correct / total if total > 0 else 0.0

        val_loss, val_accuracy, f1, val_auc = evaluate(model, val_loader, device, criterion)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f" Train Loss: {train_loss:.4f} | Acc: {train_accuracy:.2f}%")
        print(f" Val   Loss: {val_loss:.4f} | Acc: {val_accuracy:.2f}% | F1: {f1:.4f} | AUC: {val_auc:.4f}")
        print(f" LR: {optimizer.param_groups[0]['lr']:.5f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        f1_scores.append(f1)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_gain = 0
            torch.save(model.state_dict(), save_path)
            print(f"[Model Saved] {save_path}")
        else:
            no_gain += 1
            print(f"No improvement: {no_gain}/{patience}")
            if no_gain >= patience:
                break

    plot_training(
        train_losses, val_losses, train_accs, val_accs, f1_scores,
        model, val_loader, device, result_folder
    )


if __name__ == "__main__":
    print("[INFO] Starting training on Lychee Dataset")

    # ====== 训练增强 ======
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
        transforms.RandomAffine(8, translate=(0.03, 0.03), fill=128),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.5),
    ])

    # ====== 验证集 ======
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dir = "./data/lychee/train"
    val_dir = "./data/lychee/val"

    train_paths, train_labels = load_data(train_dir)
    val_paths, val_labels = load_data(val_dir)

    train_dataset = Dataset(train_paths, train_labels, train_transform)
    val_dataset = Dataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    result_folder = get_unique_result_folder()
    model_path = os.path.join(result_folder, "kan_best.pth")

    model = MobileNetV2KAN(input_size=IMAGE_SIZE)
    train_model(model, train_loader, val_loader, EPOCH, model_path, PATIENCE, result_folder)
