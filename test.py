import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

from gradcam import GradCAM
from model.MobileNetV2KAN import MobileNetV2KAN

# ===== CONFIG =====
IMG_SIZE = 64
BATCH_SIZE = 64
NUM_CLASSES = 2

MODEL_PATH = "./model_result/subset_multi_7/ratio_0_2/kan_best.pth"
TEST_DIR = "./data/lychee/test1"
RESULT_DIR = "./test_result"
MAX_CAM_SAMPLES = 10

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "gradcam"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "error_samples"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir):
    paths, labels = [], []
    for name in os.listdir(data_dir):
        if not name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        if name[0].lower() == "u":
            label = 0
        elif name[0].lower() == "r":
            label = 1
        else:
            continue
        paths.append(os.path.join(data_dir, name))
        labels.append(label)
    return paths, labels


class TestDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx], self.paths[idx]


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

paths, labels = load_data(TEST_DIR)
dataset = TestDataset(paths, labels, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


model = MobileNetV2KAN(input_size=IMG_SIZE, num_classes=NUM_CLASSES).to(device)
model.eval()

with torch.no_grad():
    _ = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("[INFO] Model loaded successfully.")


all_preds, all_labels, all_probs = [], [], []
cam_count = 0

for imgs, labels_batch, paths_batch in loader:
    imgs = imgs.to(device)
    labels_batch = labels_batch.to(device)

    with torch.no_grad():
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

    for i in range(len(imgs)):
        all_preds.append(int(preds[i].item()))
        all_labels.append(int(labels_batch[i].item()))
        all_probs.append(float(probs[i].item()))

        if preds[i] != labels_batch[i]:
            img_np = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            true_c = "u" if labels_batch[i].item() == 0 else "r"
            pred_c = "u" if preds[i].item() == 0 else "r"
            save_name = f"{true_c}_pred_{pred_c}_{os.path.basename(paths_batch[i])}"
            cv2.imwrite(
                os.path.join(RESULT_DIR, "error_samples", save_name),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            )

        if cam_count < MAX_CAM_SAMPLES:
            img_cam = imgs[i:i + 1].detach().clone().requires_grad_(True)
            gradcam = GradCAM(model, model.backbone[-1], IMG_SIZE)
            cam_map = gradcam.generate(img_cam, class_idx=int(preds[i].item()))
            gradcam.remove()
            del gradcam

            heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
            orig = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

            cv2.imwrite(
                os.path.join(RESULT_DIR, "gradcam", f"cam_{cam_count}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
            cam_count += 1
            model.zero_grad(set_to_none=True)


acc = np.mean(np.array(all_preds) == np.array(all_labels))
f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)

print("\n========== TEST RESULT ==========")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["unripe", "ripe"]))


fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(True)
plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"))
plt.close()

precision, recall, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

plt.plot(recall, precision)
plt.title(f"PR Curve (AP={ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig(os.path.join(RESULT_DIR, "pr_curve.png"))
plt.close()

print("\n[INFO] Test artifacts saved to:", RESULT_DIR)
