import os
import cv2
import numpy as np
from PIL import Image
from glob import glob
import shutil

# -------------------------- 核心配置 --------------------------
IMAGES_PATH = r"D:\MyDesktop\banana\val\images"  # 图片文件夹路径
LABELS_PATH = r"D:\MyDesktop\数字人检测系统汇总\crops_Kan\data\lychee\val\labels"  # 标注txt文件夹路径
OUTPUT_PATH = r"D:\MyDesktop\数字人检测系统汇总\crops_Kan\data\lychee\val_images"  # 裁剪图输出路径
SERIAL_DIGITS = 4  # 序号位数（4位→0001，可改3/5位）
IMAGE_EXTENSIONS = ("*.jpg", "*.png", "*.jpeg")  # 支持的图片格式
SIMILARITY_THRESHOLD = 0.95  # 相似阈值（0-1，越大越严格）
MIN_CROP_SIZE = 10  # 最小裁剪尺寸（宽/高需≥10像素，避免无效图片）


# --------------------------------------------------------------------------------

# ========== 初始化验证 ==========
def validate_paths():
    """验证输入输出路径的有效性"""
    if not os.path.exists(IMAGES_PATH):
        raise ValueError(f"图片文件夹不存在！路径：{IMAGES_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise ValueError(f"标注文件夹不存在！路径：{LABELS_PATH}")

    # 验证输出路径可写
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        # 测试创建临时文件（验证权限）
        test_file = os.path.join(OUTPUT_PATH, ".test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise PermissionError(f"输出路径无写入权限！路径：{OUTPUT_PATH}，错误：{e}")


# 执行路径验证（提前报错，避免后续白跑）
try:
    validate_paths()
except (ValueError, PermissionError) as e:
    print(f"初始化失败：{e}")
    exit(1)

# 自动创建分类输出文件夹
immature_dir = os.path.join(OUTPUT_PATH, "不成熟_u系列")
mature_dir = os.path.join(OUTPUT_PATH, "成熟_r系列")
os.makedirs(immature_dir, exist_ok=True)
os.makedirs(mature_dir, exist_ok=True)

# 序号计数器（按类别独立递增）
serial_counts = {0: 1, 1: 1}
save_failed_count = 0  # 保存失败计数器


# ========== 核心工具函数 ==========
def get_label_path(image_path):
    """根据图片路径匹配对应标注文件"""
    img_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(LABELS_PATH, f"{img_name_no_ext}.txt")
    return label_path if os.path.exists(label_path) else None


def yolo2pixel(box, img_w, img_h):
    """YOLO归一化坐标→像素坐标（严格边界修正）"""
    x_center, y_center, w, h = box
    # 计算坐标（避免负数）
    x1 = max(0, int((x_center - w / 2) * img_w))
    y1 = max(0, int((y_center - h / 2) * img_h))
    x2 = min(img_w, int((x_center + w / 2) * img_w))
    y2 = min(img_h, int((y_center + h / 2) * img_h))
    return (x1, y1, x2, y2)


def calculate_image_histogram(img_path):
    """计算图片直方图特征（用于相似度比较）"""
    try:
        with Image.open(img_path) as img_pil:
            img_pil = img_pil.convert("L").resize((64, 64))  # 灰度图+缩小，提升效率
            img_np = np.array(img_pil)
            hist = cv2.calcHist([img_np], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
    except Exception as e:
        print(f"警告：{os.path.basename(img_path)} 无法计算特征（{e}）")
        return None


def remove_similar_images(folder_path, threshold=SIMILARITY_THRESHOLD):
    """删除文件夹内的相似图片"""
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        img_paths.extend(glob(os.path.join(folder_path, ext)))

    if len(img_paths) <= 1:
        print(f"\n{os.path.basename(folder_path)}：仅{len(img_paths)}张图片，无需去重")
        return 0

    print(f"\n开始对 {os.path.basename(folder_path)} 去重（相似度阈值：{threshold}）...")
    hist_list = []
    keep_paths = []
    delete_count = 0

    for img_path in img_paths:
        hist = calculate_image_histogram(img_path)
        if hist is None:
            continue

        is_similar = False
        for keep_hist in hist_list:
            similarity = cv2.compareHist(hist, keep_hist, cv2.HISTCMP_CORREL)
            if similarity >= threshold:
                is_similar = True
                break

        if is_similar:
            os.remove(img_path)
            print(f"删除相似图片：{os.path.basename(img_path)}")
            delete_count += 1
        else:
            hist_list.append(hist)
            keep_paths.append(img_path)

    print(f"{os.path.basename(folder_path)} 去重完成：删除{delete_count}张，保留{len(keep_paths)}张")
    return delete_count


# ========== 主裁剪逻辑 ==========
# 读取所有图片路径
image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob(os.path.join(IMAGES_PATH, ext)))

if len(image_paths) == 0:
    print(f"未找到图片！检查 IMAGES_PATH：{IMAGES_PATH}")
    exit(1)

print(f"找到 {len(image_paths)} 张图片，开始裁剪（支持多行动标注）...")

for img_idx, img_path in enumerate(image_paths, 1):
    # 读取图片（PIL兼容格式）
    try:
        with Image.open(img_path) as img_pil:
            img_pil = img_pil.convert("RGB")  # 统一RGB格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"跳过：{os.path.basename(img_path)} 无法读取（{e}）")
        continue

    img_h, img_w = img.shape[:2]
    label_path = get_label_path(img_path)

    if label_path is None:
        print(f"跳过：{os.path.basename(img_path)} 无对应标注文件")
        continue

    # 读取标注（过滤空行）
    with open(label_path, "r", encoding="utf-8") as f:
        annotations = [line.strip() for line in f if line.strip()]

    if len(annotations) == 0:
        print(f"跳过：{os.path.basename(label_path)} 无有效标注")
        continue

    # 逐行处理标注（一行→一张裁剪图）
    for ann_idx, ann in enumerate(annotations, 1):
        parts = ann.split()
        if len(parts) != 5:
            print(f"跳过：{os.path.basename(label_path)} 第{ann_idx}行格式错误（需5个字段）")
            continue

        try:
            cls = int(parts[0])
            box = list(map(float, parts[1:5]))
        except ValueError:
            print(f"跳过：{os.path.basename(label_path)} 第{ann_idx}行数据类型错误")
            continue

        if cls not in (0, 1):
            print(f"跳过：{os.path.basename(label_path)} 第{ann_idx}行无效类别（仅支持0/1）")
            continue

        # 转换坐标并裁剪
        x1, y1, x2, y2 = yolo2pixel(box, img_w, img_h)
        crop_w = x2 - x1
        crop_h = y2 - y1

        # 验证裁剪尺寸（避免无效图片）
        if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
            print(f"跳过：标注{ann_idx} 裁剪尺寸过小（{crop_w}x{crop_h}，需≥{MIN_CROP_SIZE}像素）")
            continue

        cropped_img = img[y1:y2, x1:x2]

        # 生成保存路径
        serial = str(serial_counts[cls]).zfill(SERIAL_DIGITS)
        img_suffix = os.path.splitext(img_path)[1].lower()  # 统一后缀小写
        if img_suffix not in (".jpg", ".png", ".jpeg"):
            img_suffix = ".jpg"  # 强制转换为常见格式

        if cls == 0:
            output_filename = f"u{serial}{img_suffix}"
            output_path = os.path.join(immature_dir, output_filename)
        else:
            output_filename = f"r{serial}{img_suffix}"
            output_path = os.path.join(mature_dir, output_filename)

        # 保存图片（添加异常捕获和验证）
        try:
            # 强制保存为JPG/PNG（避免格式兼容问题）
            if img_suffix == ".jpg":
                # JPG不支持透明通道，压缩质量95
                success = cv2.imwrite(output_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                success = cv2.imwrite(output_path, cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            if not success:
                # 保存失败时，尝试用PIL保存兜底
                Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)).save(output_path)
                print(f"进度：{img_idx}/{len(image_paths)} | 标注{ann_idx} | 修复保存：{output_filename}")
            else:
                print(f"进度：{img_idx}/{len(image_paths)} | 标注{ann_idx} | 成功保存：{output_filename}")

            serial_counts[cls] += 1
        except Exception as e:
            print(f"失败：{os.path.basename(img_path)} 标注{ann_idx} 保存失败（{e}）")
            save_failed_count += 1

# ========== 统计结果 ==========
total_cropped = (serial_counts[0] - 1) + (serial_counts[1] - 1)
print("\n" + "=" * 60)
print(f"裁剪完成！基础统计：")
print(f"不成熟系列（标签0）：{serial_counts[0] - 1} 张 → {immature_dir}")
print(f"成熟系列（标签1）：{serial_counts[1] - 1} 张 → {mature_dir}")
print(f"总裁剪图数：{total_cropped} 张")
print(f"保存失败：{save_failed_count} 张")
print("=" * 60)

# ========== 相似图片去重（可选） ==========
while True:
    choice = input("\n是否删除相似图片？（y=是/n=否，默认n）：").strip().lower()
    if choice in ("y", "n", ""):
        break
    print("输入无效！请输入 y 或 n")

if choice == "y":
    delete_immature = remove_similar_images(immature_dir)
    delete_mature = remove_similar_images(mature_dir)
    total_deleted = delete_immature + delete_mature

    # 去重后最终统计
    keep_immature = len(glob(os.path.join(immature_dir, "*.*")))
    keep_mature = len(glob(os.path.join(mature_dir, "*.*")))
    total_keep = keep_immature + keep_mature

    print("\n" + "=" * 60)
    print(f"去重完成！最终统计：")
    print(f"不成熟系列：保留{keep_immature}张（删除{delete_immature}张相似）")
    print(f"成熟系列：保留{keep_mature}张（删除{delete_mature}张相似）")
    print(f"最终总图片数：{total_keep} 张")
    print("=" * 60)
else:
    print("\n已跳过相似图片去重功能")