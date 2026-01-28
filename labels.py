import os
from glob import glob
import traceback

# -------------------------- 核心配置（按实际路径填写）--------------------------
ROOT_PATH = r"D:\MyDesktop\banana\val"  # 分类文件夹上级路径
SERIAL_DIGITS = 4  # 序号位数（4位→0001）
IMAGE_EXTENSIONS = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
SORT_BY = "name"  # 排序方式："name"（按原名称）或 "time"（按修改时间）
OVERWRITE = False  # 不覆盖同名文件（安全第一）
MARK_FILE = ".已完成重命名.txt"  # 防重复运行标记文件
# --------------------------------------------------------------------------------

categories = [("overripe", "o"), ("ripe", "r")]


def is_valid_image(file_path):
    """验证文件是否为有效图片"""
    try:
        if os.path.getsize(file_path) == 0:
            return False, "空文件"
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix not in (".jpg", ".png", ".jpeg"):
            return False, f"不支持格式：{suffix}"
        return True, "有效"
    except Exception as e:
        return False, f"无效：{str(e)}"


def rename_images_by_category(folder_name, prefix):
    """按分类重命名（防重复+验证结果）"""
    folder_path = os.path.join(ROOT_PATH, folder_name)
    mark_file_path = os.path.join(folder_path, MARK_FILE)

    # 1. 检查是否已重命名过（防重复运行）
    if os.path.exists(mark_file_path):
        with open(mark_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"\n {folder_name}：已完成重命名（上次结果：{content}），跳过！")
        return 0, 0

    if not os.path.exists(folder_path):
        print(f"\n {folder_name}：文件夹不存在，跳过")
        return 0, 0

    # 2. 获取并排序图片
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        img_paths.extend(glob(os.path.join(folder_path, ext)))
    img_paths = list(set(img_paths))  # 去重
    if len(img_paths) == 0:
        print(f"\n📭 {folder_name}：无图片，跳过")
        return 0, 0

    if SORT_BY == "time":
        img_paths.sort(key=lambda x: os.path.getmtime(x))
    else:
        img_paths.sort()

    success_count = 0
    fail_count = 0
    success_log = []  # 记录成功的文件（用于验证）

    print(f"\n{'=' * 50}")
    print(f"处理：{folder_name}（共{len(img_paths)}张）")
    print(f"{'=' * 50}")

    for idx, old_path in enumerate(img_paths, 1):
        old_filename = os.path.basename(old_path)
        new_serial = str(idx).zfill(SERIAL_DIGITS)
        old_suffix = os.path.splitext(old_path)[1].lower()
        new_filename = f"{prefix}{new_serial}{old_suffix}"
        new_path = os.path.join(folder_path, new_filename)

        # 3. 验证文件有效性
        is_valid, valid_msg = is_valid_image(old_path)
        if not is_valid:
            print(f"跳过：{old_filename} → {valid_msg}")
            fail_count += 1
            continue

        # 4. 处理同名冲突
        if os.path.exists(new_path) and old_path != new_path:
            print(f" 失败：{old_filename} → {new_filename}（同名已存在）")
            fail_count += 1
            continue

        # 5. 执行重命名（确保真成功）
        try:
            os.rename(old_path, new_path)
            # 验证：重命名后文件是否存在
            if os.path.exists(new_path):
                print(f" 成功：{old_filename} → {new_filename}")
                success_count += 1
                success_log.append(new_filename)
            else:
                print(f" 失败：{old_filename} → {new_filename}（重命名后文件消失）")
                fail_count += 1
        except PermissionError:
            print(f" 失败：{old_filename} → {new_filename}（文件被占用/权限不足）")
            fail_count += 1
        except Exception as e:
            err_detail = traceback.format_exc().split("\n")[-2]
            print(f" 失败：{old_filename} → {new_filename}（错误：{err_detail}）")
            fail_count += 1

    # 6. 写入标记文件（防重复运行）
    if success_count > 0:
        with open(mark_file_path, "w", encoding="utf-8") as f:
            f.write(f"成功{success_count}张，失败{fail_count}张，最后更新：{os.path.getctime(mark_file_path)}")

    # 7. 显示重命名后的前5个文件（验证结果）
    print(f"\n 重命名后前5个文件：")
    new_files = glob(os.path.join(folder_path, f"{prefix}*{old_suffix}"))[:5]
    for file in new_files:
        print(f"  - {os.path.basename(file)}")

    print(f"{'=' * 50}")
    return success_count, fail_count


# 主流程
total_success = 0
total_fail = 0

print("=" * 60)
print("图片编号连续排序工具（防复原版）")
print(f"配置：{ROOT_PATH} | 序号{SERIAL_DIGITS}位 | 防重复运行已开启")
print("=" * 60)

for folder_name, prefix in categories:
    success, fail = rename_images_by_category(folder_name, prefix)
    total_success += success
    total_fail += fail

# 最终统计+验证
print("\n" + "=" * 60)
print(" 最终结果：")
print(f"总成功：{total_success} 张")
print(f"总失败：{total_fail} 张")
print(f"💡 验证方法：")
print(f"  1. 打开文件夹：{ROOT_PATH} → 进入分类文件夹")
print(f"  2. 按名称排序，查看编号是否连续")
print(f"  3. 文件夹内有 {MARK_FILE} 表示已完成，不会重复运行")
print("=" * 60)
