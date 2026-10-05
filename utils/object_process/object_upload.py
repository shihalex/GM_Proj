#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path


# ==================== 參數 ====================

ROOT = Path("./Processed_Videos/")
CLEAN_LOG_NAME = "clean.txt"
UPLOAD_LOG_NAME = "upload.txt"
UPLOAD_ROOT = ROOT / "Upload"

TARGET_NUM = 20
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ==================== 完成紀錄工具 ====================

def read_trip_log(log_path: Path) -> set:
    """
    讀取旅程紀錄。每一行代表一個旅程資料夾名稱。
    """
    if not log_path.exists():
        return set()

    trips = set()
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                name = line.strip()
                if name:
                    trips.add(name)
    except Exception as e:
        print(f"⚠️ 無法讀取紀錄檔: {log_path}\n   錯誤: {e}")

    return trips


def mark_trip_as_completed(log_path: Path, trip_name: str):
    """
    將成功完成 Upload 處理的旅程加入 upload.txt。
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{trip_name}\n")
    except Exception as e:
        raise RuntimeError(f"無法寫入完成紀錄 {log_path}: {e}")


# ==================== 影像工具 ====================

def get_sorted_images(folder: Path):
    """
    取得標誌資料夾中的影像並依照檔名排序。
    支援：1.jpg, frame_1.png, FILE260101-12000000.png 等格式。
    """
    images = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMG_EXTS]

    def sort_key(path: Path):
        name = path.stem
        # 1.jpg、2.jpg ...
        if name.isdigit():
            return (0, int(name))
        # frame_1.png ...
        if name.startswith("frame_"):
            try:
                return (1, int(name.replace("frame_", "")))
            except ValueError:
                pass
        # FILEYYMMDD-HHMMSSxx.png
        return (2, name)

    return sorted(images, key=sort_key)


# ==================== 選擇保留影像 ====================

def select_keep_indices(total: int, target: int = TARGET_NUM):
    """
    決定需要保留哪些影像 index。
    <= target：全部保留。
    > target：第一張 + 最後一張固定保留，中間等間距取 target - 2 張。
    """
    if total <= target:
        return list(range(total))

    if target < 2:
        raise ValueError("target 必須至少為 2")

    first_idx = 0
    last_idx = total - 1
    middle_count = target - 2

    # 中間位置等距取樣
    middle_indices = []
    for i in range(middle_count):
        position = (i + 1) * (total - 1) / (middle_count + 1)
        idx = max(1, min(total - 2, round(position)))
        middle_indices.append(idx)

    keep_indices = sorted(set([first_idx, *middle_indices, last_idx]))

    # 防止 round 造成 index 重複導致數量不足 target
    if len(keep_indices) < target:
        selected = set(keep_indices)
        for idx in range(1, total - 1):
            if idx not in selected:
                selected.add(idx)
                if len(selected) == target:
                    break
        keep_indices = sorted(selected)

    return keep_indices


# ==================== 標誌影像下採樣 ====================

def downsample_sign_folder(folder: Path, target: int = TARGET_NUM):
    """
    單一標誌資料夾最多留下 target 張影像。不重新命名、重新編號或修改 CSV。
    """
    images = get_sorted_images(folder)
    total = len(images)
    print(f"   🪧 {folder.name}: {total} 張", end="")

    if total <= target:
        print(f" → 保留全部 {total} 張")
        return total

    # 選擇保留名單並刪除未選中的檔案
    keep_indices = select_keep_indices(total, target)
    keep_paths = {images[idx] for idx in keep_indices}

    deleted_count = 0
    for image in images:
        if image not in keep_paths:
            image.unlink()
            deleted_count += 1

    remaining = get_sorted_images(folder)
    remaining_count = len(remaining)

    if remaining_count > target:
        raise RuntimeError(f"{folder} 下採樣後仍有 {remaining_count} 張影像")

    print(f" → 刪除 {deleted_count} 張 → 保留 {remaining_count} 張")
    return remaining_count


# ==================== 判斷標誌資料夾 ====================

def get_sign_folders(frames_dir: Path):
    """
    找出 frames/ 底下的標誌資料夾（格式：{track_id}_{class_name}）。
    """
    sign_folders = []
    for path in frames_dir.iterdir():
        if not path.is_dir() or path.name.startswith("_yolo_"):
            continue
        first_part = path.name.split("_", 1)[0]
        if first_part.isdigit():
            sign_folders.append(path)

    sign_folders.sort(key=lambda p: int(p.name.split("_", 1)[0]))
    return sign_folders


# ==================== 複製旅程到 Upload ====================

def copy_trip_to_upload(trip_dir: Path, upload_root: Path, sign_folders: list):
    """
    僅複製以下內容到 Upload/{旅程名稱}/frames/：
        1. frames/detect.csv
        2. frames/ 底下的所有標誌資料夾
    排除 detect.mp4、原始影片、暫存檔等大型檔案。
    """
    source_frames = trip_dir / "frames"
    source_detect_csv = source_frames / "detect.csv"
    destination = upload_root / trip_dir.name
    destination_frames = destination / "frames"

    # 清除殘留未完成的目錄
    if destination.exists():
        print(f"⚠️ Upload 已存在未紀錄旅程，重新建立: {destination}")
        shutil.rmtree(destination)

    destination_frames.mkdir(parents=True, exist_ok=True)

    # 複製 detect.csv
    shutil.copy2(source_detect_csv, destination_frames / "detect.csv")
    print("   ✔ 複製 detect.csv")

    # 複製標誌資料夾
    for folder in sign_folders:
        destination_sign_folder = destination_frames / folder.name
        shutil.copytree(folder, destination_sign_folder)
        print(f"   ✔ 複製標誌資料夾: {folder.name}")

    return destination


# ==================== 單一旅程處理 ====================

def process_trip(root: Path, upload_root: Path, trip_name: str) -> bool:
    trip_dir = root / trip_name
    print(f"\n{'=' * 70}\n🚗 處理旅程: {trip_name}")

    if not trip_dir.exists():
        print(f"⚠️ clean.txt 中存在旅程，但找不到資料夾:\n   {trip_dir}")
        return False

    if not trip_dir.is_dir():
        print(f"⚠️ 不是資料夾: {trip_dir}")
        return False

    frames_dir = trip_dir / "frames"
    if not frames_dir.exists():
        print(f"⚠️ 找不到 frames 資料夾:\n   {frames_dir}")
        return False

    detect_csv = frames_dir / "detect.csv"
    if not detect_csv.exists():
        print(f"⚠️ 找不到 detect.csv:\n   {detect_csv}")
        return False

    sign_folders = get_sign_folders(frames_dir)
    print(f"🪧 找到 {len(sign_folders)} 個標誌資料夾")

    # 逐一對標誌進行下採樣
    total_before = 0
    total_after = 0
    for idx, folder in enumerate(sign_folders, start=1):
        print(f"[{idx}/{len(sign_folders)}]", end=" ")
        total_before += len(get_sorted_images(folder))
        total_after += downsample_sign_folder(folder, TARGET_NUM)

    print(f"\n📊 影像下採樣結果:\n   原始影像: {total_before} 張\n   最終影像: {total_after} 張\n   刪除影像: {total_before - total_after} 張")

    # 複製到 Upload 資料夾
    print("\n📦 正在複製 detect.csv 與標誌資料夾到 Upload...")
    destination = copy_trip_to_upload(trip_dir, upload_root, sign_folders)

    # 驗證 detect.csv
    copied_detect_csv = destination / "frames" / "detect.csv"
    if not copied_detect_csv.exists():
        print(f"❌ Upload 中找不到 detect.csv:\n   {copied_detect_csv}")
        return False

    # 驗證標誌資料夾
    for source_folder in sign_folders:
        copied_folder = destination / "frames" / source_folder.name
        if not copied_folder.exists():
            print(f"❌ 標誌資料夾複製失敗:\n   {copied_folder}")
            return False

    print(f"\n✅ 複製完成:\n   {destination}")
    return True


# ==================== 掃描與處理 ====================

def scan_and_process(root: Path):
    clean_log = root / CLEAN_LOG_NAME
    upload_log = root / UPLOAD_LOG_NAME
    upload_root = root / "Upload"

    upload_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("標誌影像 Upload 前處理")
    print(f"{'=' * 70}\n📁 根目錄: {root}\n📄 清洗完成紀錄: {clean_log}\n📄 Upload 完成紀錄: {upload_log}\n📦 Upload 目錄: {upload_root}")

    if not clean_log.exists():
        print(f"\n❌ 找不到 clean.txt:\n   {clean_log}")
        return

    cleaned_trips = read_trip_log(clean_log)
    uploaded_trips = read_trip_log(upload_log)

    print(f"\n📋 已完成相似度清理: {len(cleaned_trips)} 個旅程")
    print(f"📋 已完成 Upload 處理: {len(uploaded_trips)} 個旅程")

    pending_trips = sorted(cleaned_trips - uploaded_trips)
    print(f"🔍 尚待處理: {len(pending_trips)} 個旅程")

    if not pending_trips:
        print("\n✅ 沒有需要處理的旅程。")
        return

    success_count = 0
    failed_count = 0
    total = len(pending_trips)

    for idx, trip_name in enumerate(pending_trips, start=1):
        print(f"\n[{idx}/{total}] {trip_name}")
        try:
            success = process_trip(root, upload_root, trip_name)
            if success:
                mark_trip_as_completed(upload_log, trip_name)
                uploaded_trips.add(trip_name)
                success_count += 1
                print(f"\n📝 已加入 upload.txt: {trip_name}")
            else:
                failed_count += 1
                print(f"\n⚠️ 旅程處理未完成，不加入 upload.txt: {trip_name}")
        except Exception as e:
            failed_count += 1
            print(f"\n❌ 旅程發生例外: {trip_name}\n   錯誤: {e}")

    print(f"\n{'=' * 70}\n📊 全部處理結果\n{'=' * 70}")
    print(f"待處理旅程: {total}\n成功: {success_count}\n失敗: {failed_count}\n\n✅ 全部 Upload 前處理完成")


# ==================== Main ====================

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

    if not root.exists():
        print(f"❌ 根目錄不存在: {root}")
        sys.exit(1)

    scan_and_process(root)


if __name__ == "__main__":
    main()