import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np


# ==================== 參數 ====================

ROOT = Path("./Processed_Videos/")
GETLATLON_LOG_NAME = "getlatlon.txt"
CLEAN_LOG_NAME = "clean.txt"

FLOW_THRESHOLD = 2.0
DOWNSCALE_SIZE = 640
MAX_FEATURES = 300
FAST_THRESHOLD = 25

# 相似度清理完成後，至少需要保留的影像數量
MIN_KEEP = 10

# YOLO 階段目前輸出的是 PNG
TARGET_EXTS = {".png", ".jpg", ".jpeg"}


# ==================== 完成紀錄工具 ====================

def read_trip_log(log_path: Path) -> set:
    """
    讀取旅程紀錄。
    每一行代表一個旅程資料夾名稱。
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
    將成功完成清理的旅程加入 clean.txt。
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{trip_name}\n")
    except Exception as e:
        raise RuntimeError(f"無法寫入完成紀錄 {log_path}: {e}")


# ==================== 影像工具 ====================

def get_sorted_images(folder: Path):
    """
    取得標誌資料夾中的所有影像，並依照檔名排序。
    """
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TARGET_EXTS]

    def extract_index(p: Path):
        name = p.stem
        # 已重新編號的格式 (1.jpg, 2.jpg)
        if name.isdigit():
            return (0, int(name))
        # 舊格式 (frame_1.png)
        if name.startswith("frame_"):
            try:
                return (1, int(name.replace("frame_", "")))
            except ValueError:
                pass
        # YOLO 時間戳記格式 (FILE260101-12000000.png)
        return (2, name)

    return sorted(imgs, key=extract_index)


def resize_and_gray(img):
    """
    縮小影像並轉成灰階。
    """
    h, w = img.shape[:2]
    scale = DOWNSCALE_SIZE / max(h, w)
    if scale < 1:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def optical_flow_fast(img1, img2):
    """
    使用 FAST + Lucas-Kanade Optical Flow 計算兩張影像之間的位移中位數。
    """
    fast = cv2.FastFeatureDetector_create(FAST_THRESHOLD, True)
    kp = fast.detect(img1, None)
    if not kp:
        return 999

    kp = sorted(kp, key=lambda x: -x.response)[:MAX_FEATURES]
    p0 = np.array([k.pt for k in kp], np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)

    if p1 is None or st is None:
        return 999

    good0 = p0[st == 1]
    good1 = p1[st == 1]
    if len(good0) == 0:
        return 999

    flow = np.linalg.norm(good1 - good0, axis=1)
    return float(np.median(flow))


# ==================== 清洗單一標誌資料夾 ====================

def clean_sign_folder(folder: Path):
    """
    對單一標誌資料夾進行影像相似度清理。

    規則：
    1. 使用 Optical Flow 比較影像。
    2. flow < FLOW_THRESHOLD 時視為過度相似。
    3. 刪除相似影像。
    4. 清理完成後若少於 MIN_KEEP 張，刪除整個標誌資料夾。
    """
    print(f"\n   🪧 清洗標誌: {folder.name}")
    imgs = get_sorted_images(folder)
    original_count = len(imgs)
    print(f"      原始影像: {original_count} 張")

    # 原始影像不足門檻
    if original_count < MIN_KEEP:
        shutil.rmtree(folder)
        print(f"      ❌ 原始影像不足 {MIN_KEEP} 張，刪除整個標誌資料夾")
        return False

    # 尋找第一張有效影像
    first_valid_index = None
    prev = None
    for i, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"      ⚠️ 無法讀取影像，刪除: {img_path.name}")
            try:
                img_path.unlink()
            except Exception:
                pass
            continue
        prev = resize_and_gray(img)
        first_valid_index = i
        break

    if first_valid_index is None:
        shutil.rmtree(folder)
        print("      ❌ 沒有可讀取影像，刪除整個標誌資料夾")
        return False

    # 依序計算光流清理相似影像
    removed_count = first_valid_index
    for img_path in imgs[first_valid_index + 1:]:
        cur = cv2.imread(str(img_path))
        if cur is None:
            try:
                img_path.unlink()
            except Exception:
                pass
            removed_count += 1
            print(f"      ⚠️ 無法讀取，刪除: {img_path.name}")
            continue

        cur_g = resize_and_gray(cur)
        flow = optical_flow_fast(prev, cur_g)

        if flow < FLOW_THRESHOLD:
            try:
                img_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"      ⚠️ 刪除失敗: {img_path.name} ({e})")
        else:
            prev = cur_g  # 與上一張「被保留」的影像作比較基準

    # 重新統計剩餘數量
    remaining_imgs = get_sorted_images(folder)
    remaining_count = len(remaining_imgs)
    print(f"      相似影像刪除: {removed_count} 張\n      清洗後剩餘: {remaining_count} 張")

    if remaining_count < MIN_KEEP:
        shutil.rmtree(folder)
        print(f"      ❌ 清洗後不足 {MIN_KEEP} 張，刪除整個標誌資料夾")
        return False

    print(f"      ✔ 保留標誌資料夾: {folder.name}")
    return True


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


# ==================== 單一旅程 ====================

def process_trip(root: Path, trip_name: str) -> bool:
    """
    清理單一旅程中 frames/ 下的所有標誌資料夾。
    """
    trip_dir = root / trip_name
    print(f"\n{'=' * 70}\n🚗 處理旅程: {trip_name}")

    if not trip_dir.exists():
        print(f"⚠️ getlatlon.txt 中存在旅程，但找不到資料夾:\n   {trip_dir}")
        return False

    if not trip_dir.is_dir():
        print(f"⚠️ 不是資料夾: {trip_dir}")
        return False

    frames_dir = trip_dir / "frames"
    if not frames_dir.exists():
        print(f"⚠️ 找不到 frames 資料夾:\n   {frames_dir}")
        return False

    sign_folders = get_sign_folders(frames_dir)
    print(f"🪧 找到 {len(sign_folders)} 個標誌資料夾")

    if not sign_folders:
        print("ℹ️ 此旅程沒有需要清洗的標誌資料夾")
        return True

    kept_count = 0
    deleted_count = 0

    for idx, folder in enumerate(sign_folders, start=1):
        print(f"   [{idx}/{len(sign_folders)}]")
        try:
            kept = clean_sign_folder(folder)
            if kept:
                kept_count += 1
            else:
                deleted_count += 1
        except Exception as e:
            print(f"❌ 清洗標誌資料夾失敗: {folder}\n   錯誤: {e}")
            return False

    print(f"\n📊 旅程清洗結果:\n   保留標誌: {kept_count}\n   刪除標誌: {deleted_count}")
    return True


# ==================== 掃描與處理 ====================

def scan_and_process(root: Path):
    getlatlon_log = root / GETLATLON_LOG_NAME
    clean_log = root / CLEAN_LOG_NAME

    print("=" * 70)
    print("標誌影像相似度清理")
    print(f"{'=' * 70}\n📁 根目錄: {root}\n📄 經緯度完成紀錄: {getlatlon_log}\n📄 清洗完成紀錄: {clean_log}")

    if not getlatlon_log.exists():
        print(f"\n❌ 找不到 getlatlon.txt:\n   {getlatlon_log}")
        return

    getlatlon_trips = read_trip_log(getlatlon_log)
    cleaned_trips = read_trip_log(clean_log)

    print(f"\n📋 已完成經緯度處理: {len(getlatlon_trips)} 個旅程")
    print(f"📋 已完成相似度清洗: {len(cleaned_trips)} 個旅程")

    pending_trips = sorted(getlatlon_trips - cleaned_trips)
    print(f"🔍 尚待清洗: {len(pending_trips)} 個旅程")

    if not pending_trips:
        print("\n✅ 沒有需要處理的旅程。")
        return

    success_count = 0
    failed_count = 0
    total = len(pending_trips)

    for idx, trip_name in enumerate(pending_trips, start=1):
        print(f"\n[{idx}/{total}] {trip_name}")
        try:
            success = process_trip(root, trip_name)
            if success:
                mark_trip_as_completed(clean_log, trip_name)
                cleaned_trips.add(trip_name)
                success_count += 1
                print(f"\n📝 已加入 clean.txt: {trip_name}")
            else:
                failed_count += 1
                print(f"\n⚠️ 旅程處理未完成，不加入 clean.txt: {trip_name}")
        except Exception as e:
            failed_count += 1
            print(f"\n❌ 旅程發生例外: {trip_name}\n   錯誤: {e}")

    print(f"\n{'=' * 70}\n📊 全部處理結果\n{'=' * 70}")
    print(f"待處理旅程: {total}\n成功: {success_count}\n失敗: {failed_count}\n\n✅ 全部標誌影像清洗完成")


# ==================== Main ====================

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

    if not root.exists():
        print(f"❌ 根目錄不存在: {root}")
        sys.exit(1)

    scan_and_process(root)


if __name__ == "__main__":
    main()