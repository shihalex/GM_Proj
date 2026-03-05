#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import os
import shutil
import csv

# ========= 參數 =========
ROOT = Path("/media/alex930307/TRANSCEND/GM/video/Normal")

FLOW_THRESHOLD = 2.0
DOWNSCALE_SIZE = 640
MAX_FEATURES = 300
FAST_THRESHOLD = 25
MIN_KEEP = 10

TARGET_EXTS = {".jpg", ".jpeg", ".png"}

# ========= CSV =========
def load_csv(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows

    with open(csv_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue

            frame, r, c, cls = parts[:4]
            lat  = parts[4] if len(parts) > 4 else ""
            lon  = parts[5] if len(parts) > 5 else ""
            date = parts[6] if len(parts) > 6 else ""
            time = parts[7] if len(parts) > 7 else ""

            rows.append([frame, int(r), int(c), cls, lat, lon, date, time])

    return rows

def save_csv(csv_path: Path, rows):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,row,column,class,lat,lon,date,time\n")
        for frame, r, c, cls, lat, lon, date, time in rows:
            f.write(f"{frame},{r},{c},{cls},{lat},{lon},{date},{time}\n")

# ========= 影像工具 =========
def get_sorted_images(folder: Path):
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in TARGET_EXTS]

    def extract_index(p: Path):
        name = p.stem
        if name.isdigit():
            return int(name)
        if name.startswith("frame_"):
            return int(name.replace("frame_", ""))
        raise ValueError(f"無法解析影像檔名：{p.name}")

    return sorted(imgs, key=extract_index)


def resize_and_gray(img):
    h, w = img.shape[:2]
    scale = DOWNSCALE_SIZE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def optical_flow_fast(img1, img2):
    fast = cv2.FastFeatureDetector_create(FAST_THRESHOLD, True)
    kp = fast.detect(img1, None)
    if not kp:
        return 999

    kp = sorted(kp, key=lambda x: -x.response)[:MAX_FEATURES]
    p0 = np.array([k.pt for k in kp], np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)

    good0 = p0[st == 1]
    good1 = p1[st == 1]
    if len(good0) == 0:
        return 999

    return np.median(np.linalg.norm(good1 - good0, axis=1))


# ========= 清洗單一 {id} 資料夾 =========
def clean_id_folder(folder: Path):
    print(f"\n清洗：{folder}")

    imgs = get_sorted_images(folder)
    csv_path = folder / "data.csv"
    rows = load_csv(csv_path)

    if len(imgs) < MIN_KEEP:
        shutil.rmtree(folder)
        print("❌ 原始影像不足，刪除")
        return

    prev = resize_and_gray(cv2.imread(str(imgs[0])))
    keep_mask = [True]

    for img in imgs[1:]:
        cur = cv2.imread(str(img))
        if cur is None:
            os.remove(img)
            keep_mask.append(False)
            continue

        cur_g = resize_and_gray(cur)
        flow = optical_flow_fast(prev, cur_g)

        if flow < FLOW_THRESHOLD:
            os.remove(img)
            keep_mask.append(False)
        else:
            prev = cur_g
            keep_mask.append(True)

    if sum(keep_mask) < MIN_KEEP:
        shutil.rmtree(folder)
        print("❌ 清洗後不足，刪除")
        return

    new_rows = [rows[i] for i, k in enumerate(keep_mask) if k and i < len(rows)]
    renumber_images_and_csv(folder, new_rows)
    print("✔ 完成")


def renumber_images_and_csv(folder: Path, rows):
    imgs = get_sorted_images(folder)

    # 建立 old_frame -> new_frame 對照
    frame_map = {}

    tmp = []
    for i, img in enumerate(imgs, 1):
        old_name = img.name
        t = folder / f"__tmp_{i}.jpg"
        img.rename(t)
        tmp.append((old_name, t))

    final_imgs = []
    for i, (old_name, t) in enumerate(tmp, 1):
        new_name = f"{i}.jpg"
        final = folder / new_name
        t.rename(final)
        final_imgs.append(final)
        frame_map[old_name] = new_name

    # 🔥 更新 rows 裡的 frame 欄位
    new_rows = []
    for frame, r, c, cls, lat, lon, date, time in rows:
        new_frame = frame_map.get(frame, frame)
        new_rows.append([new_frame, r, c, cls, lat, lon, date, time])
        
    # 寫回 data.csv
    save_csv(folder / "data.csv", new_rows)

    # 建立 images/ 並搬移影像
    images_dir = folder / "images"
    images_dir.mkdir(exist_ok=True)

    for img in final_imgs:
        shutil.move(str(img), images_dir / img.name)


# ========= Main =========
def main():
    trips = sorted([d for d in ROOT.iterdir() if d.is_dir()])

    for trip in trips:
        detected = trip / "detected_frame"
        if not detected.exists():
            continue

        id_folders = sorted(
            [d for d in detected.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name)
        )

        for f in id_folders:
            clean_id_folder(f)

    print("\n🎉 全部 Normal/detected_frame 清洗完成")


if __name__ == "__main__":
    main()
