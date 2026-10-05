#!/usr/bin/env python3

import shutil
from pathlib import Path
import pandas as pd


# ==================== 路徑 ====================

ROOT = Path("./Processed_Videos/")
UPLOAD_ROOT = ROOT / "Upload"
UPLOADED_ROOT = ROOT / "Uploaded"

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ==================== 工具 ====================

def get_sign_folders(frames_dir: Path):
    """
    取得 {track_id}_{class_name} 格式的標誌資料夾。
    """
    folders = [p for p in frames_dir.iterdir() if p.is_dir() and p.name.split("_", 1)[0].isdigit()]
    folders.sort(key=lambda p: int(p.name.split("_", 1)[0]))
    return folders


def get_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def parse_sign_folder(folder_name: str):
    """
    解析標誌資料夾名稱 (例如 15_Stop -> track_id=15, class_name='Stop')。
    """
    parts = folder_name.split("_", 1)
    track_id = int(parts[0])
    class_name = parts[1] if len(parts) > 1 else ""
    return track_id, class_name


# ==================== 建立 data.csv ====================

def create_data_csv(detect_df: pd.DataFrame, sign_folder: Path, output_csv: Path):
    """
    從 detect.csv 中建立該標誌專屬的 data.csv。
    欄位包含：frame, row, column, class, lat, lon, date, time
    """
    track_id, class_name = parse_sign_folder(sign_folder.name)
    images = get_images(sign_folder)
    image_names = {image.name for image in images}

    rows = []
    id_columns = [col for col in detect_df.columns if col.startswith("ID_")]

    for _, row in detect_df.iterrows():
        frame_name = str(row.get("名稱", "")).strip()

        # 跳過已被下採樣刪除的 frame
        if frame_name not in image_names:
            continue

        # 尋找對應的 Track ID 與 index
        matched_index = None
        for id_col in id_columns:
            value = row.get(id_col)
            if pd.isna(value):
                continue
            try:
                current_id = int(float(value))
            except Exception:
                continue

            if current_id == track_id:
                matched_index = id_col.split("_", 1)[1]
                break

        if matched_index is None:
            continue

        # 解析 BBox [cx, cy, w, h]
        bbox_value = row.get(f"BBox_{matched_index}")
        if pd.isna(bbox_value):
            continue

        try:
            bbox_text = str(bbox_value).strip().replace("[", "").replace("]", "")
            values = [float(v.strip()) for v in bbox_text.split(",")]
            if len(values) < 2:
                continue
            column, row_px = values[0], values[1]
        except Exception:
            continue

        # 整理單筆標記資料
        rows.append({
            "frame": frame_name,
            "row": row_px,
            "column": column,
            "class": class_name,
            "lat": row.get("緯度", ""),
            "lon": row.get("經度", ""),
            "date": row.get("日期", ""),
            "time": row.get("時間", "")
        })

    if not rows:
        raise RuntimeError(f"找不到 {sign_folder.name} 對應的 detect.csv 資料")

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return len(output_df)


# ==================== 單一標誌 ====================

def prepare_sign(detect_df: pd.DataFrame, source_sign: Path, destination_sign: Path):
    """
    建立 VGGT scene 目錄結構：
        標誌/
        ├── images/
        └── data.csv
    """
    destination_images = destination_sign / "images"

    if destination_sign.exists():
        shutil.rmtree(destination_sign)

    destination_images.mkdir(parents=True, exist_ok=True)

    # 複製標誌影像
    images = get_images(source_sign)
    for image in images:
        shutil.copy2(image, destination_images / image.name)

    # 產出專屬 data.csv
    data_csv = destination_sign / "data.csv"
    row_count = create_data_csv(detect_df, source_sign, data_csv)

    print(f"   ✔ {source_sign.name}: {len(images)} images, {row_count} rows")


# ==================== 單一旅程 ====================

def prepare_trip(trip_dir: Path):
    trip_name = trip_dir.name
    print(f"\n{'=' * 70}\n🚗 準備 VGGT 資料: {trip_name}")

    source_frames = trip_dir / "frames"
    if not source_frames.exists():
        print("⚠️ 找不到 frames/")
        return False

    detect_csv = source_frames / "detect.csv"
    if not detect_csv.exists():
        print("⚠️ 找不到 detect.csv")
        return False

    detect_df = pd.read_csv(detect_csv)

    # 建立輸出目錄並備份 detect.csv
    destination_frames = UPLOADED_ROOT / trip_name / "frames"
    destination_frames.mkdir(parents=True, exist_ok=True)
    shutil.copy2(detect_csv, destination_frames / "detect.csv")

    sign_folders = get_sign_folders(source_frames)
    print(f"🪧 共 {len(sign_folders)} 個標誌")

    success = 0
    for sign_folder in sign_folders:
        destination_sign = destination_frames / sign_folder.name
        try:
            prepare_sign(detect_df, sign_folder, destination_sign)
            success += 1
        except Exception as e:
            print(f"   ❌ {sign_folder.name}: {e}")

    print(f"✅ {trip_name}: {success}/{len(sign_folders)} 個標誌準備完成")
    return success == len(sign_folders)


# ==================== Main ====================

def main():
    UPLOADED_ROOT.mkdir(parents=True, exist_ok=True)

    if not UPLOAD_ROOT.exists():
        raise RuntimeError(f"找不到 Upload: {UPLOAD_ROOT}")

    trips = sorted([path for path in UPLOAD_ROOT.iterdir() if path.is_dir()])
    print(f"🔍 找到 {len(trips)} 個旅程")

    success = 0
    for trip in trips:
        try:
            if prepare_trip(trip):
                success += 1
        except Exception as e:
            print(f"❌ {trip.name}: {e}")

    print(f"\n{'=' * 70}\n✅ VGGT 資料準備完成: {success}/{len(trips)} 個旅程")


if __name__ == "__main__":
    main()