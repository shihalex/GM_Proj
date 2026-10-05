#!/usr/bin/env python3

import shutil
from pathlib import Path
import numpy as np
import pandas as pd


# ==================== 路徑與參數 ====================

ROOT = Path("./Processed_Videos/")
BASE_DIR = ROOT / "Uploaded"
FINISH_TXT = BASE_DIR / "finish.txt"

OUTPUT_DIR = ROOT / "Result"
OUTPUT_CSV = OUTPUT_DIR / "marker_gnss_ray.csv"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ==================== 初始化目錄與環境 ====================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not FINISH_TXT.exists():
    raise RuntimeError(f"找不到 finish.txt: {FINISH_TXT}")

with open(FINISH_TXT, "r", encoding="utf-8") as f:
    scene_ids = [line.strip() for line in f if line.strip()]

dfs = []


# ==================== 逐標誌場景讀取與匯整 ====================

for scene_id in scene_ids:
    try:
        trip_name, sign_name = scene_id.split("/", 1)
    except ValueError:
        print(f"[WARN] 無效 scene ID: {scene_id}")
        continue

    scene_dir = BASE_DIR / trip_name / "frames" / sign_name
    marker_csv = scene_dir / "marker_gnss_ray.csv"
    data_csv = scene_dir / "data.csv"
    images_dir = scene_dir / "images"

    if not marker_csv.exists():
        print(f"[WARN] 找不到: {marker_csv}")
        continue

    # 讀取定位計算結果並補充基本中繼資料
    df = pd.read_csv(marker_csv)
    df["trip"] = trip_name
    df["sign"] = sign_name
    df["scene_id"] = scene_id

    # 讀取標誌屬性 (class, date, time)
    class_value = np.nan
    date_value, time_value = None, None

    if data_csv.exists():
        try:
            data_df = pd.read_csv(data_csv)
            if not data_df.empty:
                if "class" in data_df.columns:
                    class_value = data_df["class"].iloc[0]
                if "date" in data_df.columns:
                    date_value = data_df["date"].iloc[-1]
                if "time" in data_df.columns:
                    time_value = data_df["time"].iloc[-1]
        except Exception as e:
            print(f"[WARN] data.csv 讀取失敗 {scene_id}: {e}")

    df["class"] = class_value
    if date_value is not None:
        df["date"] = date_value
    if time_value is not None:
        df["time"] = time_value

    dfs.append(df)

    # 挑選最後一張特寫作為代表影像並複製至 Result/images
    if images_dir.exists():
        images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if images:
            last_image = images[-1]
            output_name = f"{trip_name}__{sign_name}{last_image.suffix.lower()}"
            try:
                shutil.copy2(last_image, IMAGE_OUTPUT_DIR / output_name)
            except Exception as e:
                print(f"[WARN] 影像複製失敗 {scene_id}: {e}")


# ==================== 資料合併與防重複匯出 ====================

if not dfs:
    raise RuntimeError("沒有找到有效的 marker_gnss_ray.csv")

new_df = pd.concat(dfs, ignore_index=True)

if OUTPUT_CSV.exists():
    old_df = pd.read_csv(OUTPUT_CSV)
    if "scene_id" in old_df.columns:
        new_df = new_df[~new_df["scene_id"].isin(old_df["scene_id"])]
    merged_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    merged_df = new_df

merged_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"[OK] CSV saved: {OUTPUT_CSV}\n[OK] Images saved: {IMAGE_OUTPUT_DIR}")