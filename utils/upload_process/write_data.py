#!/usr/bin/env python3

import json
import shutil
from pathlib import Path
import pandas as pd


# ==================== 路徑 ====================

ROOT = Path("./Processed_Videos/")

# 8.combine_data.py 的輸出
RESULT_DIR = ROOT / "Result"
CSV_PATH = RESULT_DIR / "marker_gnss_ray.csv"
RESULT_IMAGE_DIR = RESULT_DIR / "images"

# Web 專案輸出
WEB_DIR = ROOT / "GM_Proj_Web"
JSON_PATH = WEB_DIR / "data.json"
WEB_IMAGE_DIR = WEB_DIR / "images"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ==================== 初始化 ====================

WEB_DIR.mkdir(parents=True, exist_ok=True)
WEB_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 檢查輸入 ====================

if not CSV_PATH.exists():
    raise RuntimeError(f"找不到 marker_gnss_ray.csv: {CSV_PATH}")

if not RESULT_IMAGE_DIR.exists():
    raise RuntimeError(f"找不到代表影像資料夾: {RESULT_IMAGE_DIR}")


# ==================== 讀取與驗證 CSV ====================

df = pd.read_csv(CSV_PATH)

required_cols = [
    "lat", "lon", "mean_error_m", "median_error_m", "rms_error_m",
    "max_error_m", "date", "time", "trip", "sign", "scene_id", "class"
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise RuntimeError("marker_gnss_ray.csv 缺少必要欄位: " + ", ".join(missing_cols))

# 選取 Web 所需資料並賦予標籤
df = df[required_cols].copy()
df["label"] = "predicted"

# JSON 不應輸出 NaN，將 NaN 轉為 None (輸出為 null)
df = df.astype(object).where(pd.notna(df), None)


# ==================== 讀取既有 JSON ====================

if JSON_PATH.exists():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        if not isinstance(old_data, list):
            raise RuntimeError("既有 data.json 格式不是 list")
    except Exception as e:
        raise RuntimeError(f"讀取既有 data.json 失敗: {e}")
else:
    old_data = []

# 取得已存在之 scene_id 避免重複加入
old_ids = {str(item["scene_id"]) for item in old_data if isinstance(item, dict) and "scene_id" in item}

# 僅處理未登錄的新資料
new_df = df[~df["scene_id"].astype(str).isin(old_ids)].copy()

if new_df.empty:
    print("[INFO] 沒有新的資料需要加入 data.json")
else:
    new_data = new_df.to_dict(orient="records")
    merged_data = old_data + new_data

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[OK] 新增 {len(new_data)} 筆資料到:\n     {JSON_PATH}")


# ==================== 補齊 Web 代表影像 ====================

copied_count = 0
missing_count = 0

for _, row in new_df.iterrows():
    trip_name = str(row["trip"])
    sign_name = str(row["sign"])
    scene_id = str(row["scene_id"])

    # 搜尋 Result/images 下對應副檔名的代表圖檔
    source_image = None
    for ext in IMAGE_EXTS:
        candidate = RESULT_IMAGE_DIR / f"{trip_name}__{sign_name}{ext}"
        if candidate.exists():
            source_image = candidate
            break

    if source_image is None:
        print(f"[WARN] 找不到代表影像: {scene_id}")
        missing_count += 1
        continue

    destination_image = WEB_IMAGE_DIR / source_image.name
    if destination_image.exists():
        continue

    try:
        shutil.copy2(source_image, destination_image)
        copied_count += 1
        print(f"[OK] Image copied: {source_image.name}")
    except Exception as e:
        print(f"[WARN] 影像複製失敗 {scene_id}: {e}")
        missing_count += 1


# ==================== 輸出統計摘要 ====================

print(f"\n{'=' * 70}\nWeb 資料更新完成\n{'=' * 70}")
print(f"CSV:    {CSV_PATH}\nJSON:   {JSON_PATH}\nImages: {WEB_IMAGE_DIR}")
print(f"新增資料: {len(new_df)} 筆\n新增影像: {copied_count} 張\n缺少影像: {missing_count} 張")