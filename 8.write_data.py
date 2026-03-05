import os
import pandas as pd
import json
import shutil

# ===== paths =====
csv_path = "/home/alex930307/GM/marker_gnss_ray.csv"
json_path = "/home/alex930307/GM/GM_Proj_Web/data.json"

web_image_dir = "/home/alex930307/GM/GM_Proj_Web/images"
upload_root = "/media/alex930307/TRANSCEND/GM/GM_upload"

os.makedirs(os.path.dirname(json_path), exist_ok=True)
os.makedirs(web_image_dir, exist_ok=True)

# ===== 讀 CSV =====
df = pd.read_csv(csv_path)

cols = [
    "lat", "lon",
    "mean_error_m", "median_error_m", "rms_error_m", "max_error_m",
    "date", "time",
    "folder_id", "class"
]
df = df[cols]
df["label"] = "predicted"

# ===== 讀既有 JSON（若存在）=====
if os.path.isfile(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    old_ids = {
        str(d["folder_id"])
        for d in old_data
        if "folder_id" in d
    }

else:
    old_data = []
    old_ids = set()

# ===== 只保留尚未寫入的資料 =====
new_df = df[~df["folder_id"].astype(str).isin(old_ids)]

if new_df.empty:
    print("[INFO] No new data to append.")
else:
    new_data = new_df.to_dict(orient="records")
    merged_data = old_data + new_data

    # 寫回 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Appended {len(new_data)} new records to data.json")

# ===== 補齊影像 =====
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

for fid in new_df["folder_id"].astype(str):
    dst_img = os.path.join(web_image_dir, f"{fid}.jpg")
    if os.path.isfile(dst_img):
        continue

    src_images_dir = os.path.join(upload_root, fid, "images")
    if not os.path.isdir(src_images_dir):
        print(f"[WARN] images folder not found for {fid}")
        continue

    images = [
        f for f in os.listdir(src_images_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    if not images:
        print(f"[WARN] no images in {src_images_dir}")
        continue

    # 取數字最大的檔名
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    last_image = images[-1]

    try:
        shutil.copyfile(
            os.path.join(src_images_dir, last_image),
            dst_img
        )
        print(f"[OK] Image copied for {fid}")
    except Exception as e:
        print(f"[WARN] Failed copying image for {fid}: {e}")
