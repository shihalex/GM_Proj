import os
import shutil
import pandas as pd
import numpy as np

base_dir = "/media/alex930307/TRANSCEND/GM/GM_upload"
finish_txt = os.path.join(base_dir, "finish.txt")

output_dir = "/home/alex930307/GM"
output_csv = os.path.join(output_dir, "marker_gnss_ray.csv")

image_output_dir = "/home/alex930307/GM/GM_Proj_Web/images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

dfs = []

with open(finish_txt, "r") as f:
    folder_ids = [line.strip() for line in f if line.strip()]

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

for fid in folder_ids:
    marker_csv = os.path.join(base_dir, fid, "marker_gnss_ray.csv")
    data_csv = os.path.join(base_dir, fid, "data.csv")
    images_dir = os.path.join(base_dir, fid, "images")

    if not os.path.isfile(marker_csv):
        continue

    df = pd.read_csv(marker_csv)

    # ---- date / time：不存在或為空都補 0 ----
    for col in ["date", "time"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "NaN", "None"]), col] = "0"
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # folder_id
    df["folder_id"] = fid

    # class
    class_value = np.nan
    if os.path.isfile(data_csv):
        try:
            data_df = pd.read_csv(data_csv)
            if "class" in data_df.columns and not data_df.empty:
                class_value = data_df["class"].iloc[0]
        except Exception as e:
            print(f"[WARN] Failed reading data.csv in {fid}: {e}")

    df["class"] = class_value
    dfs.append(df)

    # ---------- 複製最後一張影像（依數字排序） ----------
    if os.path.isdir(images_dir):
        images = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ]

        if images:
            images.sort(key=lambda x: int(os.path.splitext(x)[0]))
            last_image = images[-1]

            src_img = os.path.join(images_dir, last_image)
            dst_img = os.path.join(image_output_dir, f"{fid}.jpg")

            try:
                shutil.copyfile(src_img, dst_img)
            except Exception as e:
                print(f"[WARN] Failed copying image for {fid}: {e}")

if not dfs:
    raise RuntimeError("No valid marker_gnss_ray.csv found.")

new_df = pd.concat(dfs, ignore_index=True)

if os.path.isfile(output_csv):
    old_df = pd.read_csv(output_csv)
    new_df = new_df[~new_df["folder_id"].isin(old_df["folder_id"])]
    merged_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    merged_df = new_df

merged_df.to_csv(output_csv, index=False)


print(f"[OK] CSV saved: {output_csv}")
print(f"[OK] Images saved to: {image_output_dir}")
