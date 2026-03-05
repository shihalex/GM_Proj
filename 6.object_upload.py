#!/usr/bin/env python3
import shutil
from pathlib import Path
import pandas as pd

SRC_ROOT = Path("/media/alex930307/TRANSCEND/GM/video/Normal")
DST_ROOT = Path("/media/alex930307/TRANSCEND/GM/upload")
TARGET_NUM = 20
IMG_EXTS = {".jpg", ".jpeg", ".png"}

DST_ROOT.mkdir(parents=True, exist_ok=True)

def downsample_and_renumber(images_dir: Path, data_csv: Path, target=20):
    imgs = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
        key=lambda p: int(p.stem)
    )

    n = len(imgs)
    if n == 0:
        return

    # ===== 決定要保留哪些 index =====
    if n <= target:
        keep_indices = list(range(n))
    else:
        keep_indices = [0, n - 1]
        remaining = target - 2
        step = (n - 2) / remaining
        for i in range(remaining):
            idx = round(1 + i * step)
            keep_indices.append(idx)
        keep_indices = sorted(set(keep_indices))

    kept_imgs = [imgs[i] for i in keep_indices]

    # ===== 讀取 data.csv =====
    df = pd.read_csv(data_csv)

    # frame -> index 對應（假設 frame 是 1.jpg, 2.jpg ...）
    def frame_to_idx(f):
        try:
            return int(Path(f).stem) - 1
        except Exception:
            return None

    df["__idx"] = df["frame"].apply(frame_to_idx)
    df = df[df["__idx"].isin(keep_indices)].copy()

    # ===== 刪除不需要的影像 =====
    for img in imgs:
        if img not in kept_imgs:
            img.unlink()

    # ===== 重新編號影像 + 建立對應表 =====
    oldidx_to_newframe = {}
    for new_i, img in enumerate(sorted(kept_imgs, key=lambda p: int(p.stem)), start=1):
        new_name = f"{new_i}.jpg"
        old_idx = int(img.stem) - 1
        img.rename(images_dir / new_name)
        oldidx_to_newframe[old_idx] = new_name

    # ===== 更新 data.csv frame =====
    df["frame"] = df["__idx"].map(oldidx_to_newframe)
    df.drop(columns="__idx", inplace=True)
    df.to_csv(data_csv, index=False, encoding="utf-8-sig")


def main():
    new_id = 1

    for trip in sorted(SRC_ROOT.iterdir()):
        detected = trip / "detected_frame"
        if not detected.exists():
            continue

        for obj in sorted(detected.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
            if not obj.is_dir():
                continue

            images_dir = obj / "images"
            data_csv = obj / "data.csv"
            if not images_dir.exists() or not data_csv.exists():
                continue

            dst_obj = DST_ROOT / str(new_id)
            dst_obj.mkdir(parents=True, exist_ok=True)

            # ===== 複製 =====
            shutil.copytree(images_dir, dst_obj / "images")
            shutil.copy2(data_csv, dst_obj / "data.csv")

            # ===== 下採樣 + 重編號 + 同步 CSV =====
            downsample_and_renumber(
                dst_obj / "images",
                dst_obj / "data.csv",
                TARGET_NUM
            )

            print(f"✔ 完成：{trip.name}/detected_frame/{obj.name} → upload/{new_id}")
            new_id += 1

    print(f"\n🎉 全部完成，共處理 {new_id - 1} 個標誌資料夾")


if __name__ == "__main__":
    main()
