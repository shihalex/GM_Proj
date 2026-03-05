#!/usr/bin/env python3
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DEFAULT_ROOT = Path(os.getenv("DETECTED_ROOT", "/media/alex930307/TRANSCEND/GM/video"))

# ========= 時間工具 =========
def parse_time_str(t: str):
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None

def format_hms(sec: float):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def parse_date_str(d: str):
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None

# ========= frame 名稱解析 =========
def parse_frame_idx(frame_name: str):
    try:
        name = Path(frame_name).stem
        if name.startswith("frame_"):
            return int(name.split("_")[1])
        if name.isdigit():
            return int(name)
    except Exception:
        pass
    return None

# ========= 重新編號影像 + 更新 data.csv =========
def renumber_images_and_update_csv(detected_dir: Path):
    for obj_dir in detected_dir.iterdir():
        if not obj_dir.is_dir():
            continue

        data_csv = obj_dir / "data.csv"
        if not data_csv.exists():
            continue

        images = sorted(
            obj_dir.glob("frame_*.jpg"),
            key=lambda p: parse_frame_idx(p.name)
        )
        if not images:
            continue

        tmp_list = []
        for i, img in enumerate(images):
            old_idx = parse_frame_idx(img.name)
            tmp = obj_dir / f"__tmp_{i}.jpg"
            img.rename(tmp)
            tmp_list.append((old_idx, tmp))

        frame_map = {}
        for new_idx, (old_idx, tmp) in enumerate(tmp_list, start=1):
            tmp.rename(obj_dir / f"{new_idx}.jpg")
            if old_idx is not None:
                frame_map[old_idx] = new_idx

        df = pd.read_csv(data_csv)

        def map_frame(x):
            idx = parse_frame_idx(str(x))
            return f"{frame_map[idx]}.jpg" if idx in frame_map else x

        df["frame"] = df["frame"].apply(map_frame)
        df.to_csv(data_csv, index=False, encoding="utf-8-sig")

        print(f"✔ 重新編號完成：{obj_dir.name}")

# ========= 單一 ocr.csv 處理 =========
def process_ocr_csv(csv_path: Path):
    print(f"▶ 處理 {csv_path}")
    df = pd.read_csv(csv_path)

    # === GNSS 基準 ===
    base_candidates = df[df["time"].notna() & df["fps"].notna()]
    if base_candidates.empty:
        print("⚠️ 找不到 GNSS 基準列")
        return

    base = base_candidates.iloc[0]
    base_time_sec = parse_time_str(str(base["time"]))
    fps = float(base["fps"])
    base_frame = parse_frame_idx(base["frame"])

    if base_time_sec is None or fps <= 0 or base_frame is None:
        print("⚠️ GNSS 基準資料無效")
        return

    # === date 基準 ===
    date_candidates = df[df["date"].notna()]
    if date_candidates.empty:
        print("⚠️ 找不到 date 基準")
        return

    base_date = parse_date_str(str(date_candidates.iloc[0]["date"]))
    if base_date is None:
        print("⚠️ date 格式錯誤")
        return

    # === 推算 time + date ===
    times, dates, secs = [], [], []

    for _, r in df.iterrows():
        fidx = parse_frame_idx(r["frame"])
        if fidx is None:
            times.append(None)
            dates.append(None)
            secs.append(None)
            continue

        tsec = base_time_sec + (fidx - base_frame) / fps
        day_offset = int(tsec // 86400)
        sec_in_day = tsec % 86400

        times.append(format_hms(sec_in_day))
        dates.append((base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d"))
        secs.append(tsec)

    df["time"] = times
    df["date"] = dates
    df["time_sec"] = secs

    # === 清空 lat/lon ===
    df["lat"] = pd.NA
    df["lon"] = pd.NA

    # === GNSS 對齊 ===
    trip_dir = csv_path.parent.parent
    gnss_csv = trip_dir / f"{trip_dir.name}.csv"

    if gnss_csv.exists():
        gdf = pd.read_csv(gnss_csv)
        gdf["time_sec"] = gdf["time"].apply(parse_time_str)
        gdf = gdf[gdf["time_sec"].notna()]

        used = set()
        for _, g in gdf.iterrows():
            diffs = (df["time_sec"] - g["time_sec"]).abs()
            diffs = diffs[~df.index.isin(used)]
            if diffs.empty:
                continue

            idx = diffs.idxmin()
            if diffs.loc[idx] <= 0.5:
                df.at[idx, "lat"] = g["lat"]
                df.at[idx, "lon"] = g["lon"]
                used.add(idx)

    # === 插值 ===
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if df[["lat", "lon"]].dropna().shape[0] >= 2:
        df["lat"] = df["lat"].interpolate(limit_area="inside")
        df["lon"] = df["lon"].interpolate(limit_area="inside")

    # === 回寫 ocr.csv ===
    df.drop(columns=["time_sec"], inplace=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("✔ ocr.csv 完成")

    # === 同步到 data.csv ===
    frame_map = df.set_index("frame")[["lat", "lon", "date", "time"]].to_dict("index")
    detected_dir = csv_path.parent

    for sub in detected_dir.iterdir():
        data_csv = sub / "data.csv"
        if not data_csv.exists():
            continue

        ddf = pd.read_csv(data_csv)
        for col in ["lat", "lon", "date", "time"]:
            ddf[col] = ddf.get(col, pd.NA)

        for i, r in ddf.iterrows():
            f = r["frame"]
            if f in frame_map:
                for k in frame_map[f]:
                    ddf.at[i, k] = frame_map[f][k]

        ddf.to_csv(data_csv, index=False, encoding="utf-8-sig")

    print("✔ data.csv 同步完成")

    renumber_images_and_update_csv(detected_dir)

# ========= 掃描 =========
def scan_and_process(root: Path):
    targets = list(root.glob("**/detected_frame/ocr.csv"))
    print(f"🔎 找到 {len(targets)} 個 ocr.csv")
    for p in targets:
        try:
            process_ocr_csv(p)
        except Exception as e:
            print(f"❌ 失敗 {p}: {e}")

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    scan_and_process(root)

if __name__ == "__main__":
    main()
