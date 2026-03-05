#!/usr/bin/env python3
import os
import re
import cv2
import json
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import easyocr
import csv
import shutil
import multiprocessing as mp
from multiprocessing import Process

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "999999"

# ========= 參數 =========
VIDEO_ROOT = os.getenv("NEW_VIDEO_ROOT", "/media/alex930307/TRANSCEND/GM/video")
WEIGHTS_PATH = os.getenv("WEIGHTS", "../detect/weights/best.pt")
MASK_JSON = os.getenv("MASK_JSON", "../mask.json")
MIN_FRAME = int(os.getenv("MIN_FRAME", "10"))

# ========= mask 載入（rectangle） =========
def load_masks(mask_json: str) -> dict:
    with open(mask_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        shape["label"]: (
            int(shape["points"][0][0]), int(shape["points"][0][1]),
            int(shape["points"][1][0]), int(shape["points"][1][1])
        )
        for shape in data["shapes"]
    }

# ========= lat / lon OCR 清洗 =========
def clean_latlon(text: str):
    # 1️⃣ 有負號直接丟掉
    if "-" in text:
        return None

    # 2️⃣ 移除所有非數字與小數點
    cleaned = re.sub(r"[^0-9.]", "", text)

    # 3️⃣ 抽第一個合法浮點數
    m = re.search(r"\d+\.\d+", cleaned)
    if m:
        return float(m.group())

    return None

# ========= GNSS CSV 讀取（第一筆） =========
def load_gnss_first_valid_row(video_path):
    folder = os.path.dirname(video_path)
    folder_name = os.path.basename(folder)
    csv_path = os.path.join(folder, f"{folder_name}.csv")

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    # 確保欄位存在
    required = {"lat", "lon", "date", "time"}
    if not required.issubset(df.columns):
        return None

    # 轉成數值（無法轉的變 NaN），再找第一筆 lat/lon 都非 NaN
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    valid = df[df["lat"].notna() & df["lon"].notna()]
    if valid.empty:
        return None

    return valid.iloc[0]

# ========= detected_frame 重新編號 =========
def renumber_detected_frame(out_dir: str):
    folders = sorted(
        [d for d in os.listdir(out_dir)
         if os.path.isdir(os.path.join(out_dir, d)) and d.isdigit()],
        key=int
    )

    for new_id, old_id in enumerate(folders, start=1):
        if str(new_id) == old_id:
            continue
        os.rename(
            os.path.join(out_dir, old_id),
            os.path.join(out_dir, f"__tmp_{new_id}__")
        )

    for d in os.listdir(out_dir):
        if d.startswith("__tmp_") and d.endswith("__"):
            new_id = d.replace("__tmp_", "").replace("__", "")
            os.rename(
                os.path.join(out_dir, d),
                os.path.join(out_dir, new_id)
            )

# ========= 處理單一影片 =========
def process_video(video_path, model, reader, masks, device_str):

    trip_dir = os.path.dirname(video_path)
    video_file = os.path.basename(video_path)

    out_dir = os.path.join(trip_dir, "detected_frame")
    if os.path.exists(out_dir):
        print(f"⏩ 已有 detected_frame，略過 {video_file}")
        return
    os.makedirs(out_dir, exist_ok=True)

    gnss_row = load_gnss_first_valid_row(video_path)
    matched_once = False
    fps_written = False  # ★ 新增

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    '''
    # ===== output video writer =====
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_video_path = os.path.join(trip_dir, "yolo_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    '''

    print(f"🎥 {video_file}: total_frames={total_frames}, fps={fps}")
    pbar = tqdm(total=total_frames, desc=video_file, ncols=80)

    frame_counter = 0
    next_id = 1
    track_map = {}
    track_info = {}
    ocr_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame_name = f"frame_{frame_counter:05d}.jpg"

        # ===== YOLO tracking =====
        results = model.track(
            frame,
            conf=0.8,
            iou=0.5,
            half=True,
            device=device_str,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )
        
        '''
        # ===== 畫框並寫入影片 =====
        if results:
            vis_frame = results[0].plot()
        else:
            vis_frame = frame

        video_writer.write(vis_frame)
        '''

        # ===== OCR lat / lon（只在第一筆之前跑）=====
        ocr_lat = None
        ocr_lon = None
        raw_ocr = ""

        if not matched_once:
            for label, (x1, y1, x2, y2) in masks.items():
                if label not in {"lat", "lon"}:
                    continue

                roi = frame[y1:y2, x1:x2]
                text = " ".join(reader.readtext(roi, detail=0)).strip()
                raw_ocr += f"[{label}] {text} | "

                val = clean_latlon(text)
                if label == "lat":
                    ocr_lat = val
                else:
                    ocr_lon = val

        # ===== YOLO box 紀錄 =====
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                tid = int(box.id.item())
                cls_name = model.model.names[int(box.cls)]
                key = (tid, cls_name)

                if key not in track_map:
                    local_id = next_id
                    next_id += 1
                    track_map[key] = local_id

                    folder = os.path.join(out_dir, str(local_id))
                    os.makedirs(folder, exist_ok=True)

                    csv_file = open(os.path.join(folder, "data.csv"), "w", newline="")
                    writer = csv.writer(csv_file)
                    writer.writerow(["frame", "row", "column", "class"])

                    track_info[local_id] = {
                        "folder": folder,
                        "writer": writer,
                        "csv_file": csv_file,
                        "count": 0,
                        "class": cls_name,
                    }

                info = track_info[track_map[key]]

                x1, y1, x2, y2 = box.xyxy[0]
                xc = int((x1 + x2) // 2)
                yc = int((y1 + y2) // 2)

                cv2.imwrite(os.path.join(info["folder"], frame_name), frame)
                info["writer"].writerow([frame_name, yc, xc, info["class"]])
                info["count"] += 1

        # ===== 是否第一次成功偵測 lat/lon =====
        write_fps_now = False
        if (
            not matched_once
            and (ocr_lat is not None or ocr_lon is not None)
            and gnss_row is not None
        ):
            matched_once = True
            write_fps_now = True

        # ===== 組 ocr.csv row =====
        row = {
            "frame": frame_name,
            "fps": fps if write_fps_now and not fps_written else None,
            "lat_ocr": ocr_lat,
            "lon_ocr": ocr_lon,
            "date": gnss_row["date"] if write_fps_now else None,
            "time": gnss_row["time"] if write_fps_now else None,
            "lat": gnss_row["lat"] if write_fps_now else None,
            "lon": gnss_row["lon"] if write_fps_now else None,
            "raw_ocr": raw_ocr,
        }

        if write_fps_now:
            fps_written = True

        ocr_rows.append(row)
        pbar.update(1)

    cap.release()
    # video_writer.release()
    pbar.close()

    for info in track_info.values():
        info["csv_file"].close()
        if info["count"] < MIN_FRAME:
            shutil.rmtree(info["folder"])

    pd.DataFrame(ocr_rows).to_csv(
        os.path.join(out_dir, "ocr.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    renumber_detected_frame(out_dir)
    print(f"✔ 完成：{video_file}")

# ========= worker =========
def worker(video_list):
    device = "cuda:0"
    model = YOLO(WEIGHTS_PATH).to(device)
    reader = easyocr.Reader(["en"], gpu=True)
    masks = load_masks(MASK_JSON)

    for video in video_list:
        process_video(video, model, reader, masks, device)

# ========= 主程式 =========
def main():
    videos = [
        os.path.join(r, f)
        for r, _, fs in os.walk(VIDEO_ROOT)
        for f in fs if f.lower().endswith(".mp4")
    ]

    videos = sorted(videos)
    num_workers = 2
    video_chunks = [videos[i::num_workers] for i in range(num_workers)]

    processes = []
    for i in range(num_workers):
        p = Process(target=worker, args=(video_chunks[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
