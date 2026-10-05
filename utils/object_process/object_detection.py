import os
import re
import cv2
import csv
import shutil
import subprocess
import torch
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO


# ==================== 基本參數設定 ====================

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "999999999"

VIDEO_ROOT = Path("./Processed_Videos/")
DETECT_LOG_FILE = VIDEO_ROOT / "detect.txt"
WEIGHTS_PATH = Path("./Models/best.pt")

CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
TRACKER_TYPE = "./Models/botsort.yaml"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()

# Track 主類別至少需要出現 10 Frames 才會保留。
MIN_DETECTION_FRAMES = 10

# 原始標誌影像使用背景執行緒寫入。
MAX_IO_WORKERS = 12

# 旅程資料夾格式：FILEYYMMDD-HHMMSS
FOLDER_PATTERN = re.compile(r"^FILE\d{6}-\d{6}$", re.IGNORECASE)


# ==================== 旅程完成紀錄 ====================

def read_completed_trips() -> set:
    """讀取 detect.txt，取得所有已完成處理的旅程名稱。"""

    if not DETECT_LOG_FILE.exists():
        return set()

    completed = set()

    try:
        with open(DETECT_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                name = line.strip()

                if name:
                    completed.add(name)

    except Exception as e:
        print(f"⚠️ 讀取完成紀錄檔失敗: {DETECT_LOG_FILE} ({e})")

    return completed


def mark_trip_as_completed(folder_name: str):
    """將成功處理的旅程名稱加入 detect.txt。"""

    try:
        with open(DETECT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{folder_name}\n")

    except Exception as e:
        print(f"⚠️ 無法寫入完成紀錄: {e}")


# ==================== 時間工具 ====================

def parse_start_datetime(folder_name: str) -> datetime:
    """從 FILEYYMMDD-HHMMSS 解析旅程起始時間。"""

    clean_name = folder_name.upper().replace("FILE", "")

    try:
        return datetime.strptime(clean_name, "%y%m%d-%H%M%S")

    except ValueError:
        return datetime.now()


# ==================== 影像與檔名工具 ====================

def safe_image_write(file_path: Path, img: np.ndarray) -> bool:
    """安全寫入影像，供背景 Thread 使用。"""

    if img is None or img.size == 0:
        return False

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(file_path), img)

    except Exception as e:
        print(f"⚠️ 寫入影像失敗: {file_path} ({e})")
        return False


def sanitize_folder_name(name: str) -> str:
    """移除不適合作為 Linux / Windows 檔名的特殊字元。"""

    return re.sub(r'[\\/:*?"<>|]', "_", str(name)).strip()


def wait_for_futures(futures: list, description: str = ""):
    """等待所有背景影像寫入完成。"""

    if not futures:
        return

    if description:
        print(f"⏳ {description}")

    for future in futures:
        try:
            future.result()

        except Exception as e:
            print(f"⚠️ 背景寫入工作發生錯誤: {e}")


# ==================== 影片 Metadata ====================

def get_video_info(video_path: Path):
    """讀取影片 Frame 數、FPS 與解析度。"""

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        fps = 60.0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height
    }


# ==================== YOLO 輸出影片搜尋 ====================

def find_yolo_saved_video(save_dir: Path, source_video: Path):
    """取得 YOLO save=True 所產生的 Tracking 影片。"""

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    candidates = [
        p for p in save_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if not candidates:
        return None

    source_stem = source_video.stem.lower()

    for candidate in candidates:
        if candidate.stem.lower() == source_stem:
            return candidate

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


# ==================== 多段 YOLO 影片串接 ====================

def concat_yolo_videos(input_videos: list, output_path: Path) -> bool:
    """
    將 YOLO save=True 產生的影片整理為 detect.mp4。

    單一影片直接移動。
    多段影片使用 FFmpeg concat。
    """

    if len(input_videos) == 1:
        if output_path.exists():
            output_path.unlink()

        shutil.move(str(input_videos[0]), str(output_path))
        return True

    concat_list_path = output_path.parent / "_yolo_concat_list.txt"

    try:
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for video in input_videos:
                escaped_path = str(video.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        command = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            str(output_path)
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0

    except Exception as e:
        print(f"⚠️ FFmpeg 串接影片失敗: {e}")
        return False

    finally:
        if concat_list_path.exists():
            concat_list_path.unlink()


# ==================== detect.csv 儲存 ====================

def save_detection_records_to_csv(csv_path: Path, records: list):
    """將過濾後的偵測資料寫入 detect.csv。"""

    max_cols = max(len(r) for r in records)
    num_detections = max(0, (max_cols - 5) // 4)

    headers = ["名稱", "日期", "時間", "經度", "緯度"]

    for i in range(1, num_detections + 1):
        headers.extend([
            f"標誌名稱_{i}",
            f"ID_{i}",
            f"信心值_{i}",
            f"BBox_{i}"
        ])

    padded_records = [
        list(record) + [""] * (max_cols - len(record))
        for record in records
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(padded_records)


# ==================== GNSS 與 Frame 時間匹配 ====================

def match_gnss_to_nearest_frame(trip_dir: Path, folder_name: str, detect_csv_path: Path):
    """將每筆 GNSS 紀錄匹配至時間最接近的 Frame。"""

    gnss_csv_path = trip_dir / f"{folder_name}.csv"

    if not gnss_csv_path.exists():
        return

    try:
        gnss_df = pd.read_csv(gnss_csv_path, engine="python")
        detect_df = pd.read_csv(detect_csv_path, engine="python")

        lon_col = next((c for c in ["經度", "lon", "Longitude"] if c in gnss_df.columns), None)
        lat_col = next((c for c in ["緯度", "lat", "Latitude"] if c in gnss_df.columns), None)
        time_col = next((c for c in ["時間", "time", "Time"] if c in gnss_df.columns), None)
        date_col = next((c for c in ["日期", "date", "Date"] if c in gnss_df.columns), None)

        if not (lon_col and lat_col and time_col):
            print(f"⚠️ {gnss_csv_path.name} 缺少時間或經緯度欄位，略過定位同步。")
            return

        valid_gnss = gnss_df.dropna(subset=[lon_col, lat_col]).copy()

        if valid_gnss.empty or detect_df.empty:
            return

        if date_col:
            valid_gnss["_dt"] = pd.to_datetime(
                valid_gnss[date_col].astype(str) + " " + valid_gnss[time_col].astype(str),
                errors="coerce"
            )

        else:
            valid_gnss["_dt"] = pd.to_datetime(
                valid_gnss[time_col].astype(str),
                errors="coerce"
            )

        detect_df["_dt"] = pd.to_datetime(
            detect_df["日期"].astype(str) + " " + detect_df["時間"].astype(str),
            errors="coerce"
        )

        valid_gnss = valid_gnss.dropna(subset=["_dt"]).reset_index(drop=True)
        detect_df = detect_df.dropna(subset=["_dt"]).reset_index(drop=True)

        detect_df["經度"] = ""
        detect_df["緯度"] = ""

        frame_dts = detect_df["_dt"].values

        for _, gnss_row in valid_gnss.iterrows():
            gnss_dt = gnss_row["_dt"]
            time_diffs = np.abs(frame_dts - np.datetime64(gnss_dt))
            nearest_idx = int(np.argmin(time_diffs))

            detect_df.at[nearest_idx, "經度"] = gnss_row[lon_col]
            detect_df.at[nearest_idx, "緯度"] = gnss_row[lat_col]

        detect_df.drop(columns=["_dt"], inplace=True)
        detect_df.to_csv(detect_csv_path, index=False, encoding="utf-8-sig")

        print(f"📍 GNSS 紀錄匹配完成: {detect_csv_path}")

    except Exception as e:
        print(f"⚠️ GNSS 匹配失敗: {e}")


# ==================== Track 類別統計 ====================

def determine_valid_tracks(track_cls_counts: dict):
    """
    對每個 Track ID 執行類別多數決。

    dominant class 出現 >= MIN_DETECTION_FRAMES Frames 才保留。
    """

    valid_track_classes = {}
    track_summary = {}

    for track_id, cls_dict in track_cls_counts.items():
        dominant_cls, dominant_count = max(cls_dict.items(), key=lambda item: item[1])
        total_count = sum(cls_dict.values())

        track_summary[track_id] = {
            "dominant_cls": dominant_cls,
            "dominant_count": dominant_count,
            "total_count": total_count
        }

        if dominant_count >= MIN_DETECTION_FRAMES:
            valid_track_classes[track_id] = dominant_cls

    return valid_track_classes, track_summary


# ==================== 無效 Track 影像清理 ====================

def cleanup_invalid_track_images(track_dirs: dict, valid_track_classes: dict):
    """刪除未通過 Frame 門檻或非 dominant class 的 Track 影像資料夾。"""

    removed_dirs = 0
    kept_dirs = 0

    for (track_id, cls_name), folder_path in track_dirs.items():
        should_keep = (
            track_id in valid_track_classes
            and cls_name == valid_track_classes[track_id]
        )

        if should_keep:
            kept_dirs += 1
            continue

        if folder_path.exists():
            shutil.rmtree(folder_path)
            removed_dirs += 1

    print(
        f"🗑️ Track 影像清理完成："
        f"保留 {kept_dirs} 個資料夾，"
        f"刪除 {removed_dirs} 個資料夾。"
    )


# ==================== YOLO 暫存目錄清理 ====================

def cleanup_yolo_temp_dirs(temp_dirs: list):
    """刪除 YOLO save=True 所產生的暫存輸出資料夾。"""

    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ==================== 單一旅程處理 ====================

def process_trip(trip_dir: Path, model: YOLO, io_executor: ThreadPoolExecutor) -> bool:
    """
    處理單一旅程。

    1. YOLO stream=True 直接讀取影片。
    2. YOLO save=True 直接輸出 Tracking 影片。
    3. tqdm 顯示目前 Tracking 進度。
    4. 同時統計 Track ID、Class、Confidence 與 BBox。
    5. 暫時保存所有 Track 原始影像。
    6. 偵測完成後進行 Track 類別多數決。
    7. 刪除 < MIN_DETECTION_FRAMES 的 Track。
    8. 建立過濾後 detect.csv。
    9. 整理 YOLO 輸出為 detect.mp4。
    10. 執行 GNSS 時間同步。
    """

    folder_name = trip_dir.name
    videos = sorted(list(trip_dir.glob("*.mp4")) + list(trip_dir.glob("*.MP4")))

    if not videos:
        print(f"⚠️ {folder_name} 底下沒有影片。")
        return False

    # ==================== 輸出路徑 ====================

    frames_dir = trip_dir / "frames"
    detect_video_path = frames_dir / "detect.mp4"
    detect_csv_path = frames_dir / "detect.csv"

    frames_dir.mkdir(parents=True, exist_ok=True)

    if detect_video_path.exists():
        detect_video_path.unlink()

    base_datetime = parse_start_datetime(folder_name)

    # ==================== Metadata ====================

    trip_frames_meta = []
    track_cls_counts = {}
    track_dirs = {}

    image_write_futures = []
    yolo_saved_videos = []
    yolo_temp_dirs = []

    cumulative_seconds = 0.0
    global_frame_idx = 0

    # ==================== 逐影片執行 YOLO Tracking ====================

    for video_idx, video_path in enumerate(videos, start=1):
        video_info = get_video_info(video_path)

        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        width = video_info["width"]
        height = video_info["height"]

        print()
        print(f"🎬 開始 YOLO Tracking: {folder_name}")
        print(f"   影片: {video_path.name}")
        print(f"   總影格數: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   解析度: {width}x{height}")

        # ==================== YOLO 輸出目錄 ====================

        save_name = f"_yolo_tracking_{video_idx:03d}"
        yolo_save_dir = frames_dir / save_name

        if yolo_save_dir.exists():
            shutil.rmtree(yolo_save_dir)

        yolo_temp_dirs.append(yolo_save_dir)

        # ==================== YOLO Tracking ====================

        results = model.track(
            source=str(video_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_TYPE,
            device=DEVICE,
            half=USE_HALF,
            persist=True,
            stream=True,
            save=True,
            vid_stride=1,
            project=str(frames_dir),
            name=save_name,
            exist_ok=True,
            verbose=False
        )

        processed_frames = 0

        # ==================== YOLO Tracking 進度條 ====================

        pbar = tqdm(
            total=total_frames,
            desc=video_path.name,
            unit="frame",
            dynamic_ncols=True
        )

        try:
            # ==================== 逐 Frame 取得 Tracking 結果 ====================

            for frame_idx, result in enumerate(results):
                processed_frames += 1

                # ---------- Frame 時間 ----------

                elapsed_seconds = cumulative_seconds + frame_idx / fps
                current_time = base_datetime + timedelta(seconds=elapsed_seconds)

                date_str = current_time.strftime("%Y-%m-%d")
                time_str = current_time.strftime("%H:%M:%S.%f")[:-3]
                sub_sec = current_time.strftime("%f")[:2]

                frame_filename = f"FILE{current_time.strftime('%y%m%d-%H%M%S')}{sub_sec}.png"
                frame_detections = []

                # ---------- YOLO Tracking Detection ----------

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    written_tracks_this_frame = set()

                    for box in boxes:
                        if box.id is None:
                            continue

                        track_id = int(box.id.item())
                        cls_id = int(box.cls.item())
                        cls_name = model.names[cls_id]
                        conf = float(box.conf.item())

                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        cx = round((x1 + x2) / 2, 2)
                        cy = round((y1 + y2) / 2, 2)
                        w = round(x2 - x1, 2)
                        h = round(y2 - y1, 2)

                        bbox_info = f"[{cx}, {cy}, {w}, {h}]"

                        # ---------- Detection Metadata ----------

                        frame_detections.append({
                            "track_id": track_id,
                            "cls_name": cls_name,
                            "conf": conf,
                            "bbox_info": bbox_info
                        })

                        # ---------- Track 類別統計 ----------

                        if track_id not in track_cls_counts:
                            track_cls_counts[track_id] = {}

                        track_cls_counts[track_id][cls_name] = (
                            track_cls_counts[track_id].get(cls_name, 0) + 1
                        )

                        # ---------- Track 原始影像 ----------

                        track_frame_key = (track_id, cls_name)

                        if track_frame_key in written_tracks_this_frame:
                            continue

                        written_tracks_this_frame.add(track_frame_key)

                        safe_cname = sanitize_folder_name(cls_name)
                        sign_dir = frames_dir / f"{track_id}_{safe_cname}"

                        track_dirs[(track_id, cls_name)] = sign_dir

                        target_path = sign_dir / frame_filename
                        raw_frame = result.orig_img

                        if raw_frame is not None and raw_frame.size > 0:
                            future = io_executor.submit(
                                safe_image_write,
                                target_path,
                                raw_frame.copy()
                            )

                            image_write_futures.append(future)

                # ---------- Frame Metadata ----------

                trip_frames_meta.append({
                    "global_frame_idx": global_frame_idx,
                    "video_index": video_idx,
                    "video_name": video_path.name,
                    "video_frame_idx": frame_idx,
                    "filename": frame_filename,
                    "date": date_str,
                    "time": time_str,
                    "fps": fps,
                    "detections": frame_detections
                })

                global_frame_idx += 1

                # ---------- 更新進度條 ----------

                pbar.update(1)

        finally:
            pbar.close()

        # ==================== 單段影片 Tracking 完成 ====================

        cumulative_seconds += processed_frames / fps

        print(f"✅ YOLO Tracking 完成: {video_path.name}")
        print(f"   實際處理 Frames: {processed_frames}")

        # ==================== 取得 YOLO Tracking 影片 ====================

        saved_video = find_yolo_saved_video(yolo_save_dir, video_path)

        if saved_video is None:
            print(f"❌ 找不到 YOLO 輸出影片: {yolo_save_dir}")
            return False

        yolo_saved_videos.append(saved_video)

    # ==================== 等待 Track 原始影像寫入 ====================

    wait_for_futures(
        image_write_futures,
        description="等待所有 Track 原始影像寫入完成..."
    )

    # ==================== Track 類別多數決 ====================

    valid_track_classes, track_summary = determine_valid_tracks(track_cls_counts)

    print()
    print("=" * 70)
    print("📊 Track 統計結果")

    for track_id in sorted(track_summary.keys()):
        summary = track_summary[track_id]

        dominant_cls = summary["dominant_cls"]
        dominant_count = summary["dominant_count"]
        total_count = summary["total_count"]

        status = "保留" if track_id in valid_track_classes else "刪除"

        print(
            f"   ID {track_id}: "
            f"{dominant_cls} | "
            f"dominant={dominant_count} | "
            f"total={total_count} | "
            f"{status}"
        )

    print()
    print(f"📊 總共追蹤到 {len(track_cls_counts)} 個 ID")
    print(f"✅ >= {MIN_DETECTION_FRAMES} Frames：{len(valid_track_classes)} 個有效標誌")
    print(
        f"🗑️ < {MIN_DETECTION_FRAMES} Frames："
        f"{len(track_cls_counts) - len(valid_track_classes)} 個標誌將刪除"
    )

    # ==================== 刪除無效 Track 影像 ====================

    cleanup_invalid_track_images(
        track_dirs,
        valid_track_classes
    )

    # ==================== 建立過濾後 detect.csv ====================

    all_trip_records = []

    for frame_meta in trip_frames_meta:
        row_data = [
            frame_meta["filename"],
            frame_meta["date"],
            frame_meta["time"],
            None,
            None
        ]

        filtered_detections = [
            detection
            for detection in frame_meta["detections"]
            if (
                detection["track_id"] in valid_track_classes
                and detection["cls_name"] == valid_track_classes[detection["track_id"]]
            )
        ]

        for detection in filtered_detections:
            row_data.extend([
                detection["cls_name"],
                detection["track_id"],
                f'{detection["conf"]:.4f}',
                detection["bbox_info"]
            ])

        all_trip_records.append(row_data)

    save_detection_records_to_csv(
        detect_csv_path,
        all_trip_records
    )

    print(f"📝 detect.csv 輸出完成: {detect_csv_path}")

    # ==================== 整理 detect.mp4 ====================

    print("🎞️ 正在整理 YOLO Tracking 影片...")

    success = concat_yolo_videos(
        yolo_saved_videos,
        detect_video_path
    )

    if not success:
        print(f"❌ detect.mp4 建立失敗: {detect_video_path}")
        return False

    print(f"✅ detect.mp4 輸出完成: {detect_video_path}")

    # ==================== 清理 YOLO 暫存目錄 ====================

    cleanup_yolo_temp_dirs(yolo_temp_dirs)

    # ==================== GNSS 時間同步 ====================

    match_gnss_to_nearest_frame(
        trip_dir,
        folder_name,
        detect_csv_path
    )

    return True


# ==================== Tracker 狀態重置 ====================

def reset_model_tracker(model: YOLO):
    """每個旅程開始前重新初始化 Tracker 狀態。"""

    if model.predictor is None:
        return

    if not hasattr(model.predictor, "trackers"):
        return

    for tracker in model.predictor.trackers:
        tracker.reset()


# ==================== 主執行程序 ====================

def main():
    """掃描所有旅程並依序執行 YOLO Tracking。"""

    # ==================== 讀取旅程 ====================

    completed_trips = read_completed_trips()

    print(f"📋 已完成的旅程數量: {len(completed_trips)}")

    valid_trip_dirs = sorted([
        path
        for path in VIDEO_ROOT.iterdir()
        if path.is_dir() and FOLDER_PATTERN.match(path.name)
    ])

    print(f"🔍 共掃描到 {len(valid_trip_dirs)} 個目標旅程資料夾。")

    # ==================== 載入 YOLO ====================

    print()
    print(f"🚀 正在載入 YOLO 模型至 {DEVICE} (FP16: {USE_HALF})...")

    model = YOLO(str(WEIGHTS_PATH))
    io_executor = ThreadPoolExecutor(max_workers=MAX_IO_WORKERS)

    try:
        # ==================== 逐旅程處理 ====================

        for idx, trip_dir in enumerate(valid_trip_dirs, start=1):
            trip_name = trip_dir.name

            print()
            print("=" * 70)
            print(f"[{idx}/{len(valid_trip_dirs)}] 檢查旅程: {trip_name}")

            if trip_name in completed_trips:
                print(f"⏭️ 旅程已完成，略過: {trip_name}")
                continue

            reset_model_tracker(model)

            success = process_trip(
                trip_dir,
                model,
                io_executor
            )

            if success:
                mark_trip_as_completed(trip_name)
                completed_trips.add(trip_name)

                print(f"✅ 旅程完成: {trip_name}")

            else:
                print(f"⚠️ 旅程處理失敗: {trip_name}")

    finally:
        # ==================== 關閉 I/O ====================

        print()
        print("⏳ 等待背景影像寫入完成...")

        io_executor.shutdown(wait=True)

        # ==================== 釋放 GPU ====================

        print("🧹 釋放 GPU 模型資源...")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🎉 全部任務處理完成！")


# ==================== 程式進入點 ====================

if __name__ == "__main__":
    main()