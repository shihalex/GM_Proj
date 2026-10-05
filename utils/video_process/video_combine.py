import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("./Original_Videos/")
OUT_ROOT = Path("./Processed_Videos/")

# 只處理 ROOT 底下的 Normal
SOURCE_NAME = "Normal"

COMPLETED_FILE = OUT_ROOT / "video_combine.txt"

# 相鄰段落時間差 ≤ 30s 視為同一趟旅程
GAP_TOLERANCE_SEC = 30
VERBOSE = True
SHOW_FFMPEG_CMD = False

# ===== 檔名格式 =====
# Normal250101-120000F.mp4
# Normal250101-120000F.nmea
NAME_RE = re.compile(
    r"^(?P<event>[A-Za-z]+)"
    r"(?P<date>\d{6})"
    r"[-_]"
    r"(?P<time>\d{6})"
    r"(?P<cam>[FRfr])"
    r"\.(?P<ext>mp4|nmea)$",
    re.IGNORECASE,
)


def ensure_bins():
    """確認 ffmpeg / ffprobe 可用"""
    for b in ("ffmpeg", "ffprobe"):
        subprocess.run([b, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,)


def load_completed_trips():
    """
    讀取 video_combine.txt。

    每一行代表一個已經成功合併完成的 trip_name。
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not COMPLETED_FILE.exists():
        return set()

    completed = set()

    with COMPLETED_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            trip_name = line.strip()
            if trip_name:
                completed.add(trip_name)

    return completed


def mark_trip_completed(trip_name):
    """
    將成功完成的旅程寫入 combined_competed.txt。
    """
    with COMPLETED_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{trip_name}\n")
        f.flush()


# ===== 掃描資料夾（不遞迴）=====
def scan_dir(dirpath: Path, exts):
    items = []

    if not dirpath.exists():
        return items

    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            m = NAME_RE.match(p.name)
            if m:
                items.append((p, m.groupdict()))

    return items


def debug_list_unmatched(dirpath: Path, raw_exts):
    """列出未被採用的 NMEA（命名錯誤或非 F）"""
    print(f"\n[DEBUG] 未採用的 NMEA：{dirpath}")

    if not dirpath.exists():
        return

    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in raw_exts:
            m = NAME_RE.match(p.name)
            if not m or m.group("cam").upper() != "F":
                print(f"  - {p.name}")


# ===== 時間工具 =====
def yymmdd_hhmmss_to_datetime(yymmdd, hhmmss):
    return datetime(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]), int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6]))


def ffprobe_duration_seconds(path: Path):
    """取得影片長度（秒）"""
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path),], text=True,).strip()
        return float(out)

    except Exception as e:
        print(f"[WARNING] 無法取得影片長度：{path}")
        print(f"          {e}")
        return None


# ===== 合併工具 =====
def log_concat(label, parts, out_path):
    if VERBOSE:
        print(f"➜ 合併 {label}: {len(parts)} 段 → {out_path}")
        for p in parts:
            print(f"   • {p}")


def concat_videos(parts, out_path: Path):
    """使用 ffmpeg concat 合併 mp4"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 只有一段影片時直接複製
    if len(parts) == 1:
        log_concat("Video", parts, out_path)
        shutil.copy2(parts[0], out_path)
        return

    log_concat("Video", parts, out_path)
    list_txt = out_path.with_suffix(".concat.txt")

    try:
        with list_txt.open("w", encoding="utf-8") as f:
            for p in parts:
                # 避免單引號影響 ffmpeg concat 格式
                escaped_path = p.resolve().as_posix().replace("'", r"'\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out_path),]

        if SHOW_FFMPEG_CMD:
            print("[FFMPEG]", " ".join(cmd))

        subprocess.run(cmd, check=True)

    finally:
        # 不管成功或失敗都移除暫存 concat list
        list_txt.unlink(missing_ok=True)


def concat_text(parts, out_path: Path):
    """直接串接 NMEA 文字"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_concat("NMEA", parts, out_path)

    with open(out_path, "wb") as w:
        for i, p in enumerate(parts):
            with open(p, "rb") as r:
                data = r.read()

            if i > 0 and data and not data.startswith(b"\n"):
                w.write(b"\n")

            w.write(data)


# ===== 處理 Normal =====
def process_normal():
    print(f"\n========== {SOURCE_NAME} ==========")

    base = ROOT / SOURCE_NAME
    out_base = OUT_ROOT

    F_dir = base / "F"
    NMEA_dir = base / "NMEA"

    print(f"輸入資料夾：{base}")
    print(f"輸出資料夾：{out_base}")

    if not base.exists():
        raise FileNotFoundError(f"找不到 Normal 資料夾：{base}")

    # 一開始就讀取已完成紀錄
    completed_trips = load_completed_trips()
    print(f"已完成旅程紀錄數量：{len(completed_trips)}")

    sessions = defaultdict(lambda: {"F": [], "NMEA": []})

    # ===== 收集 F 視訊 =====
    for p, g in scan_dir(F_dir, {"mp4"}):
        # 只處理 F Camera
        if g["cam"].upper() != "F":
            continue

        key = f"{g['event']}{g['date']}_{g['time']}"
        sessions[(g["event"], key)]["F"].append(p)

    # ===== 收集 NMEA =====
    for p, g in scan_dir(NMEA_dir, {"nmea"}):
        # 只要 F Camera 的 NMEA
        if g["cam"].upper() == "F":
            key = f"{g['event']}{g['date']}_{g['time']}"
            sessions[(g["event"], key)]["NMEA"].append(p)

    debug_list_unmatched(NMEA_dir, {"nmea"})

    # ===== 依時間建立 session =====
    events = defaultdict(list)

    for (event, key), parts in sessions.items():
        m = re.match(r"^([A-Za-z]+)(\d{6})_(\d{6})$", key)

        if not m:
            print(f"[WARNING] 無法解析時間：{key}")
            continue

        start = yymmdd_hhmmss_to_datetime(m.group(2), m.group(3))

        # 使用所有 F 影片長度推算這個 session 的結束時間
        durations = []
        for p in parts["F"]:
            duration = ffprobe_duration_seconds(p)
            if duration is not None:
                durations.append(duration)

        # 原始邏輯：如果完全無法取得 duration，預設這一段為 180 秒。
        dur = sum(durations) if durations else 180
        end = start + timedelta(seconds=dur)

        events[event].append({"key": key, "start": start, "end": end, "parts": parts})

    # ===== 合併輸出 =====
    total_trips = 0
    skipped_trips = 0
    completed_now = 0

    for event, sess in events.items():
        sess.sort(key=lambda x: x["start"])

        trips = []
        cur = []

        for s in sess:
            if not cur:
                cur.append(s)
                continue

            gap = (s["start"] - cur[-1]["end"]).total_seconds()

            if gap <= GAP_TOLERANCE_SEC:
                cur.append(s)
            else:
                trips.append(cur)
                cur = [s]

        if cur:
            trips.append(cur)

        # ===== 處理每趟旅程 =====
        for trip in trips:
            total_trips += 1

            trip_name = trip[0]["key"].replace("_", "-")

            print(f"\n=== 旅程 {trip_name} ===")

            # 先檢查 combined_competed.txt
            if trip_name in completed_trips:
                print(f"⏭ 已完成，跳過：{trip_name}")
                skipped_trips += 1
                continue

            out_dir = out_base / trip_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # 按時間順序取得影片
            f_parts = []
            for t in trip:
                f_parts.extend(t["parts"]["F"])

            # 按時間順序取得 NMEA
            nmea_parts = []
            for t in trip:
                nmea_parts.extend(t["parts"]["NMEA"])

            try:
                # ===== 合併 MP4 =====
                if f_parts:
                    video_out = out_dir / f"{trip_name}.mp4"
                    concat_videos(f_parts, video_out)
                else:
                    print("[WARNING] 此旅程沒有 F 影片")

                # ===== 合併 NMEA =====
                if nmea_parts:
                    nmea_out = out_dir / f"{trip_name}.NMEA"
                    concat_text(nmea_parts, nmea_out)
                else:
                    print("[WARNING] 此旅程沒有 F NMEA")

                # 只有前面全部成功後，才加入 combined_competed.txt
                mark_trip_completed(trip_name)

                # 同一次執行中也更新 set，避免重複處理
                completed_trips.add(trip_name)
                completed_now += 1

                print(f"✓ 旅程完成：{trip_name}")

            except Exception as e:
                # 發生錯誤時不加入 completed，下次執行會重新嘗試。
                print(f"\n[ERROR] 旅程合併失敗：{trip_name}")
                print(f"        {e}")

                # 繼續處理下一趟，不因一趟失敗而整個程式停止
                continue

    print("\n========== 處理完成 ==========")
    print(f"找到旅程：{total_trips}")
    print(f"原本已完成並跳過：{skipped_trips}")
    print(f"本次新完成：{completed_now}")
    print(f"完成紀錄：{COMPLETED_FILE}")


def main():
    ensure_bins()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 只處理 Normal
    process_normal()


if __name__ == "__main__":
    main()