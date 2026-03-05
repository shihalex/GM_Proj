import re
import sys
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# ===== 路徑（全部寫死）=====
ROOT = Path("/media/alex930307/TRANSCEND/GM/raw_video")
OUT_ROOT = Path("/media/alex930307/TRANSCEND/GM/video")

# ===== 旅程分群設定 =====
GAP_TOLERANCE_SEC = 30  # 相鄰段落時間差 ≤ 此值視為同一趟旅程

# ===== 顯示設定 =====
VERBOSE = True
SHOW_FFMPEG_CMD = False

# ===== 檔名格式（只處理 cam=F）=====
NAME_RE = re.compile(
    r'^(?P<event>[A-Za-z]+)(?P<date>\d{6})[-_](?P<time>\d{6})(?P<cam>[FRfr])\.(?P<ext>mp4|nmea)$',
    re.IGNORECASE
)

# ===== 環境檢查 =====
def ensure_bins():
    """確認 ffmpeg / ffprobe 可用"""
    for b in ("ffmpeg", "ffprobe"):
        subprocess.run(
            [b, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

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
    return datetime(
        2000 + int(yymmdd[:2]),
        int(yymmdd[2:4]),
        int(yymmdd[4:6]),
        int(hhmmss[:2]),
        int(hhmmss[2:4]),
        int(hhmmss[4:6])
    )

def ffprobe_duration_seconds(path: Path):
    """取得影片長度（秒）"""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ], text=True).strip()
        return float(out)
    except Exception:
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
    if len(parts) == 1:
        shutil.copy2(parts[0], out_path)
        return

    log_concat("Video", parts, out_path)
    list_txt = out_path.with_suffix(".concat.txt")
    with list_txt.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(list_txt),
        "-c", "copy",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
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

# ===== 處理 Event / Normal =====
def process_one_source(source_name: str):
    print(f"\n========== {source_name} ==========")
    base = ROOT / source_name
    out_base = OUT_ROOT / source_name

    F_dir = base / "F"
    NMEA_dir = base / "NMEA"

    sessions = defaultdict(lambda: {"F": [], "NMEA": []})

    # 收集 F 視訊
    for p, g in scan_dir(F_dir, {"mp4"}):
        key = f"{g['event']}{g['date']}_{g['time']}"
        sessions[(g["event"], key)]["F"].append(p)

    # 收集 NMEA（只要 F）
    for p, g in scan_dir(NMEA_dir, {"nmea"}):
        if g["cam"].upper() == "F":
            key = f"{g['event']}{g['date']}_{g['time']}"
            sessions[(g["event"], key)]["NMEA"].append(p)

    debug_list_unmatched(NMEA_dir, {"nmea"})

    # ===== 依時間分群為旅程 =====
    events = defaultdict(list)
    for (event, key), parts in sessions.items():
        m = re.match(r'^([A-Za-z]+)(\d{6})_(\d{6})$', key)
        start = yymmdd_hhmmss_to_datetime(m.group(2), m.group(3))
        dur = sum(ffprobe_duration_seconds(p) or 0 for p in parts["F"]) or 180
        end = start + timedelta(seconds=dur)
        events[event].append({
            "key": key,
            "start": start,
            "end": end,
            "parts": parts
        })

    # ===== 合併輸出 =====
    for event, sess in events.items():
        sess.sort(key=lambda x: x["start"])
        trips, cur = [], []

        for s in sess:
            if not cur or (s["start"] - cur[-1]["end"]).total_seconds() <= GAP_TOLERANCE_SEC:
                cur.append(s)
            else:
                trips.append(cur)
                cur = [s]
        if cur:
            trips.append(cur)

        for trip in trips:
            trip_name = trip[0]["key"].replace("_", "-")
            out_dir = out_base / trip_name
            out_dir.mkdir(parents=True, exist_ok=True)

            f_parts = sum((t["parts"]["F"] for t in trip), [])
            nmea_parts = sum((t["parts"]["NMEA"] for t in trip), [])

            print(f"\n=== 旅程 {trip_name} ===")
            if f_parts:
                concat_videos(f_parts, out_dir / f"{trip_name}.mp4")
            if nmea_parts:
                concat_text(nmea_parts, out_dir / f"{trip_name}.NMEA")

# ===== Main =====
def main():
    ensure_bins()
    for src in ("Event", "Normal"):
        process_one_source(src)

if __name__ == "__main__":
    main()
