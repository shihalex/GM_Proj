#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次轉換 video 目錄下所有 {trip_name}.NMEA → {trip_name}.csv

功能：
- 自動掃描 video/Event、video/Normal 各旅程資料夾
- 解析 NMEA（RMC/GGA/GSA/GSENSORD）
- 轉為結構化 CSV（時間已轉 UTC+8）
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ===== 路徑與設定（全部寫死）=====
VIDEO_ROOT = Path("/media/alex930307/TRANSCEND/GM/video")
ENCODING = "utf-8"

# ===== 工具函式 =====
def nmea_to_deg(raw, direction):
    """NMEA 座標（ddmm.mmmm / dddmm.mmmm）→ 十進位"""
    if not raw or "." not in raw:
        return None
    try:
        d_len = 2 if direction in ("N", "S") else 3
        deg = float(raw[:d_len]) + float(raw[d_len:]) / 60
        return -deg if direction in ("S", "W") else deg
    except Exception:
        return None

# ===== 處理單一 NMEA =====
def process_nmea(nmea_path: Path):
    """解析單一 .NMEA → 同名 .csv"""
    out_csv = nmea_path.with_suffix(".csv")
    records = []
    current = {}
    current_date = None

    with open(nmea_path, "r", encoding=ENCODING, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("$"):
                continue
            parts = line.split(",")

            # --- RMC：時間 / 座標 / 速度 ---
            if parts[0] == "$GNRMC" and len(parts) > 9:
                utc_time, status = parts[1], parts[2]
                lat = nmea_to_deg(parts[3], parts[4])
                lon = nmea_to_deg(parts[5], parts[6])
                speed = float(parts[7]) * 1.852 if parts[7] else None

                # 日期
                if parts[9]:
                    try:
                        current_date = datetime.strptime(parts[9], "%d%m%y")
                    except Exception:
                        continue

                # 時間（UTC+8）
                if utc_time and current_date:
                    try:
                        fmt = "%H%M%S.%f" if "." in utc_time else "%H%M%S"
                        t = datetime.strptime(utc_time, fmt)
                        dt = datetime.combine(
                            current_date.date(), t.time()
                        ) + timedelta(hours=8)
                        date_str = dt.strftime("%Y-%m-%d")
                        time_str = dt.strftime("%H:%M:%S.%f")[:-4]
                    except Exception:
                        date_str = time_str = None
                else:
                    date_str = time_str = None

                current = {
                    "date": date_str,
                    "time": time_str,
                    "status": status,
                    "lat": lat,
                    "lon": lon,
                    "speed_kmh": speed,
                    "course": parts[8],
                    "fix_quality": None,
                    "altitude": None,
                    "mode": None,
                    "pdop": None,
                    "hdop": None,
                    "vdop": None,
                    "x_a": None, "y_a": None, "z_a": None,
                    "x_g": None, "y_g": None, "z_g": None,
                }
                records.append(current)

            # --- GGA：定位品質 / 高度 ---
            elif parts[0] == "$GNGGA" and current:
                if len(parts) > 6:
                    current["fix_quality"] = parts[6]
                if len(parts) > 9:
                    current["altitude"] = parts[9]

            # --- GSA：DOP ---
            elif parts[0] == "$GNGSA" and current and len(parts) >= 17:
                current["mode"] = parts[2]
                current["pdop"] = parts[-3]
                current["hdop"] = parts[-2]
                current["vdop"] = parts[-1].split("*")[0]

            # --- GSENSORD：IMU ---
            elif parts[0] == "$GSENSORD" and current:
                try:
                    current["x_a"], current["y_a"], current["z_a"] = parts[1:4]
                    nums = re.findall(r"-?\d+\.\d+", ",".join(parts[4:]))
                    if len(nums) == 3:
                        current["x_g"], current["y_g"], current["z_g"] = nums
                except Exception:
                    pass

    if records:
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 輸出 {out_csv}")
    else:
        print(f"⚠️ 無有效資料：{nmea_path}")

# ===== Main =====
def main():
    nmeas = list(VIDEO_ROOT.glob("**/*.NMEA"))
    if not nmeas:
        print(f"{VIDEO_ROOT} 下找不到 .NMEA")
        return

    print(f"找到 {len(nmeas)} 個 .NMEA，開始轉換...\n")
    for i, p in enumerate(sorted(nmeas), 1):
        print(f"[{i}/{len(nmeas)}] {p}")
        try:
            process_nmea(p)
        except Exception as e:
            print(f"❌ 失敗：{p} ({e})")

if __name__ == "__main__":
    main()
