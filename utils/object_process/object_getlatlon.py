#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import pandas as pd


# ==================== 基本參數設定 ====================

DEFAULT_ROOT = Path("./Processed_Videos/")
DETECT_LOG_NAME = "detect.txt"
GETLATLON_LOG_NAME = "getlatlon.txt"


# ==================== 完成紀錄工具 ====================

def read_trip_log(log_path: Path) -> set:
    """
    讀取旅程完成紀錄。
    每一行代表一個旅程資料夾名稱。
    """
    if not log_path.exists():
        return set()

    completed = set()
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                name = line.strip()
                if name:
                    completed.add(name)
    except Exception as e:
        print(f"⚠️ 無法讀取紀錄檔: {log_path}\n   錯誤: {e}")

    return completed


def mark_trip_as_completed(log_path: Path, trip_name: str):
    """
    將成功完成經緯度處理的旅程加入 getlatlon.txt。
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{trip_name}\n")
    except Exception as e:
        raise RuntimeError(f"無法寫入完成紀錄 {log_path}: {e}")


# ==================== detect.csv 處理 ====================

def process_detect_csv(csv_path: Path) -> bool:
    """
    處理單一旅程的 frames/detect.csv。

    處理內容：
    1. 新增 / 重建 Original 欄位。
    2. 原本「經度」與「緯度」皆有資料的列：Original = 1
    3. 其他列：Original 保持空白
    4. 使用原始有效 GNSS 點進行線性內插。
    5. 僅內插兩個有效 GNSS 點之間的資料，不對頭尾進行外插。
    """
    print(f"▶ 處理: {csv_path}")

    if not csv_path.exists():
        print(f"⚠️ 找不到 detect.csv: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path, engine="python")
    except Exception as e:
        print(f"❌ 無法讀取 CSV: {csv_path}\n   錯誤: {e}")
        return False

    # 檢查必要欄位
    required_columns = ["經度", "緯度"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ 缺少必要欄位: {', '.join(missing_columns)}")
        return False

    if df.empty:
        print("⚠️️ detect.csv 為空")
        return False

    # 保存原始經緯度（無效轉 NaN）
    original_lon = pd.to_numeric(df["經度"], errors="coerce")
    original_lat = pd.to_numeric(df["緯度"], errors="coerce")

    # 必須經緯度同時有值，才視為原始有效點
    original_mask = original_lon.notna() & original_lat.notna()
    original_count = int(original_mask.sum())
    print(f"📍 原始有效 GNSS 點數: {original_count}")

    # 建立 Original 標記欄位
    original_column = pd.Series(pd.NA, index=df.index, dtype="Int64")
    original_column.loc[original_mask] = 1
    df["Original"] = original_column

    # 僅保留雙值有效的經緯度作為內插錨點
    df["經度"] = original_lon.where(original_mask)
    df["緯度"] = original_lat.where(original_mask)

    # 線性內插處理
    if original_count >= 2:
        df["經度"] = df["經度"].interpolate(method="linear", limit_area="inside")
        df["緯度"] = df["緯度"].interpolate(method="linear", limit_area="inside")

        interpolated_mask = ~original_mask & df["經度"].notna() & df["緯度"].notna()
        interpolated_count = int(interpolated_mask.sum())
        print(f"📐 線性內插完成: {interpolated_count} 筆")
    else:
        print("⚠️ 有效 GNSS 點少於 2 筆，無法進行線性內插。")

    # 寫回 CSV
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"❌ detect.csv 寫入失敗: {e}")
        return False

    print("✅ detect.csv 處理完成")
    return True


# ==================== 單一旅程處理 ====================

def process_trip(root: Path, trip_name: str) -> bool:
    trip_dir = root / trip_name
    print(f"\n{'=' * 70}\n🚗 處理旅程: {trip_name}")

    if not trip_dir.exists():
        print(f"⚠️ detect.txt 中存在旅程，但找不到資料夾:\n   {trip_dir}")
        return False

    if not trip_dir.is_dir():
        print(f"⚠️ 不是資料夾: {trip_dir}")
        return False

    detect_csv = trip_dir / "frames" / "detect.csv"
    if not detect_csv.exists():
        print(f"⚠️ 找不到 detect.csv: {detect_csv}")
        return False

    return process_detect_csv(detect_csv)


# ==================== 掃描與處理 ====================

def scan_and_process(root: Path):
    detect_log = root / DETECT_LOG_NAME
    getlatlon_log = root / GETLATLON_LOG_NAME

    print("=" * 70)
    print("GNSS 經緯度線性內插")
    print(f"{'=' * 70}\n📁 根目錄: {root}\n📄 YOLO 完成紀錄: {detect_log}\n📄 經緯度完成紀錄: {getlatlon_log}")

    if not detect_log.exists():
        print(f"\n❌ 找不到 detect.txt: {detect_log}")
        return

    detected_trips = read_trip_log(detect_log)
    processed_trips = read_trip_log(getlatlon_log)

    print(f"\n📋 detect.txt 已完成物件偵測: {len(detected_trips)} 個旅程")
    print(f"📋 getlatlon.txt 已完成經緯度處理: {len(processed_trips)} 個旅程")

    pending_trips = sorted(detected_trips - processed_trips)
    print(f"🔍 尚待經緯度處理: {len(pending_trips)} 個旅程")

    if not pending_trips:
        print("\n✅ 沒有需要處理的旅程。")
        return

    success_count = 0
    failed_count = 0
    total = len(pending_trips)

    for idx, trip_name in enumerate(pending_trips, start=1):
        print(f"\n[{idx}/{total}] {trip_name}")
        try:
            success = process_trip(root, trip_name)
            if success:
                mark_trip_as_completed(getlatlon_log, trip_name)
                processed_trips.add(trip_name)
                success_count += 1
                print(f"📝 已加入 getlatlon.txt: {trip_name}")
            else:
                failed_count += 1
                print(f"⚠️ 旅程未完成，不加入 getlatlon.txt: {trip_name}")
        except Exception as e:
            failed_count += 1
            print(f"❌ 處理旅程發生例外: {trip_name}\n   {e}")

    # 輸出統計摘要
    print(f"\n{'=' * 70}\n📊 處理結果\n{'=' * 70}")
    print(f"待處理旅程: {total}\n成功: {success_count}\n失敗: {failed_count}\n\n✅ 全部任務處理完成")


# ==================== 主程式 ====================

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT

    if not root.exists():
        print(f"❌ 根目錄不存在: {root}")
        sys.exit(1)

    scan_and_process(root)


if __name__ == "__main__":
    main()