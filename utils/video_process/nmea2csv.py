import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

VIDEO_ROOT = Path("./Processed_Videos/")
COMPLETE_FILE = VIDEO_ROOT / "nema2csv.txt"
ENCODING = "utf-8"
CSV_COLUMNS = [
    "日期", "時間", "定位有效標籤", "緯度", "緯度標籤", "經度", "經度標籤", 
    "對地速度(km/h)", "對地航向", "定位品質", "衛星數目", "海拔高度", 
    "PDOP", "HDOP", "VDOP", "x_a", "y_a", "z_a", "x_g", "y_g", "z_g",
]

# 定義旅程名稱格式：FILE 開頭 + 6碼英數 + '-' + 6碼英數
# 若確定固定為純數字，可改為 r"^FILE\d{6}-\d{6}"
TRIP_PATTERN = re.compile(r"^FILE[A-Za-z0-9]{6}-[A-Za-z0-9]{6}", re.IGNORECASE)


def utc_to_utc8(utc_date: str, utc_time: str):
    if not utc_date or not utc_time:
        return None, None

    try:
        date_obj = datetime.strptime(utc_date, "%d%m%y")

        if "." in utc_time:
            time_obj = datetime.strptime(utc_time, "%H%M%S.%f")
        else:
            time_obj = datetime.strptime(utc_time, "%H%M%S")

        dt_utc = datetime.combine(date_obj.date(), time_obj.time())

        # UTC -> UTC+8
        dt_local = dt_utc + timedelta(hours=8)

        date_str = dt_local.strftime("%Y-%m-%d")

        # 保留至毫秒
        time_str = dt_local.strftime("%H:%M:%S.%f")[:-3]

        return date_str, time_str

    except Exception:
        return None, None


def nmea_coordinate_to_degree(raw, direction):
    if not raw or not direction:
        return None

    try:
        raw = raw.strip()
        direction = direction.strip().upper()

        if not raw:
            return None

        # 緯度 ddmm.mmmm
        if direction in ("N", "S"):
            if len(raw) < 4:
                return None
            degrees = float(raw[:2])
            minutes = float(raw[2:])

        # 經度 dddmm.mmmm
        elif direction in ("E", "W"):
            if len(raw) < 5:
                return None
            degrees = float(raw[:3])
            minutes = float(raw[3:])

        else:
            return None

        decimal_degree = degrees + minutes / 60.0

        if direction in ("S", "W"):
            decimal_degree *= -1

        return decimal_degree

    except (ValueError, TypeError):
        return None


def knot_to_kmh(speed):
    if speed is None:
        return None

    try:
        return float(speed) * 1.852

    except (ValueError, TypeError):
        return None


def remove_checksum(value):
    if value is None:
        return None

    return value.split("*")[0]


def empty_to_none(value):
    if value is None:
        return None

    value = value.strip()
    return value if value else None


def read_complete_file():
    if not COMPLETE_FILE.exists():
        return set()

    completed = set()

    try:
        with open(COMPLETE_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)

    except Exception as e:
        print(f"⚠️ 無法讀取完成紀錄：{COMPLETE_FILE} ({e})")

    return completed


def mark_trip_complete(trip_name):
    COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(COMPLETE_FILE, "a", encoding="utf-8") as f:
        f.write(f"{trip_name}\n")


def get_trip_name(nmea_path: Path):
    try:
        relative_path = nmea_path.relative_to(VIDEO_ROOT)

        if len(relative_path.parts) >= 2:
            return relative_path.parts[0]

        return nmea_path.stem

    except Exception:
        return nmea_path.parent.name


def is_valid_trip(trip_name: str) -> bool:
    """檢查旅程名稱是否符合 FILExxxxxx-xxxxxx 開頭的格式"""
    return bool(TRIP_PATTERN.match(trip_name))


# GSENSORD 解析
def parse_gsensord(line):
    """
    解析：
    $GSENSORD,x_a,y_a,z_a,[x_g,y_g,z_g][0]
    """
    result = {"x_a": None, "y_a": None, "z_a": None, "x_g": None, "y_g": None, "z_g": None}

    try:
        parts = line.split(",")

        # Accelerometer
        if len(parts) >= 4:
            result["x_a"] = empty_to_none(parts[1])
            result["y_a"] = empty_to_none(parts[2])
            result["z_a"] = empty_to_none(parts[3])

        # Gyroscope：只取第一組 [x_g,y_g,z_g]
        match = re.search(
            r"\[\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"\s*,\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"\s*,\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"\s*\]",
            line,
        )

        if match:
            result["x_g"] = match.group(1)
            result["y_g"] = match.group(2)
            result["z_g"] = match.group(3)

    except Exception:
        pass

    return result


# 處理單一 NMEA
def process_nmea(nmea_path: Path):
    """
    將單一 .NMEA 轉成同名 .csv。
    分組規則：每遇到一筆 $GNRMC，建立新的一列資料。
    """
    out_csv = nmea_path.with_suffix(".csv")
    records = []
    current = None
    gsa_extracted = False

    try:
        with open(nmea_path, "r", encoding=ENCODING, errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line.startswith("$"):
                    continue

                parts = line.split(",")
                if not parts:
                    continue

                sentence = parts[0]

                # GNRMC
                if sentence == "$GNRMC":
                    if len(parts) < 10:
                        continue

                    utc_time = empty_to_none(parts[1])
                    status = empty_to_none(parts[2])
                    latitude_raw = empty_to_none(parts[3])
                    latitude_direction = empty_to_none(parts[4])
                    longitude_raw = empty_to_none(parts[5])
                    longitude_direction = empty_to_none(parts[6])

                    # 原始速度單位為 knot
                    speed_knot = empty_to_none(parts[7])
                    course = empty_to_none(parts[8])
                    utc_date = empty_to_none(parts[9])

                    # UTC -> UTC+8
                    date_str, time_str = utc_to_utc8(utc_date, utc_time)

                    # 經緯度 -> 十進位度
                    latitude = nmea_coordinate_to_degree(latitude_raw, latitude_direction)
                    longitude = nmea_coordinate_to_degree(longitude_raw, longitude_direction)

                    # knot -> km/h
                    speed_kmh = knot_to_kmh(speed_knot)

                    # 建立新的一列
                    current = {
                        # RMC
                        "日期": date_str, "時間": time_str, "定位有效標籤": status, "緯度": latitude, "緯度標籤": latitude_direction,
                        "經度": longitude, "經度標籤": longitude_direction, "對地速度(km/h)": speed_kmh, "對地航向": course,
                        # GGA
                        "定位品質": None, "衛星數目": None, "海拔高度": None,
                        # GSA
                        "PDOP": None, "HDOP": None, "VDOP": None,
                        # GSENSORD
                        "x_a": None, "y_a": None, "z_a": None, "x_g": None, "y_g": None, "z_g": None,
                    }

                    records.append(current)
                    gsa_extracted = False

                # GNGGA
                elif sentence == "$GNGGA" and current is not None:
                    if len(parts) > 6:
                        current["定位品質"] = empty_to_none(parts[6])
                    if len(parts) > 7:
                        current["衛星數目"] = empty_to_none(parts[7])
                    if len(parts) > 9:
                        current["海拔高度"] = empty_to_none(parts[9])

                # GNGSA，每組只提取第一筆 GNGSA
                elif sentence == "$GNGSA" and current is not None and not gsa_extracted:
                    if len(parts) >= 18:
                        current["PDOP"] = empty_to_none(parts[15])
                        current["HDOP"] = empty_to_none(parts[16])
                        current["VDOP"] = empty_to_none(remove_checksum(parts[17]))
                        gsa_extracted = True

                # GSENSORD
                elif sentence == "$GSENSORD" and current is not None:
                    sensor = parse_gsensord(line)
                    current["x_a"] = sensor["x_a"]
                    current["y_a"] = sensor["y_a"]
                    current["z_a"] = sensor["z_a"]
                    current["x_g"] = sensor["x_g"]
                    current["y_g"] = sensor["y_g"]
                    current["z_g"] = sensor["z_g"]

    except Exception as e:
        print(f"❌ 讀取失敗：{nmea_path} ({e})")
        return False

    if not records:
        print(f"⚠️ 無有效 GNRMC 資料：{nmea_path}")
        return False

    try:
        df = pd.DataFrame(records, columns=CSV_COLUMNS)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"✅ 輸出：{out_csv}")
        print(f"   共 {len(df)} 筆資料")
        return True

    except Exception as e:
        print(f"❌ CSV 輸出失敗：{out_csv} ({e})")
        return False


# Main
def main():
    if not VIDEO_ROOT.exists():
        print(f"❌ 找不到資料夾：{VIDEO_ROOT}")
        return

    completed_trips = read_complete_file()
    print(f"已完成旅程紀錄：{len(completed_trips)} 個")

    nmeas = []
    for p in VIDEO_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".nmea":
            nmeas.append(p)

    if not nmeas:
        print(f"⚠️ {VIDEO_ROOT} 下找不到 .NMEA")
        return

    nmeas = sorted(nmeas)
    trips = {}
    ignored_count = 0

    # 建立旅程分組，並過濾旅程名稱
    for nmea_path in nmeas:
        trip_name = get_trip_name(nmea_path)

        # 檢查旅程名稱是否符合 FILExxxxxx-xxxxxx 格式
        if not is_valid_trip(trip_name):
            ignored_count += 1
            continue

        if trip_name not in trips:
            trips[trip_name] = []

        trips[trip_name].append(nmea_path)

    matched_nmeas_count = sum(len(v) for v in trips.values())
    print(f"掃描到 {len(nmeas)} 個 NMEA 檔案")
    if ignored_count > 0:
        print(f"ℹ️ 忽略非目標旅程檔案：{ignored_count} 個")
    print(f"符合目標格式的 NMEA：{matched_nmeas_count} 個，共 {len(trips)} 個旅程\n")

    if not trips:
        print("⚠️ 沒有找到符合命名規則的旅程資料。")
        return

    trip_items = sorted(trips.items(), key=lambda x: x[0])

    for trip_index, (trip_name, trip_nmeas) in enumerate(trip_items, start=1):
        print("=" * 70)
        print(f"[旅程 {trip_index}/{len(trip_items)}] {trip_name}")

        if trip_name in completed_trips:
            print(f"⏭️ 已完成，跳過：{trip_name}")
            continue

        print(f"包含 {len(trip_nmeas)} 個 NMEA")
        trip_success = True

        for file_index, nmea_path in enumerate(sorted(trip_nmeas), start=1):
            print(f"\n  [{file_index}/{len(trip_nmeas)}] {nmea_path.name}")
            success = process_nmea(nmea_path)

            if not success:
                trip_success = False

        if trip_success:
            try:
                mark_trip_complete(trip_name)
                completed_trips.add(trip_name)

                print(f"\n✅ 旅程完成：{trip_name}")
                print(f"   已寫入：{COMPLETE_FILE}")

            except Exception as e:
                print(f"\n⚠️ CSV 已完成，但無法寫入完成紀錄：{e}")

        else:
            print(f"\n❌ 旅程未完全成功：{trip_name}")
            print("   不寫入 nema2csv.txt")

    print("\n" + "=" * 70)
    print("NMEA -> CSV 處理完成")


if __name__ == "__main__":
    main()