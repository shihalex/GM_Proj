#!/usr/bin/env bash

set -e

ROOT="./Processed_Videos/Uploaded/"
FINISH_FILE="$ROOT/finish.txt"
ERROR_FILE="$ROOT/error.txt"

# ==================== 初始化 ====================
mkdir -p "$ROOT"
touch "$FINISH_FILE" "$ERROR_FILE"

# ==================== 逐旅程 ====================
for TRIP_DIR in "$ROOT"/*; do
    [ ! -d "$TRIP_DIR" ] && continue
    TRIP_NAME=$(basename "$TRIP_DIR")
    FRAMES_DIR="$TRIP_DIR/frames"
    [ ! -d "$FRAMES_DIR" ] && continue

    # ==================== 逐標誌 ====================
    for SCENE_DIR in "$FRAMES_DIR"/*; do
        [ ! -d "$SCENE_DIR" ] && continue
        SCENE_NAME=$(basename "$SCENE_DIR")

        # 標誌格式：{TrackID}_{ClassName}，驗證開頭為整數
        TRACK_ID="${SCENE_NAME%%_*}"
        [[ ! "$TRACK_ID" =~ ^[0-9]+$ ]] && continue

        SCENE_ID="${TRIP_NAME}/${SCENE_NAME}"

        # 狀態檢查：已完成或曾失敗則跳過
        if grep -Fxq "$SCENE_ID" "$FINISH_FILE"; then
            echo "⏩ 已完成，跳過：$SCENE_ID"
            continue
        fi

        if grep -Fxq "$SCENE_ID" "$ERROR_FILE"; then
            echo "⛔ 曾失敗，跳過：$SCENE_ID"
            continue
        fi

        echo -e "\n============================================================\n🚀 開始處理：$SCENE_ID\n============================================================"

        # 清除 GPU 暫存
        python -c "import torch; torch.cuda.empty_cache(); print('🧹 GPU cache cleared before scene')"

        # ==================== VGGT 執行函式 ====================
        run_vggt () {
            local MQP=$1
            echo "▶️ VGGT + BA (max_query_pts=$MQP)"
            python demo_colmap.py --scene_dir "$SCENE_DIR" --use_ba --max_query_pts "$MQP" --query_frame_num 4 --shared_camera
        }

        VGGT_OK=0

        # 嘗試 4096 查詢點
        set +e
        run_vggt 4096
        STATUS=$?
        set -e

        if [ $STATUS -eq 0 ] && [ -d "$SCENE_DIR/sparse" ]; then
            VGGT_OK=1
            echo "✅ VGGT 成功 (4096)"
        else
            echo "⚠️ VGGT 4096 失敗，嘗試 2048"
            python -c "import torch; torch.cuda.empty_cache(); print('🧹 GPU cache cleared before retry')"

            # 降階至 2048 查詢點重試
            set +e
            run_vggt 2048
            STATUS=$?
            set -e

            if [ $STATUS -eq 0 ] && [ -d "$SCENE_DIR/sparse" ]; then
                VGGT_OK=1
                echo "✅ VGGT 成功 (2048)"
            else
                echo "❌ VGGT 失敗：$SCENE_ID"
                echo "$SCENE_ID" >> "$ERROR_FILE"
                continue
            fi
        fi

        # ==================== 產出 images_gnss.txt ====================
        python << EOF
import pandas as pd
from pathlib import Path

scene = Path("$SCENE_DIR")
df = pd.read_csv(scene / "data.csv").dropna(subset=["lat", "lon"])
out_path = scene / "images_gnss.txt"

with open(out_path, "w") as f:
    for _, r in df.iterrows():
        f.write(f"{str(r['frame']).strip()} {float(r['lat'])} {float(r['lon'])} 0.0\n")

print(f"✅ images_gnss.txt: {len(df)} entries")
EOF

        # ==================== COLMAP GNSS Alignment ====================
        rm -rf "$SCENE_DIR/sparse_aligned"
        mkdir -p "$SCENE_DIR/sparse_aligned"

        set +e
        colmap model_aligner \
            --input_path "$SCENE_DIR/sparse" \
            --output_path "$SCENE_DIR/sparse_aligned" \
            --ref_images_path "$SCENE_DIR/images_gnss.txt" \
            --ref_is_gps 1 \
            --alignment_type ecef \
            --alignment_max_error 10
        ALIGN_STATUS=$?
        set -e

        if [ $ALIGN_STATUS -ne 0 ] || [ ! -f "$SCENE_DIR/sparse_aligned/cameras.bin" ]; then
            echo "❌ model_aligner 失敗：$SCENE_ID"
            echo "$SCENE_ID" >> "$ERROR_FILE"
            continue
        fi

        echo "✅ sparse_aligned created"

        # ==================== Ray Intersection (前方交會) ====================
        set +e
        python << EOF
import pycolmap
import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path

scene = Path("$SCENE_DIR")
recon = pycolmap.Reconstruction(scene / "sparse_aligned")
image_name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
df = pd.read_csv(scene / "data.csv").dropna(subset=["row", "column"])
ecef_to_gps = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

rays = []
for _, r in df.iterrows():
    frame = str(r["frame"]).strip()
    if frame not in image_name_to_id:
        continue

    img = recon.images[image_name_to_id[frame]]
    cam = recon.cameras[img.camera_id]
    u, v = float(r["column"]), float(r["row"])

    K = cam.calibration_matrix()
    x_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    x_cam /= np.linalg.norm(x_cam)

    Rcw = img.cam_from_world.rotation.matrix()
    C = img.cam_from_world.inverse().translation
    d = Rcw.T @ x_cam
    d /= np.linalg.norm(d)
    rays.append((C, d))

if len(rays) < 2:
    raise RuntimeError("射線不足，無法前方交會")

# 求解最小二乘交會點：sum((I - d d^T) (X - C)) = 0
A = np.zeros((3, 3))
b = np.zeros(3)
for C, d in rays:
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ C

X = np.linalg.solve(A, b)

# 計算各射線歐氏距離誤差
errors = np.array([np.linalg.norm((X - C) - np.dot(X - C, d) * d) for C, d in rays])
lon, lat, alt = ecef_to_gps.transform(X[0], X[1], X[2])

# 輸出統整指標與個別射線殘差
pd.DataFrame([{
    "lat": lat, "lon": lon, "alt": alt, "num_rays": len(errors),
    "mean_error_m": errors.mean(), "median_error_m": np.median(errors),
    "rms_error_m": np.sqrt((errors ** 2).mean()), "max_error_m": errors.max()
}]).to_csv(scene / "marker_gnss_ray.csv", index=False)

pd.DataFrame({"ray_id": np.arange(len(errors)), "error_m": errors}).to_csv(scene / "ray_errors.csv", index=False)
print("✅ Marker GNSS + ray errors saved")
EOF
        RAY_STATUS=$?
        set -e

        if [ $RAY_STATUS -ne 0 ]; then
            echo "❌ 射線前方交會失敗：$SCENE_ID"
            echo "$SCENE_ID" >> "$ERROR_FILE"
            continue
        fi

        # 標記處理完成
        echo "$SCENE_ID" >> "$FINISH_FILE"
        echo "🎉 完成：$SCENE_ID"
        python -c "import torch; torch.cuda.empty_cache()"
    done
done

echo -e "\n============================================================\n✅ 全部可處理標誌已完成\n============================================================"