# Road Sign Detection, Geolocation, and Automated Map Updating

## Demo Videos

### Detection Demo
[▶ Watch `Detection_Demo.mp4`](./Demo_Video/Detection_Demo.mp4)

### System Demo
[▶ Watch `System_Demo.mp4`](./Demo_Video/System_Demo.mp4)

---

## Overview

This project implements an end-to-end road-sign processing pipeline that combines video preprocessing, NMEA/GNSS parsing, YOLO-based road-sign detection and tracking, GNSS interpolation, image filtering, VGGT-based 3D reconstruction, COLMAP alignment, ray intersection, and final result export.

The pipeline is designed for road-sign mapping from forward-facing road video and GNSS/NMEA data. It performs the following major stages:

1. Merge temporally continuous video/NMEA segments into trips.
2. Convert NMEA records into CSV GNSS data.
3. Detect and track road signs with a trained YOLO model and BoT-SORT.
4. Match GNSS observations to video frames and interpolate coordinates.
5. Remove highly similar sign images using optical flow.
6. Downsample each sign track to at most 20 representative images.
7. Convert the retained data into the directory/data format required by VGGT.
8. Reconstruct each sign scene with VGGT and align the reconstruction to GNSS using COLMAP.
9. Estimate road-sign coordinates through multi-view ray intersection.
10. Merge all reconstructed sign results and export data for the web project.

## Repository Structure

```text
GM_Proj/
├── Demo_Video/
│   ├── Detection_Demo.mp4
│   └── System_Demo.mp4
├── Models/
│   ├── best.pt
│   └── botsort.yaml
├── Original_Videos/
├── Processed_Videos/
├── utils/
│   ├── video_process/
│   │   ├── video_combine.py
│   │   └── nmea2csv.py
│   ├── object_process/
│   │   ├── object_detection.py
│   │   ├── object_getlatlon.py
│   │   ├── object_clean.py
│   │   ├── object_upload.py
│   │   └── prepare_vggt.py
│   ├── vggt/
│   │   └── bash.sh
│   └── upload_process/
│       ├── combine_data.py
│       └── write_data.py
├── requirement.txt
└── README.md
```

All project scripts use relative paths such as `./Original_Videos/`, `./Processed_Videos/`, and `./Models/`. **Run all commands from the repository root** unless you intentionally modify these paths.

---

## Script Descriptions

| Script | Function |
| --- | --- |
| `utils/video_process/video_combine.py` | Groups temporally continuous front-camera video and NMEA segments into individual trips, merges MP4 files with FFmpeg, and concatenates NMEA records. |
| `utils/video_process/nmea2csv.py` | Parses each merged NMEA file, converts UTC timestamps to UTC+8, converts NMEA coordinates to decimal degrees, and exports trip-level GNSS CSV files. |
| `utils/object_process/object_detection.py` | Runs YOLO road-sign detection and BoT-SORT tracking, stores sign-track images, creates `detect.csv`/`detect.mp4`, and matches GNSS observations to the nearest frames. |
| `utils/object_process/object_getlatlon.py` | Marks original GNSS observations and linearly interpolates longitude/latitude between valid GNSS anchor frames. |
| `utils/object_process/object_clean.py` | Removes highly similar sign images using FAST features and Lucas-Kanade optical flow; sign folders with fewer than 10 images after cleaning are removed. |
| `utils/object_process/object_upload.py` | Limits each retained sign track to at most 20 images, preserving the first and last frames and uniformly sampling intermediate frames, then copies `detect.csv` and sign folders into `Processed_Videos/Upload/`. |
| `utils/object_process/prepare_vggt.py` | Converts each retained trip/sign into the VGGT scene format (`images/` + `data.csv`) without changing trip, sign, or image filenames. |
| `utils/vggt/bash.sh` | Runs VGGT reconstruction with bundle adjustment, creates a COLMAP sparse model, aligns it to GNSS, performs multi-view ray intersection, and writes per-sign geolocation/error results. |
| `utils/upload_process/combine_data.py` | Collects completed VGGT sign scenes, combines `marker_gnss_ray.csv` results, adds trip/sign metadata, and copies one representative image per sign. |
| `utils/upload_process/write_data.py` | Converts the combined result CSV into `data.json` and copies representative images into the web-data directory. |

---

## Requirements

### Platform

The current workflow is intended for Linux because it uses Bash, FFmpeg/FFprobe, the COLMAP command-line interface, and CUDA-enabled deep-learning tools.

A CUDA-capable NVIDIA GPU is strongly recommended for YOLO and VGGT.

### Conda Environment

The project was developed with **Python 3.11**.

```bash
conda create -n gm_proj python=3.11 -y
conda activate gm_proj
python -m pip install --upgrade pip
pip install -r requirement.txt
```

The included `requirement.txt` contains the Python packages used directly by the project and the packages required by the VGGT/COLMAP processing stage.

### System Dependencies

Install Git, FFmpeg/FFprobe, and COLMAP before running the pipeline. On Ubuntu/Debian systems:

```bash
sudo apt update
sudo apt install -y git ffmpeg colmap
```

Verify the commands are available:

```bash
git --version
ffmpeg -version
ffprobe -version
colmap -h
```

> PyTorch/CUDA compatibility depends on the installed NVIDIA driver and CUDA environment. If the PyTorch build in `requirement.txt` does not match your system, install the appropriate PyTorch build for your CUDA setup before running the GPU stages.

---

## Model Files

The object-detection stage expects:

```text
Models/
├── best.pt
└── botsort.yaml
```

- `best.pt`: trained road-sign YOLO weights.
- `botsort.yaml`: BoT-SORT tracker configuration.

`object_detection.py` reads these files through the relative paths `./Models/best.pt` and `./Models/botsort.yaml`.

---

## Input Data Format

Place the original forward-camera video and NMEA files under:

```text
Original_Videos/
└── Normal/
    ├── F/
    │   ├── FILE250101-120000F.mp4
    │   ├── FILE250101-120300F.mp4
    │   └── ...
    └── NMEA/
        ├── FILE250101-120000F.nmea
        ├── FILE250101-120300F.nmea
        └── ...
```

The filename pattern is:

```text
<event><YYMMDD>-<HHMMSS><camera>.<extension>
```

For this pipeline, use `FILE` as the event prefix and `F` as the front-camera suffix, for example:

```text
FILE250101-120000F.mp4
FILE250101-120000F.nmea
```

Using the `FILE` prefix is important because downstream scripts process trip directories in the form:

```text
FILEYYMMDD-HHMMSS
```

Video segments separated by no more than 30 seconds are grouped into the same trip by `video_combine.py`.

---

## Execution Order

The commands below are intended to be copied and executed from the **repository root**.

### 1. Combine video and NMEA segments

```bash
python utils/video_process/video_combine.py
```

Output examples:

```text
Processed_Videos/
└── FILE250101-120000/
    ├── FILE250101-120000.mp4
    └── FILE250101-120000.NMEA
```

### 2. Convert NMEA to GNSS CSV

```bash
python utils/video_process/nmea2csv.py
```

This creates a trip-level CSV beside the merged NMEA file.

### 3. Run road-sign detection and tracking

```bash
python utils/object_process/object_detection.py
```

Major outputs are stored under each trip's `frames/` directory, including `detect.csv`, `detect.mp4`, and per-track sign folders.

### 4. Interpolate GNSS coordinates

```bash
python utils/object_process/object_getlatlon.py
```

This adds the `Original` marker to original GNSS observations and linearly interpolates valid coordinates between GNSS anchor frames.

### 5. Remove highly similar sign images

```bash
python utils/object_process/object_clean.py
```

A sign track is retained only if at least 10 images remain after similarity filtering.

### 6. Prepare retained sign tracks for upload/VGGT conversion

```bash
python utils/object_process/object_upload.py
```

Each sign folder is limited to at most 20 images. The trip/sign naming structure and original image filenames are preserved.

### 7. Convert the retained data into VGGT scene format

```bash
python utils/object_process/prepare_vggt.py
```

The resulting structure is written under:

```text
Processed_Videos/Uploaded/
└── FILE250101-120000/
    └── frames/
        └── 1_SignClass/
            ├── images/
            └── data.csv
```

---

## Install VGGT Before the Reconstruction Stage

Before running `bash.sh`, clone the official VGGT repository from Meta/Facebook Research:

```bash
git clone https://github.com/facebookresearch/vggt.git VGGT
```

Install VGGT as an editable local package:

```bash
pip install -e ./VGGT
```

The official VGGT repository provides `demo_colmap.py`, which is used by this project's reconstruction script. The current `bash.sh` calls `python demo_colmap.py` from the repository root, so create a symbolic link:

```bash
ln -sf VGGT/demo_colmap.py ./demo_colmap.py
```

Confirm that it can be resolved:

```bash
python demo_colmap.py --help
```

VGGT's official `demo_colmap.py` expects each scene to contain an `images/` directory. `prepare_vggt.py` creates exactly this structure before reconstruction.

Official VGGT repository: <https://github.com/facebookresearch/vggt>

---

### 8. Run VGGT + COLMAP alignment + ray intersection

Make the script executable once:

```bash
chmod +x utils/vggt/bash.sh
```

Then run:

```bash
bash utils/vggt/bash.sh
```

For every trip/sign scene, the script attempts VGGT + bundle adjustment with `max_query_pts=4096`, retries with `2048` if necessary, aligns the COLMAP model to GNSS coordinates, and estimates the road-sign position using ray intersection.

Successful scenes are recorded in:

```text
Processed_Videos/Uploaded/finish.txt
```

Failed scenes are recorded in:

```text
Processed_Videos/Uploaded/error.txt
```

Representative per-scene outputs include:

```text
marker_gnss_ray.csv
ray_errors.csv
sparse/
sparse_aligned/
```

### 9. Combine all completed scene results

```bash
python utils/upload_process/combine_data.py
```

Outputs:

```text
Processed_Videos/Result/
├── marker_gnss_ray.csv
└── images/
```

The combined CSV uses `scene_id` in the form:

```text
FILE250101-120000/1_SignClass
```

to uniquely identify a sign scene across trips.

### 10. Generate web data

```bash
python utils/upload_process/write_data.py
```

Outputs:

```text
Processed_Videos/GM_Proj_Web/
├── data.json
└── images/
```

---

## Complete Copy-and-Run Workflow

After placing the input files and model files in the expected locations, the processing stages can be executed in this order:

```bash
# Create and activate the environment
conda create -n gm_proj python=3.11 -y
conda activate gm_proj
python -m pip install --upgrade pip
pip install -r requirement.txt

# System preprocessing and object-processing pipeline
python utils/video_process/video_combine.py
python utils/video_process/nmea2csv.py
python utils/object_process/object_detection.py
python utils/object_process/object_getlatlon.py
python utils/object_process/object_clean.py
python utils/object_process/object_upload.py
python utils/object_process/prepare_vggt.py

# Clone and install official VGGT before reconstruction
git clone https://github.com/facebookresearch/vggt.git VGGT
pip install -e ./VGGT
ln -sf VGGT/demo_colmap.py ./demo_colmap.py

# VGGT reconstruction, GNSS alignment, and ray intersection
chmod +x utils/vggt/bash.sh
bash utils/vggt/bash.sh

# Merge results and generate web data
python utils/upload_process/combine_data.py
python utils/upload_process/write_data.py
```

---

## Processing State Files

The pipeline records completed work so that subsequent executions can skip already processed items:

```text
Processed_Videos/video_combine.txt
Processed_Videos/nema2csv.txt
Processed_Videos/detect.txt
Processed_Videos/getlatlon.txt
Processed_Videos/clean.txt
Processed_Videos/upload.txt
Processed_Videos/Uploaded/finish.txt
Processed_Videos/Uploaded/error.txt
```

Deleting one of these files allows the corresponding stage to reconsider previously processed trips/scenes, but do this carefully because some stages also modify or delete intermediate files.

---

## Notes

- Run commands from the repository root because the current scripts use relative filesystem paths.
- The `FILEYYMMDD-HHMMSS` trip naming convention should be preserved across all processing stages.
- `object_clean.py` may permanently remove redundant images and sign folders from the processed trip data.
- `object_upload.py` may reduce retained sign tracks to at most 20 images.
- `prepare_vggt.py` builds the VGGT-ready copy under `Processed_Videos/Uploaded/`; VGGT outputs are also stored in each corresponding sign scene.
- `bash.sh` requires both the Python `pycolmap` package and the system `colmap` executable.
- The VGGT model weights may be downloaded automatically by the official VGGT code on first use, so an Internet connection may be required during the first reconstruction.

---

## External Components

This project uses or integrates with:

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO inference/tracking APIs.
- [VGGT](https://github.com/facebookresearch/vggt) for multi-view 3D reconstruction.
- [COLMAP](https://colmap.github.io/) for model alignment and reconstruction utilities.
- FFmpeg/FFprobe for video concatenation and metadata inspection.
