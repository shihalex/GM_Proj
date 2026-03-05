import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_DIR = '/media/alex930307/TRANSCEND/GM/video/Normal/FILE251020-181307'
OUTPUT_FRAME_ROOT = './all_frames'
assert os.path.exists(OUTPUT_FRAME_ROOT), f"請先手動建立 {OUTPUT_FRAME_ROOT}"

video_files = sorted(glob.glob(f"{VIDEO_DIR}/*.mp4"))

def extract_frames(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(OUTPUT_FRAME_ROOT, base)
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, f"{base}_frame_%05d.jpg")

    # 加上 mpdecimate 過濾重複幀
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', 'fps=6,mpdecimate,setpts=N/FRAME_RATE/TB',
        '-q:v', '1',
        output_pattern
    ]

    print(f"開始：{video_path} → {frames_dir}", flush=True)
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"完成：{frames_dir}", flush=True)
    return video_path

if __name__ == "__main__":
    if not video_files:
        print("沒有找到任何 .mp4 檔案")
    else:
        num_workers = 8
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_frames, vp): vp for vp in video_files}
            for future in as_completed(futures):
                try:
                    video_path = future.result()
                    print(f"【完成】{video_path}")
                except Exception as e:
                    print(f"處理失敗：{futures[future]}，錯誤：{e}")
