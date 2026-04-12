
"""
What this script does:
  Downloads each MP4 clip from Supabase Storage and extracts
  frames at regular intervals, saving them as JPG files in a
  local folder called 'extracted_frames/'.

  Run this FIRST, then run dino_classify.py second.

Usage:
  python scripts/extract_frames.py
"""

import os
import sys
import cv2
import urllib.request
import tempfile

# ── Locate project root and load .env ─────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_env(path):
    if not os.path.exists(path):
        print(f"ERROR: .env file not found at {path}")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env(os.path.join(PROJECT_ROOT, ".env"))

# ── Configuration ──────────────────────────────────────────────────────────────

SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "dive-videos")

DIVE_ID        = "4aa851b7-e583-4b2f-bb73-c185b97622d6"
CLIPS_PATH     = f"dives/{DIVE_ID}/clips"

# Extract 1 frame every N seconds
# Lower number = more frames (slower but more thorough)
# Higher number = fewer frames (faster but less coverage)
FRAME_EVERY_SEC = 2

# Where to save the extracted frames on your PC
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")

# ── Main ───────────────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path, output_dir, clip_name, every_n_sec=2):
    """
    Opens a video file and saves 1 frame every N seconds as a JPG.
    Returns list of saved file paths.
    """
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 10
    interval = max(1, int(fps * every_n_sec))

    saved_paths = []
    frame_idx   = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save every Nth frame
        if frame_idx % interval == 0:
            # e.g. clip_005_frame_0042.jpg
            clip_stem = clip_name.replace(".mp4", "")
            filename  = f"{clip_stem}_frame_{frame_idx:06d}.jpg"
            save_path = os.path.join(output_dir, filename)

            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved_paths.append(save_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_paths


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving frames to: {OUTPUT_DIR}\n")

    # List all clips
    print("Listing clips in Supabase Storage...")
    result = sb.storage.from_(STORAGE_BUCKET).list(CLIPS_PATH)
    clips  = sorted([item["name"] for item in result if item["name"].endswith(".mp4")])
    print(f"Found {len(clips)} clips\n")

    total_frames = 0
    temp_dir     = tempfile.mkdtemp(prefix="aquasight_extract_")

    for i, clip_name in enumerate(clips):
        print(f"[{i+1:02d}/{len(clips)}] {clip_name}")

        # Download clip
        storage_path = f"{CLIPS_PATH}/{clip_name}"
        url          = sb.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
        local_path   = os.path.join(temp_dir, clip_name)

        print(f"  Downloading...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"{size_mb:.1f} MB")
        except Exception as e:
            print(f"FAILED ({e}) — skipping")
            continue

        # Extract frames
        print(f"  Extracting frames (1 every {FRAME_EVERY_SEC}s)...", end=" ", flush=True)
        saved = extract_frames_from_video(local_path, OUTPUT_DIR, clip_name, FRAME_EVERY_SEC)
        print(f"{len(saved)} frames saved")
        total_frames += len(saved)

        # Delete downloaded clip to save disk space
        os.remove(local_path)

    print(f"\n✓ Done!")
    print(f"  Total frames extracted : {total_frames}")
    print(f"  Saved to               : {OUTPUT_DIR}")
   


if __name__ == "__main__":
    main()